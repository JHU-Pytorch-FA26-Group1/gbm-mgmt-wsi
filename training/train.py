"""Training: an epoch of weighted bag loss with patient-level steps, and a stage with early stop."""
import copy
import math

import torch


def make_loss(methylated_weight):
    """Binary cross-entropy on logits; methylated patients count `methylated_weight` times."""
    weight = None if methylated_weight is None else torch.tensor(methylated_weight)
    return torch.nn.BCEWithLogitsLoss(pos_weight=weight)


def train_epoch(model, batches, optimizer, loss_fn, patients_per_update):
    """Train on `batches` of (bags, labels); step every `patients_per_update` patients.

    Each patient's loss is divided by `patients_per_update`, so an update averages its patients.
    A final partial group still steps, with its gradients rescaled to an average over the patients
    it has. Returns the mean loss per patient.
    """
    model.train()
    device = next(model.parameters()).device
    total, seen, pending = 0.0, 0, 0
    optimizer.zero_grad()
    for bags, labels in batches:
        logits, _ = model(bags.to(device))
        loss = loss_fn(logits, labels.to(device))
        (loss * len(labels) / patients_per_update).backward()
        total += loss.item() * len(labels)
        seen += len(labels)
        pending += len(labels)
        if pending >= patients_per_update:
            optimizer.step()
            optimizer.zero_grad()
            pending = 0
    if pending:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad *= patients_per_update / pending
        optimizer.step()
        optimizer.zero_grad()
    return total / seen


def validation_loss(rows, labels, loss_fn):
    """Mean loss per patient over `evaluate.predict` rows, given {patient_id: label}."""
    logits = torch.tensor([logit for _, logit, _ in rows])
    targets = torch.tensor([float(labels[patient]) for patient, _, _ in rows])
    return loss_fn(logits, targets).item()


def fit_stage(model, epochs, patience, run_epoch, validate,  # pylint: disable=too-many-arguments
              *, on_best=None):
    """Run up to `epochs` epochs; stop after `patience` epochs without a better validation loss.

    `run_epoch(epoch)` trains and returns the training loss, `validate()` returns the validation
    loss. The model is left at its best epoch. Returns [(epoch, train_loss, validation_loss)] and
    the best epoch (1-based). Raises if no epoch gave a validation loss to select. `on_best()`, if
    given, runs each time an epoch sets a new best, so the best weights can be saved as training
    goes.
    """
    history, best_loss, best_epoch, best_state = [], math.inf, 0, None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(epoch)
        loss = validate()
        history.append((epoch, train_loss, loss))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            if on_best is not None:
                on_best()
        elif patience is not None and epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("no epoch gave a finite validation loss to select")
    model.load_state_dict(best_state)
    return history, best_epoch
