"""Regression tests for training/train.py."""
import pytest
import torch
from torch import nn

from training import train


class Counting(torch.optim.SGD):
    """SGD that records the gradient it was about to apply at each step."""

    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.grads = []

    def step(self, closure=None):
        self.grads.append(self.param_groups[0]["params"][0].grad.clone())
        return super().step(closure)


class OneWeight(nn.Module):
    """Logit = w * (the bag's first value), attention uniform; w starts at 0."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, bags):
        """Logits and a dummy attention."""
        logits = self.weight * bags[:, 0, 0]
        return logits, torch.ones(len(bags), 1)


def batches(values, labels):
    """One-patient batches: bag of one tile holding `value`."""
    return [(torch.tensor([[[v]]]), torch.tensor([float(label)]))
            for v, label in zip(values, labels)]


def test_an_epoch_leaves_the_model_in_training_mode():
    """Dropout and stochastic depth only act in training mode, so an eval-mode model must switch."""
    model = OneWeight().eval()
    train.train_epoch(model, batches([1.0], [1]), Counting(model.parameters()),
                      train.make_loss(None), 1)
    assert model.training


def test_the_epoch_loss_is_the_mean_per_patient_not_per_batch():
    """A batch of two patients counts twice: (1 + 1 + 0) / 3, not (1 + 0) / 2."""
    model = OneWeight()
    two_methylated = (torch.ones(2, 1, 1), torch.ones(2))
    one_unmethylated = (torch.ones(1, 1, 1), torch.zeros(1))

    def label_mean(logits, labels):
        return logits.sum() * 0 + labels.mean()

    loss = train.train_epoch(model, [two_methylated, one_unmethylated],
                             Counting(model.parameters()), label_mean, 1)
    assert loss == pytest.approx(2 / 3)


def test_a_step_follows_every_group_of_patients_and_the_partial_last_group():
    """5 patients at 2 per update make 3 steps; the last, on one patient, is not half-weighted."""
    model = OneWeight()
    optimizer = Counting(model.parameters())
    data = batches([1.0] * 5, [1] * 5)
    train.train_epoch(model, data, optimizer, train.make_loss(None), 2)
    assert len(optimizer.grads) == 3
    assert torch.allclose(optimizer.grads[0], optimizer.grads[2])


def test_an_update_averages_its_patients_gradients():
    """Gradients from one update's patients are averaged, so the rate does not scale with them."""
    single, grouped = OneWeight(), OneWeight()
    one, many = Counting(single.parameters()), Counting(grouped.parameters())
    data = batches([1.0, 1.0], [1, 1])
    train.train_epoch(single, data[:1], one, train.make_loss(None), 1)
    train.train_epoch(grouped, data, many, train.make_loss(None), 2)
    assert torch.allclose(one.grads[0], many.grads[0])


def test_methylated_patients_weigh_the_class_weight_times_more():
    """The weight multiplies the methylated loss only."""
    logits = torch.zeros(1)
    plain, weighted = train.make_loss(None), train.make_loss(2.0)
    ones, zeros = torch.ones(1), torch.zeros(1)
    assert weighted(logits, ones).item() == pytest.approx(2 * plain(logits, ones).item())
    assert weighted(logits, zeros).item() == pytest.approx(plain(logits, zeros).item())


def run_stage(validation, patience):
    """fit_stage over scripted validation losses; the model weight records the epoch trained."""
    model = OneWeight()
    losses = iter(validation)

    def run_epoch(epoch):
        model.weight.data.fill_(float(epoch))
        return 0.0

    history, best = train.fit_stage(model, len(validation), patience, run_epoch,
                                    lambda: next(losses))
    return model, history, best


def test_training_stops_after_patience_and_returns_the_best_epoch_weights():
    """With patience 2, epochs 4 and 5 fail to beat epoch 3, so epoch 5 is the last run."""
    model, history, best = run_stage([5.0, 4.0, 3.0, 3.5, 3.2, 1.0, 0.5], patience=2)
    assert [row[0] for row in history] == [1, 2, 3, 4, 5]
    assert best == 3
    assert model.weight.item() == 3.0


def test_the_best_weights_are_handed_over_at_each_new_best_and_only_then():
    """on_best runs when an epoch beats the best so far, with the model as of that epoch."""
    model = OneWeight()
    saved = []
    losses = iter([5.0, 4.0, 4.5, 3.0, 3.0])

    def run_epoch(epoch):
        model.weight.data.fill_(float(epoch))
        return 0.0

    train.fit_stage(model, 5, None, run_epoch, lambda: next(losses),
                    on_best=lambda: saved.append(model.weight.item()))
    assert saved == [1.0, 2.0, 4.0]


def test_a_tie_keeps_the_earlier_epoch():
    """Equal validation loss is not an improvement, so the first of two equal epochs is kept."""
    model, _, best = run_stage([3.0, 3.0, 5.0], patience=None)
    assert best == 1 and model.weight.item() == 1.0


def test_a_stage_with_no_usable_validation_loss_is_an_error():
    """NaN losses never beat the best, so there is no epoch to restore."""
    with pytest.raises(RuntimeError, match="no epoch"):
        run_stage([float("nan")] * 3, patience=None)


def test_no_patience_runs_every_epoch():
    """A null patience trains for the whole stage and still returns the best epoch."""
    model, history, best = run_stage([3.0, 4.0, 5.0, 6.0], patience=None)
    assert len(history) == 4 and best == 1 and model.weight.item() == 1.0


def test_validation_loss_pairs_each_score_with_its_own_patients_label():
    """Rows come back in the order asked, not label order."""
    rows = [("b", 5.0, 0.99), ("a", -5.0, 0.01)]
    assert train.validation_loss(rows, {"a": 0, "b": 1}, train.make_loss(None)) < 0.01
    assert train.validation_loss(rows, {"a": 1, "b": 0}, train.make_loss(None)) > 4
