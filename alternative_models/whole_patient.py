"""One global patient loss, with bounded image activation memory.

First encode each image chunk without retaining its graph. Differentiate the
single patient loss through all embeddings and the unchanged owner pool. Then
replay each encoder chunk with its embedding gradient. This is the chain rule,
not an average of chunk losses. Only the caller may perform an optimizer step.
"""
import torch


def _device_indices(device):
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Whole-patient replay currently supports CPU or one CUDA device")
    return [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []


def backward_patient(model, chunks, load_chunk, target, loss_fn, device):
    """Populate head/encoder gradients for exactly one whole-patient BCE.

    ``load_chunk(index)`` must reproduce identical images on both passes.
    CPU and CUDA torch RNG states are replayed, including dropout/DropPath.
    Stateful batch normalization in training mode is refused: replay would
    update its running statistics twice. ConvNeXt V2/Swin V2 use LayerNorm.
    Embeddings, their gradients and RNG snapshots reside on CPU between passes;
    the global attention graph still scales with the patient's embedding count.
    """
    if not chunks or any(len(chunk) == 0 for chunk in chunks):
        raise ValueError("Nonempty patient chunks required")
    devices = _device_indices(device)
    for module in model.backbone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError("BatchNorm must be in eval mode during encoder replay")
    states, pieces = [], []
    buffers = {name: value.detach().clone() for name, value in model.backbone.named_buffers()}
    with torch.no_grad():
        for index, chunk in enumerate(chunks):
            images = load_chunk(index).to(device)
            if images.shape[0] != len(chunk):
                raise ValueError("Image chunk length differs from its tile list")
            states.append((torch.get_rng_state(), [torch.cuda.get_rng_state(i) for i in devices]))
            encoded = model.encode(images)
            if encoded.ndim != 2 or encoded.shape[0] != len(chunk) or not torch.isfinite(encoded).all():
                raise ValueError("Invalid/nonfinite patient embeddings")
            pieces.append(encoded.detach().cpu())
            del images, encoded
    for name, value in model.backbone.named_buffers():
        if not torch.equal(value, buffers[name]):
            raise ValueError(f"Encoder buffer changed during replay forward: {name}")
    full = torch.cat(pieces).to(device).requires_grad_(True)
    logits, attention = model.pool(full.unsqueeze(0))
    loss = loss_fn(logits, target.to(device))
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("Invalid/nonfinite whole-patient loss")
    loss.backward()
    gradients = full.grad.detach().cpu()
    result = dict(loss=loss.item(), logit=logits.detach().item(),
                  tiles=full.shape[0], chunks=len(chunks))
    del full, logits, attention, loss
    offset = 0
    for index, (chunk, expected, state) in enumerate(zip(chunks, pieces, states)):
        # Loading pixels may consume RNG; restore encoder RNG only after loading.
        with torch.random.fork_rng(devices=devices):
            images = load_chunk(index).to(device)
            torch.set_rng_state(state[0])
            for ordinal, value in zip(devices, state[1]):
                torch.cuda.set_rng_state(value, ordinal)
            encoded = model.encode(images)
            if not torch.allclose(encoded.detach().cpu(), expected, rtol=1e-5, atol=1e-6):
                raise ValueError("Encoder replay changed: images, RNG or model state differ")
            if encoded.requires_grad:
                torch.autograd.backward(encoded, gradients[offset:offset+len(chunk)].to(device))
            offset += len(chunk)
            del images, encoded
    return result


def finetune_parameters(model, backbone_name):
    """Freeze the encoder except its final stage and final normalization layers."""
    encoder = model.backbone
    encoder.requires_grad_(False)
    if backbone_name == "convnext_v2_t":
        modules = [encoder.stages[-1]]
        modules += [getattr(encoder, name) for name in ("norm_pre", "head") if hasattr(encoder, name)]
    elif backbone_name == "swin_v2_t":
        modules = [encoder.features[-1], encoder.norm]
    else:
        raise ValueError("Unsupported backbone for controlled fine-tuning")
    for module in modules:
        module.requires_grad_(True)
    selected = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    if not selected:
        raise ValueError("No final-stage encoder parameters selected")
    model.pool.requires_grad_(True)
    return selected


class Plateau:
    """Significant improvements reset patience; actual minima select checkpoints."""
    def __init__(self, patience, min_delta):
        if patience < 1 or min_delta < 0:
            raise ValueError("Positive patience and nonnegative min_delta required")
        self.patience, self.min_delta = patience, min_delta
        self.best, self.anchor, self.bad = float("inf"), float("inf"), 0

    def observe(self, loss):
        import math
        if not math.isfinite(loss):
            raise ValueError("Nonfinite development loss")
        improved = loss < self.best
        self.best = min(self.best, loss)
        if loss < self.anchor - self.min_delta:
            self.anchor, self.bad = loss, 0
        else:
            self.bad += 1
        return improved, self.bad >= self.patience
