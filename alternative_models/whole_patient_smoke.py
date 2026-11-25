"""ConvNeXt-first mathematical and memory smoke; no patient data or downloads.

Run from the repo root: python -m alternative_models.whole_patient_smoke --help
Random weights/tensors test implementation only, never classification quality.
"""
import argparse
import copy
import gc
import os
from pathlib import Path
import time


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from training.models import build_model
    from alternative_models.engine import runtime, source_hashes
    from alternative_models.prepare import write_json
    from alternative_models.whole_patient import backward_patient, finetune_parameters
    if args.out.exists():
        raise ValueError("Smoke output already exists; refusing overwrite")
    if min(args.tiles, args.tile_chunk, args.reference_tiles) < 1:
        raise ValueError("Positive tile counts/chunk required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA allocation required; use the verified GPU Python environment")
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = dict(complete=False, valid_for_results=False, phase="whole-patient-smoke",
                  random_weights=True, random_images=True, backbone=args.backbone,
                  runtime=runtime(torch), source_sha256=source_hashes(Path(__file__).resolve().parents[1]),
                  settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    write_json(args.out, record)
    started = time.monotonic()
    try:
        # Same owner factory/pool. No pretrained download or data-loader imports.
        net = build_model({"attention_hidden": 128, "attention_dropout": 0.0}, args.backbone, pretrained=False).to(args.device).eval()
        finetune_parameters(net, args.backbone)
        reference = copy.deepcopy(net)
        images = torch.randn(args.reference_tiles, 3, 224, 224)
        reference_chunk = max(1, args.reference_tiles // 2)
        chunks = list(images.split(reference_chunk))
        target = torch.ones(1, device=args.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.7, device=args.device))
        logits, reference_attention = reference(images.unsqueeze(0).to(args.device))
        expected_loss = loss_fn(logits, target)
        expected_loss.backward()
        result = backward_patient(net, chunks, lambda i: chunks[i], target, loss_fn, args.device)
        torch.testing.assert_close(torch.tensor(result["loss"]), expected_loss.detach().cpu(), rtol=1e-4, atol=1e-5)
        max_error = 0.0
        compared = 0
        for (name, first), (_, second) in zip(reference.named_parameters(), net.named_parameters()):
            if not first.requires_grad:
                if first.grad is not None or second.grad is not None:
                    raise ValueError(f"Frozen parameter received a gradient: {name}")
                continue
            torch.testing.assert_close(first.grad, second.grad, rtol=1e-3, atol=1e-5, msg=name)
            max_error = max(max_error, (first.grad-second.grad).abs().max().item())
            compared += 1
        record["equivalence"] = dict(passed=True, reference_tiles=args.reference_tiles,
                    reference_chunk=reference_chunk, tensors_compared=compared,
                    loss=result["loss"], max_gradient_absolute_error=max_error,
                    gradient_rtol=.001, gradient_atol=.00001,
                    encoder_mode="eval; CPU tests additionally verify dropout RNG replay")
        print("Whole-patient loss and gradient equivalence: PASSED", flush=True)
        del reference, images, chunks, logits, reference_attention, expected_loss, first, second
        net.zero_grad(set_to_none=True)
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(args.device)
        # First update includes Adam-state allocation. A partial final chunk is retained.
        net.train()
        optimizer = torch.optim.Adam([
            {"params": [p for p in net.backbone.parameters() if p.requires_grad], "lr": 1e-6},
            {"params": net.pool.parameters(), "lr": 1e-5}])
        indexes = [list(range(start, min(start+args.tile_chunk, args.tiles))) for start in range(0, args.tiles, args.tile_chunk)]
        def load(index):
            generator = torch.Generator().manual_seed(args.seed+index)
            return torch.randn(len(indexes[index]), 3, 224, 224, generator=generator)
        result = backward_patient(net, indexes, load, target, loss_fn, args.device)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise ValueError("Nonfinite capacity-test gradients")
        optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        record["capacity"] = dict(passed=True, tiles=args.tiles, tile_chunk=args.tile_chunk,
                    chunks=len(indexes), final_chunk=len(indexes[-1]), patient_losses=1, optimizer_updates=1,
                    loss=result["loss"])
        if args.device.startswith("cuda"):
            record["capacity"].update(
                peak_allocated_GiB=torch.cuda.max_memory_allocated(args.device)/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved(args.device)/2**30,
                visible_memory_GiB=torch.cuda.get_device_properties(args.device).total_memory/2**30)
        record.update(complete=True, elapsed_seconds=time.monotonic()-started,
                      limitations="Random tensors; excludes real I/O, Macenko, full real patient counts, accuracy and pretrained checkpoint verification")
        write_json(args.out, record)
        print(record["capacity"], flush=True)
        print(f"Smoke complete: {args.out}", flush=True)
    except Exception as exc:
        record.update(failure=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
        write_json(args.out, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("convnext_v2_t", "swin_v2_t"), default="convnext_v2_t")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tile-chunk", type=int, default=128)
    parser.add_argument("--tiles", type=int, default=257)
    parser.add_argument("--reference-tiles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
