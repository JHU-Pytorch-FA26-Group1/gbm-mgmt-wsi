"""Bounded GPU memory benchmark, using synthetic tensors; not a model-quality result."""
import gc
import time

from alternative_models.engine import build, runtime, source_hashes
from alternative_models.prepare import write_json


def benchmark(args):
    import torch
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Benchmark requires the intended CUDA allocation")
    if args.out.exists():
        raise ValueError("Benchmark output exists; refusing overwrite")
    results = []
    for size in args.bags:
        if size < 1:
            raise ValueError("Bag sizes must be positive")
        model, optimizer, images, loss, logits, attention = None, None, None, None, None, None
        try:
            torch.manual_seed(args.seed)
            torch.cuda.empty_cache()
            model, _ = build(args.backbone, args.seed, pretrained=False)
            model.to(args.device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            images = torch.rand(1, size, 3, 224, 224, device=args.device)
            # Warmup allocates Adam state too; later steps measure steady-state work.
            measured = []
            torch.cuda.reset_peak_memory_stats(args.device)
            for step in range(4):
                torch.cuda.synchronize(args.device)
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                logits, attention = model(images)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(1, device=args.device))
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(args.device)
                if step:
                    measured.append(time.monotonic()-started)
            allocated = torch.cuda.max_memory_allocated(args.device)/2**30
            reserved = torch.cuda.max_memory_reserved(args.device)/2**30
            total = torch.cuda.get_device_properties(args.device).total_memory/2**30
            result = dict(bag_size=size, fits=True, mean_step_seconds=sum(measured)/len(measured),
                          peak_allocated_GiB=allocated, peak_reserved_GiB=reserved,
                          visible_memory_GiB=total,
                          reserved_fraction=reserved/total)
        except torch.cuda.OutOfMemoryError:
            result = dict(bag_size=size, fits=False, failure="CUDA out of memory")
        finally:
            model, optimizer, images, loss, logits, attention = None, None, None, None, None, None
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        print(result, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, dict(phase="memory-benchmark-only", valid_for_results=False,
                             backbone=args.backbone, fp32=True, optimizer="Adam",
                             random_weights=True, random_input_tensors=True,
                             runtime=runtime(torch), source_sha256=source_hashes(args.repo), results=results,
                             limitations="Excludes real image decoding/Macenko/I/O and attention over full frozen pools; use same bag size in every arm; no automatic training launch"))
