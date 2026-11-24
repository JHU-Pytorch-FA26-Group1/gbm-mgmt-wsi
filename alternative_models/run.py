"""CLI entry point; invoke from the isolated repo root with python -m alternative_models.run."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/"data_prep"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--package", type=Path, required=True)
    prep.add_argument("--synthetic-inputs", type=Path)
    prep.add_argument("--fold", type=int, choices=range(5), required=True)
    prep.add_argument("--cap", type=int, required=True)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--out", type=Path, required=True)
    prep.set_defaults(repo=REPO)
    bench = commands.add_parser("benchmark")
    bench.add_argument("--backbone", choices=("swin_v2_t", "convnext_v2_t"), required=True)
    bench.add_argument("--bags", type=int, nargs="+", default=[256, 64])
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--device", default="cuda:0")
    bench.add_argument("--out", type=Path, required=True)
    bench.set_defaults(repo=REPO)
    encoder = commands.add_parser("cache")
    encoder.add_argument("--plan", type=Path, required=True)
    encoder.add_argument("--backbone", choices=("swin_v2_t", "convnext_v2_t"), required=True)
    encoder.add_argument("--reference-patch-id", required=True)
    encoder.add_argument("--device", default="cuda:0")
    encoder.add_argument("--chunk", type=int, default=16)
    encoder.add_argument("--out", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--arm", choices=("real", "resize", "crop"), required=True)
    train.add_argument("--bag-size", type=int, choices=(64, 128, 256), required=True)
    train.add_argument("--frozen-lr", type=float, required=True)
    train.add_argument("--finetune-lr", type=float, required=True)
    train.add_argument("--frozen-cycles", type=int, required=True)
    train.add_argument("--finetune-cycles", type=int, required=True)
    train.add_argument("--patience", type=int, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--chunk", type=int, default=16)
    train.add_argument("--out", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        from alternative_models.prepare import prepare
        prepare(args)
    elif args.command == "benchmark":
        from alternative_models.benchmark import benchmark
        benchmark(args)
    else:
        from alternative_models import engine
        getattr(engine, args.command)(args)


if __name__ == "__main__":
    main()
