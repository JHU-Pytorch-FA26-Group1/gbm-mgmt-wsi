"""Command line for the training pipeline: python -m training.run <command>.

    python -m training.run verify --package data/packages/B_qc-v5

Exit codes: 0 done, 1 problems found, 2 setup or usage problem.
"""
import argparse
import sys

import loader


def verify(package):
    """Rehash a package against its manifest and print any problems; return the exit code."""
    try:
        problems = loader.verify_package(package)
    except (OSError, ValueError, KeyError) as error:
        print(f"cannot verify {package}: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv=None):
    """Run the chosen command and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m training.run",
                                     description=__doc__.split("\n", maxsplit=1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    verify_parser = commands.add_parser("verify", help="rehash a package against its manifest")
    verify_parser.add_argument("--package", required=True, help="package or combined folder")
    args = parser.parse_args(argv)
    return verify(args.package)


if __name__ == "__main__":
    sys.exit(main())
