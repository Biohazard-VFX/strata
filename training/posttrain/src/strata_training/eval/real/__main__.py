import argparse
from pathlib import Path

from strata_training.eval.real.fetch import CATALOG, fetch


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen real-image coverage evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    get = commands.add_parser("fetch")
    get.add_argument("root", type=Path)
    get.add_argument("--datasets", nargs="+", choices=list(CATALOG), default=list(CATALOG))
    get.add_argument("--workers", type=int, default=8)
    score = commands.add_parser("score")
    score.add_argument("root", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("--predictions", type=Path)
    score.add_argument("--name")
    score.add_argument("--limit", type=int)
    score.add_argument("--frames-per-sequence", type=int)
    infer_command = commands.add_parser("infer-cpu")
    infer_command.add_argument("root", type=Path)
    infer_command.add_argument("output", type=Path)
    infer_command.add_argument("checkpoint", type=Path)
    infer_command.add_argument("curves", type=Path)
    infer_command.add_argument("--limit", type=int, required=True)
    infer_command.add_argument("--max-side", type=int, required=True)
    infer_command.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.command == "infer-cpu":
        import torch

        from strata_training.eval.real.infer import infer

        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
        print(
            infer(args.root, args.output, args.checkpoint, args.curves, args.limit, args.max_side)
        )
    elif args.command == "fetch":
        result = fetch(args.root, args.datasets, args.workers)
        for name, value in result.items():
            print(name, value["status"], value.get("error", value.get("blocked", "")))
        if any(value["status"] == "blocked" for value in result.values()):
            raise SystemExit(1)
    else:
        from strata_training.eval.real.runner import cached, evaluate, fake

        predictor = cached(args.predictions) if args.predictions else fake
        name = args.name or ("cached-alpha" if args.predictions else "fake-deep")
        result = evaluate(
            args.root, args.output, predictor, name, args.limit, args.frames_per_sequence
        )
        print(result["macro_mean"])


if __name__ == "__main__":
    main()
