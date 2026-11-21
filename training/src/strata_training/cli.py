import argparse
import json
from pathlib import Path

from strata_training.data.cache import prepare
from strata_training.data.convert import convert_v004
from strata_training.eval.protocol import freeze


def main() -> None:
    parser = argparse.ArgumentParser(prog="strata-train")
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert-v004")
    convert.add_argument("project", type=Path)
    convert.add_argument("output", type=Path)
    targets = sub.add_parser("targets")
    targets.add_argument("dataset", type=Path)
    targets.add_argument("output", type=Path)
    targets.add_argument("--layers", type=int, default=4)
    split = sub.add_parser("freeze")
    split.add_argument("index", type=Path)
    split.add_argument("output", type=Path)
    features = sub.add_parser("features")
    features.add_argument("index", type=Path)
    features.add_argument("output", type=Path)
    features.add_argument("--preset", choices=("depthpro2-s", "depthpro2-h"), default="depthpro2-s")
    training = sub.add_parser("train")
    training.add_argument("config", type=Path)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("index", type=Path)
    evaluation.add_argument("protocol", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--checkpoint", type=Path, action="append", default=[])
    evaluation.add_argument("--legacy-results", type=Path)
    evaluation.add_argument("--export-frames", type=int, default=1)
    args = parser.parse_args()
    if args.command == "convert-v004":
        print(convert_v004(args.project, args.output))
    elif args.command == "targets":
        print(json.dumps({"frames": len(prepare(args.dataset, args.output, args.layers))}))
    elif args.command == "freeze":
        print(json.dumps(freeze(args.index, args.output)))
    elif args.command == "features":
        from strata_training.model.features import extract

        extract(args.index, args.output, args.preset)
    elif args.command == "train":
        from strata_training.train import train

        print(json.dumps(train(args.config)))
    elif args.command == "evaluate":
        from strata_training.eval.runner import evaluate

        report = evaluate(
            args.index,
            args.protocol,
            args.output,
            args.checkpoint,
            args.legacy_results,
            args.export_frames,
        )
        print(json.dumps({"rows": len(report["summary"]), "frames": len(report["test_ids"])}))


if __name__ == "__main__":
    main()
