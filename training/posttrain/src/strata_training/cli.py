import argparse
import json
from pathlib import Path

from strata_training.data.cache import prepare
from strata_training.eval.protocol import freeze
from strata_training.torch_api import api


def main() -> None:
    parser = argparse.ArgumentParser(prog="strata-train")
    parser.add_argument("--threads", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
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
    features.add_argument("--clip-length", type=int, default=1)
    features.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    features.add_argument("--samples", type=Path)
    baselines = sub.add_parser("baselines")
    for name in ("index", "protocol", "features", "output", "probe", "normalization"):
        baselines.add_argument(name, type=Path)
    baselines.add_argument("--calibration", type=Path)
    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("config", type=Path)
    training = sub.add_parser("train")
    training.add_argument("config", type=Path)
    prediction = sub.add_parser("predict")
    for name in ("index", "protocol", "checkpoint", "output"):
        prediction.add_argument(name, type=Path)
    prediction.add_argument("--export-frames", type=int, default=1)
    prediction.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    prediction.add_argument("--scale-mode", choices=("raw", "oracle-aligned"))
    prediction.add_argument("--features", type=Path)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("index", type=Path)
    evaluation.add_argument("protocol", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--checkpoint", type=Path, action="append", default=[])
    evaluation.add_argument("--predictions", type=Path, action="append", default=[])
    evaluation.add_argument("--baselines", type=Path)
    evaluation.add_argument("--export-frames", type=int, default=1)
    evaluation.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        api.set_num_threads(args.threads)
    if args.command == "targets":
        print(json.dumps({"frames": len(prepare(args.dataset, args.output, args.layers))}))
    elif args.command == "freeze":
        print(json.dumps(freeze(args.index, args.output)))
    elif args.command == "features":
        from strata_training.model.features import extract

        extract(args.index, args.output, args.preset, args.clip_length, args.device, args.samples)
    elif args.command == "baselines":
        from strata_training.baselines.prepare import prepare_baselines

        prepare_baselines(
            args.index,
            args.protocol,
            args.features,
            args.output,
            args.probe,
            args.normalization,
            args.calibration,
        )
    elif args.command == "pipeline":
        from strata_training.pipeline import pipeline

        pipeline(args.config)
    elif args.command == "train":
        from strata_training.train import train

        print(json.dumps(train(args.config)))
    elif args.command == "predict":
        from strata_training.eval.predict import cache_predictions

        print(
            json.dumps(
                cache_predictions(
                    args.index,
                    args.protocol,
                    args.checkpoint,
                    args.output,
                    args.export_frames,
                    args.device,
                    args.scale_mode,
                    args.features,
                )
            )
        )
    elif args.command == "evaluate":
        from strata_training.eval.runner import evaluate

        report = evaluate(
            args.index,
            args.protocol,
            args.output,
            args.checkpoint,
            args.export_frames,
            args.baselines,
            args.predictions,
            args.device,
        )
        print(json.dumps({"rows": len(report["summary"]), "frames": len(report["test_ids"])}))


if __name__ == "__main__":
    main()
