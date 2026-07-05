from __future__ import annotations

import argparse

from aiops.inference.pipeline import run_baseline_inference, write_json
from aiops.procedure import Procedure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIOps baseline inference on a video.")
    parser.add_argument("--video", required=True, help="Path to an input video.")
    parser.add_argument("--procedure", required=True, help="Path to a procedure JSON file.")
    parser.add_argument("--output", default="runs/predictions.json", help="Where to write predictions JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    procedure = Procedure.from_json(args.procedure)
    payload = run_baseline_inference(args.video, procedure)
    write_json(payload, args.output)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()

