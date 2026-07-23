"""Create sealed-test, participant-grouped IndustReal fold and episode manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiops.evaluation.industreal_grouped import (
    prepare_grouped_evaluation,
    write_grouped_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build participant-disjoint cross-validation folds and matched "
            "correct/incorrect causal episodes from an IndustReal cache. "
            "Official test records are always sealed."
        )
    )
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-radius", type=int, default=32)
    parser.add_argument("--matches-per-incorrect", type=int, default=3)
    parser.add_argument(
        "--operator-map",
        help="Optional JSON object mapping nonstandard recording IDs to operator IDs.",
    )
    args = parser.parse_args()

    operator_map = None
    if args.operator_map:
        operator_map = json.loads(Path(args.operator_map).read_text(encoding="utf-8"))
        if not isinstance(operator_map, dict):
            raise ValueError("--operator-map must contain a JSON object")
    payload = prepare_grouped_evaluation(
        args.cache_index,
        folds=args.folds,
        seed=args.seed,
        episode_radius=args.episode_radius,
        matches_per_incorrect=args.matches_per_incorrect,
        operator_map=operator_map,
    )
    write_grouped_evaluation(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "folds": payload["fold_count"],
                "development_recordings": payload["development_recordings"],
                "development_operators": len(payload["development_operators"]),
                "official_test_sealed": payload["policy"]["official_test_sealed"],
                "fold_summary": [
                    {
                        "fold": fold["fold"],
                        "validation_operators": len(fold["validation_operator_ids"]),
                        "validation_recordings": len(fold["validation_recording_ids"]),
                        "validation_incorrect_events": fold["validation"][
                            "incorrect_event_count"
                        ],
                        "validation_matched_pairs": fold["validation"][
                            "matched_pair_count"
                        ],
                        "validation_unmatched_incorrect": len(
                            fold["validation"]["unmatched_incorrect_episode_ids"]
                        ),
                    }
                    for fold in payload["folds"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
