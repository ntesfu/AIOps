#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from aiops.features.assembly101_video_cache import merge_shard_indexes, relabel_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge completed ego30 cache shards.")
    parser.add_argument("--output-dir", default="data/processed/assembly101_ego30_stategraph")
    parser.add_argument("--manifest", default="data/raw/assembly101/ego30_manifest.json")
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Recompute targets after merging without rerunning visual extraction.",
    )
    args = parser.parse_args()
    result = merge_shard_indexes(args.output_dir, args.manifest)
    if args.relabel:
        result["relabel"] = relabel_cache(result["index"], args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
