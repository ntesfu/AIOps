#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from aiops.features.assembly101_video_cache import merge_shard_indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge completed ego30 cache shards.")
    parser.add_argument("--output-dir", default="data/processed/assembly101_ego30_stategraph")
    parser.add_argument("--manifest", default="data/raw/assembly101/ego30_manifest.json")
    args = parser.parse_args()
    print(json.dumps(merge_shard_indexes(args.output_dir, args.manifest), indent=2))


if __name__ == "__main__":
    main()
