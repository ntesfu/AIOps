#!/usr/bin/env python3
"""Report whether a StateGraph cache contains usable auxiliary sensor rows."""

from __future__ import annotations

import argparse
import json

import numpy as np

from aiops.data.stategraph_cache import read_cache_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-index", required=True)
    args = parser.parse_args()
    _, records = read_cache_index(args.cache_index)
    total = present = nonzero = finite = 0
    for record in records:
        with np.load(record.path) as arrays:
            sensor = arrays["sensor"]
            mask = arrays["modality_mask"][:, -1].astype(bool)
            total += len(sensor)
            present += int(mask.sum())
            nonzero += int(np.any(sensor != 0, axis=1).sum())
            finite += int(np.isfinite(sensor).all(axis=1).sum())
    print(
        json.dumps(
            {
                "recordings": len(records),
                "rows": total,
                "sensor_present_rows": present,
                "sensor_present_fraction": present / max(total, 1),
                "sensor_nonzero_rows": nonzero,
                "sensor_nonzero_fraction": nonzero / max(total, 1),
                "sensor_finite_rows": finite,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
