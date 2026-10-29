#!/usr/bin/env python3
"""C. Training-data coverage: sampled trajectories vs full Train-Seen.

Counts records/maps, then replays the teacher over all Train-Seen episodes to
get the full six-action distribution and trajectory-length statistics.  Also
reports the sampled token distribution from the existing 12,800-frame cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

AUDIT = Path(__file__).resolve().parents[1]
ROOT = AUDIT.parents[2]
sys.path.insert(0, str(ROOT))

from src.config import load_data_config  # noqa: E402
from src.data.actions import ACTION_NAMES  # noqa: E402
from src.data.citynav_dataset import CityNavDataset  # noqa: E402

SAMPLED_MAPS = (
    "birmingham_block_13", "birmingham_block_11", "birmingham_block_12",
    "birmingham_block_10", "cambridge_block_25", "cambridge_block_17",
    "cambridge_block_23", "cambridge_block_4",
)
SAMPLED_EPISODES = 625


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=AUDIT / "results/coverage_train_seen.json")
    args = ap.parse_args()

    data_config = load_data_config()
    dataset = CityNavDataset(data_config, "train_seen",
                             include_processed_descriptions=False)
    counts = Counter()
    lengths = []
    for record in dataset.records:
        result = record.teacher(dataset.teacher_config)
        counts.update(int(a) for a in result.actions)
        lengths.append(int(len(result.actions)))

    total = sum(counts.values())
    full_dist = {
        ACTION_NAMES[action]: round(counts[action] / total, 4)
        for action in range(6)
    }

    token_meta = json.loads(
        (ROOT / "data_cache/t5_frozen_1000/tokens/t5_tokens.json").read_text(
            encoding="utf-8"
        )
    )
    sampled = token_meta["action_counts"]["0"]
    sampled_total = sum(sampled.values())
    sampled_dist = {k: round(v / sampled_total, 4) for k, v in sampled.items()}

    summary = {
        "train_seen": {
            "records": len(dataset.records),
            "maps": len({r.map_name for r in dataset.records}),
            "teacher_actions_total": int(total),
            "trajectory_length": {
                "mean": round(float(np.mean(lengths)), 2),
                "median": int(np.median(lengths)),
                "p90": int(np.percentile(lengths, 90)),
                "max": int(np.max(lengths)),
            },
            "action_distribution": full_dist,
        },
        "sampled_token_cache": {
            "trajectories": 625,
            "windows": 1250,
            "frames": 10000,
            "maps": list(SAMPLED_MAPS),
            "record_coverage": round(SAMPLED_EPISODES / len(dataset.records), 4),
            "map_coverage": round(len(SAMPLED_MAPS) / len({r.map_name for r in dataset.records}), 4),
            "frames_per_trajectory": 16,
            "sampled_fraction_of_median_trajectory": round(
                16 / float(np.median(lengths)), 4
            ),
            "action_distribution": sampled_dist,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
