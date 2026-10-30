#!/usr/bin/env python3
"""Data-lineage diagram for Stage 07 (from DATA_LINEAGE.md)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
W, H = 2.1, 1.25

BOXES = {
    "data":  (1.20, 6.60, "DATA/rgbd\n34 rasters · 28 dev maps\nRGB PNG + elevation TIF", "src"),
    "human": (1.20, 4.60, "Original human trajectories\n22,002 Train-Seen episodes", "src"),
    "s01":   (1.20, 2.40, "01 asset audit\nGROUND_LEVEL\nPLY only for assets", "done"),
    "s02":   (3.75, 5.70, "02 quality audit\n14,237 high / 4,284 med / 3,481 low\nanalysis labels only", "done"),
    "s03":   (3.75, 3.60, "03 path-following teacher\n19,105 / 22,002 replayable\nprojection of human paths", "old"),
    "s03b":  (6.30, 4.80, "03B direct teacher (current)\n22,002 / 2,498 / 2,826 all succeed\nstart + target only, replanned", "cur"),
    "s04":   (6.30, 2.50, "04 visual pipeline\n7x7x2048 · target 2048 · history 16\n12-d belief · candidates24", "done"),
    "s05":   (8.85, 4.80, "05 training (seed 42, 30 ep)\n22,002 ep / 876,398 states\n5.19 M draws · 40,590 updates", "cur"),
    "s06":   (11.10, 4.80, "06 closed loop\n5,324 tasks · SR 23.7%\n1,260 success", "cur"),
    "video": (11.10, 2.60, "06 videos\n12 success + 6 failure\nreal step records", "done"),
    "s07":   (8.85, 2.50, "07 audit (this stage)\nlineage · sampling · leakage\nfailure · literature", "cur"),
}

EDGES = [
    ("data", "s01", "-"), ("data", "human", "-"),
    ("human", "s02", "-"), ("human", "s03", "-"),
    ("human", "s03b", "-"), ("s03", "s03b", "--"),
    ("s03b", "s04", "-"), ("s03b", "s05", "-"),
    ("s04", "s05", "-"), ("s05", "s06", "-"),
    ("s06", "video", "-"), ("s06", "s07", "-"), ("s05", "s07", "-"),
]
STYLE = {"src": ("#eef3f7", "#33475b"), "done": ("#e8f1ea", "#33475b"),
         "cur": ("#e6eef9", "#1f4e79"), "old": ("#f2e6e6", "#8a4a4a")}


def edge_point(box, other):
    x, y = box[0], box[1]
    dx, dy = other[0] - x, other[1] - y
    sx = (W / 2) / abs(dx) if abs(dx) > 1e-9 else 1e9
    sy = (H / 2) / abs(dy) if abs(dy) > 1e-9 else 1e9
    s = min(sx, sy)
    return x + dx * s, y + dy * s


def main() -> int:
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(1.4, 7.5)
    ax.axis("off")
    for key, (x, y, text, kind) in BOXES.items():
        face, edge = STYLE[kind]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    linewidth=1.3, edgecolor=edge, facecolor=face))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.2,
                color="#7a4a4a" if kind == "old" else "#1d2b36")
    for a, b, ls in EDGES:
        pa = edge_point(BOXES[a], BOXES[b])
        pb = edge_point(BOXES[b], BOXES[a])
        ax.annotate("", xy=pb, xytext=pa,
                    arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#55697b",
                                "linestyle": ls, "shrinkA": 0, "shrinkB": 0})
    ax.text(6.2, 7.2, "CityNav-MEC fixed-50 m: data lineage (Stage 00-07)",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(3.75, 6.55, "02 labels never filtered training\n03 labels superseded by 03B",
            ha="center", fontsize=7.5, color="#8a3b3b")
    ax.text(11.1, 3.55, "rendered from stored\nstep records only", ha="center",
            fontsize=7, color="#55697b")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "data_lineage.png", dpi=160, bbox_inches="tight")
    print(f"wrote {FIG / 'data_lineage.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
