"""Automated figure QA: detect ink touching any canvas edge.

Scans every PNG in paper/figures for non-white pixels within a guard band of
each edge — the programmatic version of "is anything cut off?". Run after
every figure regeneration; a clean set prints only OK lines.

Run:  python -X utf8 scripts/check_figures.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
BAND = 8          # pixels from each edge that must stay ink-free
THRESH = 245      # channel value below which a pixel counts as ink


def main():
    bad = 0
    for f in sorted(FIG.glob("*.png")):
        a = np.asarray(Image.open(f).convert("L"))
        h, w = a.shape
        edges = {
            "top": a[:BAND, :], "bottom": a[-BAND:, :],
            "left": a[:, :BAND], "right": a[:, -BAND:],
        }
        hits = {k: int((v < THRESH).sum()) for k, v in edges.items()
                if (v < THRESH).any()}
        if hits:
            bad += 1
            det = ", ".join(f"{k}: {n}px" for k, n in hits.items())
            print(f"  CUT-OFF RISK  {f.name:32s} {det}")
        else:
            print(f"  ok            {f.name}")
    print(f"\n{bad} figure(s) with edge ink" if bad else "\nall clean")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
