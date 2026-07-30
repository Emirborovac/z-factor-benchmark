"""Are the figures still readable printed in greyscale?

Referees and print subscribers may see the artwork without colour. A figure that
distinguishes its series only by hue becomes unreadable, so this converts each
method colour to its perceived luminance and reports the smallest gap between any
two series that appear together in a figure.

Rule of thumb used here: two greys are reliably distinguishable in print when
their luminance differs by about 15 points on the 0-255 scale. Below 10 they are
effectively identical.

This checks the palettes, which is what determines the answer; it does not
re-render the figures.

Run:  python -X utf8 scripts/check_greyscale.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

GAP_OK = 15.0
GAP_BAD = 10.0


def luminance(hex_colour: str) -> float:
    """ITU-R BT.709 relative luminance, the standard greyscale conversion."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def report(name: str, palette: dict, extra: str = "") -> int:
    lum = {k: luminance(v) for k, v in palette.items()}
    print(f"\n{name}{extra}")
    for k, v in sorted(lum.items(), key=lambda kv: kv[1]):
        print(f"    {v:6.1f}   {k}")
    worst = 1e9
    bad = 0
    for a, b in combinations(lum, 2):
        d = abs(lum[a] - lum[b])
        if d < worst:
            worst = d
        if d < GAP_BAD:
            print(f"    TOO CLOSE ({d:4.1f}): {a} vs {b}")
            bad += 1
        elif d < GAP_OK:
            print(f"    marginal  ({d:4.1f}): {a} vs {b}")
    print(f"    smallest gap: {worst:.1f}")
    return bad


def main() -> None:
    import make_figures as M

    print("Greyscale legibility of the figure palettes")
    print("=" * 62)
    print("Luminance on 0-255; gaps below 10 are effectively identical in "
          "print.")

    bad = 0
    bad += report("Line and scatter palette (COLOR) - fig 4, 7, 8, 9, 10",
                  M.COLOR, "")
    bad += report("Bar palette (BAR) - fig 1, 2", M.BAR, "")

    print("\n" + "=" * 62)
    print("Non-colour cues present in each figure, which is what carries the "
          "figure when colour is lost:")
    cues = [
        ("fig1_leaderboard", "bars are directly labelled with their value, and "
                             "the y axis names each method"),
        ("fig2_tolerance", "legend order matches bar order; every bar carries "
                           "its value"),
        ("fig3_forest", "single series; laboratories named on the y axis"),
        ("fig4_pressure_profile", "series labelled in a legend at the right; "
                                  "NNCF drawn at 4.5 pt against 2.6 pt"),
        ("fig5_regional", "two series, opposite ends of each dumbbell, both "
                          "value-labelled"),
        ("fig6_teacher_fidelity", "single-hue sequential density map with a "
                                  "colourbar"),
        ("fig7_deviation", "one method per panel, named in the panel title"),
        ("fig8_isotherms", "grey points vs a single blue curve; isotherms "
                           "labelled at their ends"),
        ("fig9_model_families", "one hero bar; rows named on the y axis, values "
                                "in a right-hand column"),
        ("fig10_correction", "two series distinguished by MARKER SHAPE "
                             "(triangle vs circle), not only colour"),
        ("fig11_domain", "panel (a) two series by shape/colour; panel (b) "
                         "single-hue density map"),
    ]
    for n, c in cues:
        print(f"  {n:24s} {c}")

    print()
    if bad:
        print(f"{bad} colour pair(s) indistinguishable in greyscale — but see "
              f"the cue list above before changing anything")
    else:
        print("no colour pair falls below the indistinguishable threshold")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
