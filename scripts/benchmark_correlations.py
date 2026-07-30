"""First benchmark: classical correlations vs the digitized Standing-Katz chart.

Each correlation is evaluated only inside its published validity range
(zfactor.correlations.VALIDITY). Metrics on train+val chart records.

Run:  python -X utf8 scripts/benchmark_correlations.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zfactor.correlations import CORRELATIONS, VALIDITY  # noqa: E402


def metrics(z_true, z_pred):
    err = z_pred - z_true
    ape = np.abs(err) / np.abs(z_true) * 100
    return {
        "n": len(z_true),
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAPE_%": np.mean(ape),
        "MaxAPE_%": np.max(ape),
        "bias": np.mean(err),
        "R2": 1 - np.sum(err**2) / np.sum((z_true - z_true.mean()) ** 2),
    }


def main():
    m = pd.read_parquet(ROOT / "data" / "processed" / "master.parquet")
    sk = m[(m.tier == "chart_digitized") & (m.quality_flag == "ok")]
    ppr, tpr, z = sk.Ppr.to_numpy(), sk.Tpr.to_numpy(), sk.z.to_numpy()

    rows = []
    for name, func in CORRELATIONS.items():
        t0, t1, p0, p1 = VALIDITY[name]
        mask = (tpr >= t0) & (tpr <= t1) & (ppr >= p0) & (ppr <= p1)
        pred = func(ppr[mask], tpr[mask])
        good = np.isfinite(pred)
        rows.append({"correlation": name, **metrics(z[mask][good], pred[good])})

    df = pd.DataFrame(rows).set_index("correlation").sort_values("MAPE_%")
    out = df.round(4)
    print(out.to_markdown())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "correlations_vs_sk_chart.md").write_text(
        "# Classical correlations vs Standing-Katz chart (within validity ranges)\n\n"
        + out.to_markdown(), encoding="utf-8")
    print("\nWrote reports/correlations_vs_sk_chart.md")


if __name__ == "__main__":
    main()
