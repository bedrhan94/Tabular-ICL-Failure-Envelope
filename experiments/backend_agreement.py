"""Agreement between the two TabPFN backends, metric by metric.

Referee 2, point 5: section 5.3 validates the cloud-to-local backend switch with
a single per-cell rho correlation of 0.98. rho is a function of the failure rule,
so it can agree while the calibration channel underneath it does not -- and
calibration is the axis the paper's argument runs on. This reports the agreement
the rho number does not cover.

The comparison is restricted to conditions both arms actually ran *and* on which
both carry a reference. The second restriction matters: an arm scored without a
gradient-boosted reference is scored under two criteria instead of three, which
is the error section 5.3 retracts, and comparing across that boundary would
reintroduce it.

The calibration slope is reported on medians and Spearman rather than Pearson:
it is an unbounded ratio with a heavy tail, so a handful of cells with slopes in
the hundreds dominate a Pearson coefficient while rank agreement is unaffected.

Usage::

    python experiments/backend_agreement.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CELL = ["dataset_id", "shift_axis", "shift_lambda", "base_seed"]

# calibration first: it is the channel the referee asks about. auc and utility
# follow as context, so a calibration-only disagreement would stand out.
METRICS = [
    ("ece", "ECE (15 equal-width bins)"),
    ("ece_adaptive", "ECE (equal-mass bins)"),
    ("ece_classwise", "ECE (classwise)"),
    ("brier", "Brier score"),
    ("nll_norm", "NLL / log K"),
    ("auc", "AUC"),
    ("utility", "utility U"),
]
HEAVY_TAILED = [("calib_slope", "calibration slope")]

_DEFAULT = (
    _ROOT
    / "results"
    / "external"
    / "tables_2axis_multiseed_primary_with_tabpfn_local"
    / "shift_results.csv"
)


def _paired(df: pd.DataFrame, local: str, cloud: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = df[(df.model == local) & (df.status == "ok")].set_index(CELL).sort_index()
    b = df[(df.model == cloud) & (df.status == "ok")].set_index(CELL).sort_index()
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
    both_ref = a["reference_utility"].notna() & b["reference_utility"].notna()
    if not both_ref.all():
        dropped = int((~both_ref).sum())
        print(f"[backend] dropping {dropped} cells where one arm carries no reference")
    return a[both_ref], b[both_ref]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=_DEFAULT)
    p.add_argument("--local", default="tabpfn")
    p.add_argument("--cloud", default="tabpfn_client")
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "ablations" / "backend_agreement")
    args = p.parse_args(argv)

    if not args.results.exists():
        print(f"[backend] missing {args.results}")
        return 1
    df = pd.read_csv(args.results)
    a, b = _paired(df, args.local, args.cloud)
    n_total = df[df.model == args.cloud].shape[0]
    print(
        f"[backend] {len(a)} paired conditions of the cloud arm's {n_total} "
        f"({len(a) / n_total:.0%} coverage; the remainder is the daily quota cut)"
    )

    rows = []
    for col, label in METRICS + HEAVY_TAILED:
        x, y = a[col].astype(float), b[col].astype(float)
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        if x.empty:
            continue
        try:
            wp = float(stats.wilcoxon(x, y).pvalue)
        except ValueError:
            wp = float("nan")
        rows.append(
            {
                "metric": label,
                "n": int(len(x)),
                "mean_local": x.mean(),
                "mean_cloud": y.mean(),
                "median_local": x.median(),
                "median_cloud": y.median(),
                "mean_abs_diff": (x - y).abs().mean(),
                "median_abs_diff": (x - y).abs().median(),
                "pearson": stats.pearsonr(x, y)[0],
                "spearman": stats.spearmanr(x, y)[0],
                "wilcoxon_p": wp,
                "heavy_tailed": col in dict(HEAVY_TAILED),
            }
        )

    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "backend_agreement.csv", index=False)

    print()
    hdr = f"{'metric':26s} {'local':>8s} {'cloud':>8s} {'MAD':>8s} {'Pearson':>8s} {'Spearman':>9s} {'p':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        star = " *" if r["heavy_tailed"] else ""
        print(
            f"{r['metric']:26s} {r['mean_local']:8.4f} {r['mean_cloud']:8.4f} "
            f"{r['mean_abs_diff']:8.4f} {r['pearson']:8.3f} {r['spearman']:9.3f} "
            f"{r['wilcoxon_p']:7.3f}{star}"
        )
    if any(r["heavy_tailed"] for r in rows):
        s = next(r for r in rows if r["heavy_tailed"])
        print(
            f"\n* {s['metric']} is an unbounded ratio: medians {s['median_local']:.3f} vs "
            f"{s['median_cloud']:.3f}, median |diff| {s['median_abs_diff']:.3f}. "
            f"Read Spearman ({s['spearman']:.3f}), not Pearson ({s['pearson']:.3f})."
        )

    worst = min(r["spearman"] for r in rows if not r["heavy_tailed"])
    print(
        f"\n[backend] weakest rank agreement among the bounded metrics: "
        f"Spearman {worst:.3f}; no metric shows a significant paired shift at 0.05"
        if all(r["wilcoxon_p"] > 0.05 for r in rows)
        else f"\n[backend] weakest rank agreement: Spearman {worst:.3f}"
    )
    print(f"[backend] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
