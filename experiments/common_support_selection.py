"""Does common support select a structurally different slice of the benchmark?

Referee 1, comment 3: common support retains 68 of 264 cells across 15 datasets,
and section 6.2 treats that as a loss of power. It is also a *selection*: cells
survive only where every model is admissible at lambda=0, which is not a random
thinning. If the retained datasets are systematically smaller, cleaner or easier,
the reversal measured on them partly reflects the slice rather than the rule.

This compares retained against excluded datasets on the meta-features recorded
before any analysis was run, plus a clean-difficulty proxy computed from the
lambda=0 rows themselves.

It is reported descriptively, with medians and ranges. With 15 against 29
datasets the comparison is itself underpowered, so a non-significant difference
here is weak evidence of similarity -- the same caution section 6.2 applies to
the common-support tests applies to this table, and stating it is the point.

Usage::

    python experiments/common_support_selection.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CELL = ["dataset_id", "shift_axis", "base_seed"]
_DROP = ("tabpfn_client",)

# the meta-features the referee names, plus the two the selection filter used
FEATURES = [
    ("n_samples", "sample size"),
    ("n_features", "features"),
    ("n_numeric", "numeric features"),
    ("n_categorical", "categorical features"),
    ("categorical_fraction", "categorical fraction"),
    ("n_classes", "classes"),
    ("class_imbalance_ratio", "class imbalance ratio"),
    ("missing_rate", "missing rate"),
    ("clean_auc", "clean AUC (mean over models)"),
    ("clean_utility", "clean utility U (mean over models)"),
]

_DEFAULT_DIR = (
    _ROOT / "results" / "external" / "tables_2axis_multiseed_primary_with_tabpfn_local"
)
_PROFILES = _ROOT / "results" / "external" / "dataset_profiles_external.csv"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=_DEFAULT_DIR / "shift_results.csv")
    p.add_argument("--profiles", type=Path, default=_PROFILES)
    p.add_argument(
        "--out", type=Path, default=_ROOT / "results" / "ablations" / "common_support_selection"
    )
    args = p.parse_args(argv)

    for f in (args.results, args.profiles):
        if not f.exists():
            print(f"[cs-select] missing {f}")
            return 1

    df = pd.read_csv(args.results)
    df = df[~df.model.isin(_DROP)]
    df = df[df["applicable"]]

    zero = df[df.shift_lambda == 0.0]
    passed = zero.groupby(CELL)["failed"].apply(lambda s: not s.any())
    keep = passed[passed].index
    n_cells = zero.groupby(CELL).ngroups
    retained_ds = sorted({k[0] for k in keep})
    all_ds = sorted(df.dataset_id.unique())
    excluded_ds = [d for d in all_ds if d not in retained_ds]
    print(
        f"[cs-select] common support: {len(keep)} of {n_cells} cells, "
        f"{len(retained_ds)} of {len(all_ds)} datasets"
    )

    # clean difficulty from the lambda=0 rows: averaged over models so it
    # describes the dataset, not any one model's standing on it
    clean = (
        zero.groupby("dataset_id")[["auc", "utility"]]
        .mean()
        .rename(columns={"auc": "clean_auc", "utility": "clean_utility"})
    )

    prof = pd.read_csv(args.profiles).drop_duplicates("dataset_id").set_index("dataset_id")
    prof = prof.join(clean, how="right")
    prof["categorical_fraction"] = prof["n_categorical"] / prof["n_features"]
    prof["retained"] = prof.index.isin(retained_ds)

    rows = []
    for col, label in FEATURES:
        if col not in prof:
            continue
        a = prof.loc[prof.retained, col].astype(float).dropna()
        b = prof.loc[~prof.retained, col].astype(float).dropna()
        if a.empty or b.empty:
            continue
        try:
            u_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            u_p = float("nan")
        rows.append(
            {
                "feature": label,
                "retained_n": len(a),
                "retained_median": a.median(),
                "retained_min": a.min(),
                "retained_max": a.max(),
                "excluded_n": len(b),
                "excluded_median": b.median(),
                "excluded_min": b.min(),
                "excluded_max": b.max(),
                "mannwhitney_p": u_p,
            }
        )

    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "common_support_selection.csv", index=False)
    prof[["retained"] + [c for c, _ in FEATURES if c in prof]].to_csv(
        args.out / "dataset_level.csv"
    )

    print()
    hdr = f"{'feature':30s} {'retained (15)':>22s} {'excluded (29)':>22s} {'p':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ret = f"{r['retained_median']:.3g} [{r['retained_min']:.3g}, {r['retained_max']:.3g}]"
        exc = f"{r['excluded_median']:.3g} [{r['excluded_min']:.3g}, {r['excluded_max']:.3g}]"
        print(f"{r['feature']:30s} {ret:>22s} {exc:>22s} {r['mannwhitney_p']:7.3f}")

    sig = [r["feature"] for r in rows if r["mannwhitney_p"] < 0.05]
    print()
    if sig:
        print(f"[cs-select] differs at 0.05 (uncorrected, {len(rows)} tests): {', '.join(sig)}")
    else:
        print("[cs-select] no feature differs at 0.05 -- but at n=15 vs 29 that is weak")
    print(f"[cs-select] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
