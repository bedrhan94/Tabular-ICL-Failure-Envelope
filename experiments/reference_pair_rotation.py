"""Rotate the gradient-boosted reference pair and re-score the envelope.

Referee 2, point 1: the paper sweeps the three *thresholds* over a 336-point grid
but never varies the *reference pair* those thresholds are applied against. The
pair is a configured constant -- ``("xgboost", "catboost")`` in
:mod:`tice.config` -- and section 6.1 shows the relative criterion carries 95% of
the zero-shift failures, so the pair is the load-bearing choice the sweep never
reaches.

This re-scores stored conditions under alternative reference sets. No models are
re-run: the per-condition metrics are fixed, only which model supplies
``reference_utility`` changes. The published set is re-scored first and asserted
against the stored ``failed`` flags, so a drift in the rule would fail loudly
rather than produce a plausible table.

The stored ``best_gbdt`` and ``reference_utility`` columns belong to the
published pair and are deliberately *not* reused -- they are recomputed from the
``utility`` column for every candidate set, exactly as
``tice.pipeline._select_best_gbdt`` does it: argmax clean (lambda=0) utility,
per dataset and seed.

Usage::

    python experiments/reference_pair_rotation.py
    python experiments/reference_pair_rotation.py --results <shift_results.csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tice.config import Thresholds  # noqa: E402
from tice.envelope.reliability import envelope_radius  # noqa: E402
from tice.metrics.utility import evaluate_failure  # noqa: E402

_TH = Thresholds()

# The client arm is quota-limited (844 of 1584 conditions) and superseded by the
# local weights; it is excluded so min-ICL is not supplied by the thinner arm.
_DROP = ("tabpfn_client",)

ICL = ("tabicl", "tabpfn")
GBDT = ("catboost", "xgboost", "hist_gbdt")

# The published pair leads; the rest vary which boosted models may be the
# reference. Single-model sets are the extreme cases and bracket the pairs.
CANDIDATES: list[tuple[str, tuple[str, ...]]] = [
    ("published (xgboost, catboost)", ("xgboost", "catboost")),
    ("(xgboost, hist_gbdt)", ("xgboost", "hist_gbdt")),
    ("(catboost, hist_gbdt)", ("catboost", "hist_gbdt")),
    ("all three boosted", ("xgboost", "catboost", "hist_gbdt")),
    ("catboost only", ("catboost",)),
    ("xgboost only", ("xgboost",)),
    ("hist_gbdt only", ("hist_gbdt",)),
]

_DEFAULT = (
    _ROOT
    / "results"
    / "external"
    / "tables_2axis_multiseed_primary_with_tabpfn_local"
    / "shift_results.csv"
)


def _clean_reference(df: pd.DataFrame, reference_models: tuple[str, ...]) -> pd.Series:
    """Map (dataset, seed) -> (best model, its clean utility) for one candidate set.

    Mirrors ``_select_best_gbdt``: argmax clean-lambda=0 utility among the
    configured models. Keyed by seed as well as dataset because each seed was
    scored as its own run before the arms were merged.
    """
    clean = df[(df["shift_lambda"] == 0.0) & (df["status"] == "ok")]
    clean = clean[clean["model"].isin(reference_models)]
    clean = clean.dropna(subset=["utility"])
    if clean.empty:
        raise SystemExit("[ref-rotation] no clean reference rows for this candidate set")
    # one clean utility per (dataset, seed, model): the axes agree at lambda=0,
    # which the assertion below enforces rather than assumes
    per = clean.groupby(["dataset_id", "base_seed", "model"])["utility"].agg(["min", "max"])
    if not ((per["max"] - per["min"]).abs() < 1e-9).all():
        raise SystemExit("[ref-rotation] clean utility disagrees across axes at lambda=0")
    u = per["max"].rename("utility").reset_index()
    idx = u.groupby(["dataset_id", "base_seed"])["utility"].idxmax()
    best = u.loc[idx].set_index(["dataset_id", "base_seed"])
    return best


def rescore(df: pd.DataFrame, reference_models: tuple[str, ...]) -> pd.DataFrame:
    """Return the per-row frame with `failed` recomputed under a reference set."""
    best = _clean_reference(df, reference_models)
    keys = list(zip(df["dataset_id"], df["base_seed"]))
    ref_util = pd.Series(
        [best["utility"].get(k, float("nan")) for k in keys], index=df.index
    )
    ref_model = pd.Series(
        [best["model"].get(k, "") for k in keys], index=df.index
    )
    failed = []
    reasons = []
    for util, ece, nll, ref, ok in zip(
        df["utility"], df["ece"], df["nll_norm"], ref_util, df["status"] == "ok"
    ):
        res = evaluate_failure(
            utility=util,
            ece=ece,
            nll_norm=nll,
            reference_utility=ref,
            thresholds=_TH,
            status_ok=bool(ok),
        )
        failed.append(res.failed)
        reasons.append(res.reason_str)
    out = df.copy()
    out["ref_model"] = ref_model
    out["reference_utility_new"] = ref_util
    out["failed_new"] = failed
    out["failure_reason_new"] = reasons
    return out


def envelopes(scored: pd.DataFrame) -> pd.DataFrame:
    """rho per (model, dataset, axis, seed) from the recomputed failure flags."""
    rows = []
    group = scored.groupby(["model", "dataset_id", "shift_axis", "base_seed"])
    for (model, ds, axis, seed), g in group:
        g = g.sort_values("shift_lambda")
        rows.append(
            {
                "model": model,
                "dataset_id": ds,
                "shift_axis": axis,
                "base_seed": seed,
                "rho": envelope_radius(g["shift_lambda"], g["failed_new"]),
                "passed_lambda0": not bool(
                    g[g["shift_lambda"] == 0.0]["failed_new"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarise(env: pd.DataFrame) -> pd.DataFrame:
    """AURE, admissibility A, conditional tolerance T, and the lambda=0 fail rate."""
    rows = []
    for model, g in env.groupby("model"):
        admitted = g[g["passed_lambda0"]]
        aure = g["rho"].mean()
        a = g["passed_lambda0"].mean()
        t = admitted["rho"].mean() if len(admitted) else float("nan")
        rows.append(
            {
                "model": model,
                "aure": aure,
                "A": a,
                "T": t,
                "lambda0_fail_rate": 1.0 - a,
                "n_cells": len(g),
            }
        )
    out = pd.DataFrame(rows).sort_values("aure", ascending=False)
    # the identity the decomposition rests on; assert rather than trust
    ok = ((out["A"] * out["T"].fillna(0.0) - out["aure"]).abs() < 1e-9).all()
    if not ok:
        raise SystemExit("[ref-rotation] A*T != AURE -- decomposition broken")
    return out


def margin(summary: pd.DataFrame) -> float:
    s = summary.set_index("model")["aure"]
    icl = [m for m in ICL if m in s.index]
    gbdt = [m for m in GBDT if m in s.index]
    return float(s.loc[icl].min() - s.loc[gbdt].max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=_DEFAULT)
    p.add_argument("--out", type=Path, default=_ROOT / "results" / "ablations" / "reference_rotation")
    args = p.parse_args(argv)

    if not args.results.exists():
        print(f"[ref-rotation] missing {args.results}")
        return 1
    df = pd.read_csv(args.results)
    df = df[~df["model"].isin(_DROP)]
    df = df[df["applicable"]]
    print(f"[ref-rotation] {len(df)} applicable rows, {df['model'].nunique()} models")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    who = []
    for label, refs in CANDIDATES:
        scored = rescore(df, refs)

        if refs == ("xgboost", "catboost"):
            # fidelity gate: the published set must reproduce the stored flags
            same = (scored["failed_new"] == scored["failed"]).mean()
            if same < 1.0:
                bad = scored[scored["failed_new"] != scored["failed"]]
                raise SystemExit(
                    f"[ref-rotation] published set reproduces only {same:.4%} of "
                    f"stored flags ({len(bad)} rows differ) -- re-scoring is not faithful"
                )
            print("[ref-rotation] published set reproduces stored failure flags exactly")

        env = envelopes(scored)
        s = summarise(env)
        s.insert(0, "reference_set", label)
        rows.append(s)

        # which model actually supplies the reference, per cell
        cells = scored[scored["shift_lambda"] == 0.0]
        share = cells.groupby("ref_model").size() / len(cells)
        who.append(
            {"reference_set": label, **{f"ref_is_{k}": round(v, 4) for k, v in share.items()}}
        )
        print(f"  {label:32s} margin = {margin(s):+.4f}   ref share: "
              + ", ".join(f"{k} {v:.0%}" for k, v in share.items()))

    table = pd.concat(rows, ignore_index=True)
    table.to_csv(args.out / "reference_rotation_full.csv", index=False)

    marg = (
        table.groupby("reference_set", sort=False)
        .apply(lambda g: margin(g), include_groups=False)
        .rename("min_icl_minus_best_gbdt")
        .reset_index()
    )
    marg.to_csv(args.out / "reference_rotation_margins.csv", index=False)
    pd.DataFrame(who).to_csv(args.out / "reference_share.csv", index=False)

    print()
    print(marg.to_string(index=False))
    print(f"\n[ref-rotation] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
