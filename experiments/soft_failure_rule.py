"""Is the confound an artefact of the hard threshold, or of the reference?

Referee 2, point 2: the paper sweeps the threshold *values* (section 6.4), the
utility *weights* (section 6.4), the grid *quantisation* (continuous_aure.py) and
the summary *class* (section 6.7), but never the *functional form*. Every variant
still converts a condition into a pass or a fail. If the confound were a property
of hard-thresholding, softening the rule should dissolve it.

It does not, and the reason is structural. Replace the indicator with a
pass-probability

    p(lambda) = prod_c sigmoid( margin_c(lambda) / s_c )

over the three criteria, and the radius with its expectation under the same
contiguous-run construction,

    rho_soft = sum_i [ prod_{j<=i} p(lambda_j) ] * (lambda_i - lambda_{i-1}),

which returns the published rho exactly as s -> 0. Softness is swept as a
fraction alpha of each criterion's own threshold, so the three criteria soften
commensurably.

Softening spreads a cell's contribution but does not re-anchor it: a model whose
clean state sits below the reference still enters the product with a small
p(lambda=0), and that factor multiplies every later term. The admissibility
factor survives in attenuated form, which is the answer to the referee -- the
confound belongs to the reference anchor, not to the hardness of the bar.

Usage::

    python experiments/soft_failure_rule.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from tice.config import Thresholds  # noqa: E402

_TH = Thresholds()
_DROP = ("tabpfn_client",)
ICL = ("tabicl", "tabpfn")
GBDT = ("catboost", "xgboost", "hist_gbdt")
CELL = ["model", "dataset_id", "shift_axis", "base_seed"]

# alpha = 0 is the published hard rule and is asserted against it; the rest
# soften each criterion by that fraction of its own threshold.
ALPHAS = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

_DEFAULT = (
    _ROOT
    / "results"
    / "external"
    / "tables_2axis_multiseed_primary_with_tabpfn_local"
    / "shift_results.csv"
)


def _margins(df: pd.DataFrame) -> pd.DataFrame:
    """Signed slack on each criterion: positive passes, negative fails."""
    ref = df["reference_utility"].astype(float)
    out = pd.DataFrame(index=df.index)
    # utility: passes while U >= ref - tau_u. No reference -> criterion inert,
    # which is the section 5.3 hazard; it is preserved here rather than patched
    # so the soft rule scores exactly the cells the hard rule scored.
    out["u"] = np.where(ref.notna(), df["utility"].astype(float) - (ref - _TH.tau_utility), np.inf)
    out["ece"] = _TH.tau_ece - df["ece"].astype(float)
    out["nll"] = _TH.tau_nll - df["nll_norm"].astype(float)
    # an undefined metric is a failure in the published rule, at any softness
    bad = (df["status"] != "ok") | df["utility"].isna()
    for c in ("u", "ece", "nll"):
        out.loc[out[c].isna(), c] = -np.inf
        out.loc[bad, c] = -np.inf
    return out


def _pass_prob(m: pd.DataFrame, alpha: float) -> pd.Series:
    if alpha <= 0:
        return ((m["u"] >= 0) & (m["ece"] >= 0) & (m["nll"] >= 0)).astype(float)
    scales = {"u": alpha * _TH.tau_utility, "ece": alpha * _TH.tau_ece, "nll": alpha * _TH.tau_nll}
    p = pd.Series(1.0, index=m.index)
    for c, s in scales.items():
        z = np.clip(m[c].to_numpy(dtype=float) / s, -500, 500)
        p *= 1.0 / (1.0 + np.exp(-z))
    return p


def _soft_rho(g: pd.DataFrame) -> tuple[float, float]:
    """(rho_soft, p at lambda=0) for one cell, rows already sorted by lambda."""
    lam = g["shift_lambda"].to_numpy(dtype=float)
    p = g["_p"].to_numpy(dtype=float)
    surv = np.cumprod(p)
    rho = float(np.sum(surv[1:] * np.diff(lam)))
    return rho, float(p[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=_DEFAULT)
    ap.add_argument("--out", type=Path, default=_ROOT / "results" / "ablations" / "soft_rule")
    args = ap.parse_args(argv)

    if not args.results.exists():
        print(f"[soft-rule] missing {args.results}")
        return 1
    df = pd.read_csv(args.results)
    df = df[~df.model.isin(_DROP)]
    df = df[df["applicable"]].sort_values(CELL + ["shift_lambda"])
    m = _margins(df)
    print(f"[soft-rule] {len(df)} applicable rows")

    rows = []
    for alpha in ALPHAS:
        d = df.copy()
        d["_p"] = _pass_prob(m, alpha).to_numpy()
        recs = []
        for keys, g in d.groupby(CELL, sort=False):
            rho, p0 = _soft_rho(g)
            recs.append({"model": keys[0], "rho": rho, "p0": p0})
        e = pd.DataFrame(recs)
        s = e.groupby("model").agg(aure=("rho", "mean"), A=("p0", "mean"))
        s["T"] = s["aure"] / s["A"]
        marg = float(s.loc[list(ICL), "aure"].min() - s.loc[list(GBDT), "aure"].max())
        a_gap = float(s.loc[list(ICL), "A"].min() - s.loc[list(GBDT), "A"].max())
        for model, r in s.iterrows():
            rows.append(
                {"alpha": alpha, "model": model, "aure": r.aure, "A": r.A, "T": r["T"]}
            )
        print(
            f"  alpha={alpha:4.2f}  margin={marg:+.4f}  A-gap={a_gap:+.4f}   "
            + "  ".join(f"{k}={s.loc[k, 'aure']:.4f}" for k in ("tabicl", "tabpfn", "catboost"))
        )

        if alpha == 0.0:
            # the hard limit must reproduce the published AURE
            published = {"tabicl": 0.0977, "tabpfn": 0.0888, "catboost": 0.0744}
            for k, v in published.items():
                if abs(s.loc[k, "aure"] - v) > 5e-4:
                    raise SystemExit(
                        f"[soft-rule] hard limit gives {k}={s.loc[k, 'aure']:.4f}, "
                        f"published {v} -- the soft rule does not reduce to the hard one"
                    )
            print("         (hard limit reproduces the published AURE)")

    out = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "soft_rule.csv", index=False)

    piv = out.pivot(index="alpha", columns="model", values="A")
    print("\nadmissibility A (the confounded factor) as the rule softens:")
    print(piv[["tabicl", "tabpfn", "catboost", "xgboost", "hist_gbdt", "logreg"]].to_string(
        float_format=lambda x: f"{x:.3f}"
    ))
    print(f"\n[soft-rule] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
