"""Figure for the reference-confound ablation (Sprint 7).

Three panels, telling the argument in order:

  A. cause       -- lambda=0 failure rate per model: how often a cell is decided
                    before any shift is applied (rho is forced to 0 there).
  B. effect      -- AURE per model under the published rule vs the two
                    confound-free variants (common support, self-referenced).
  C. consequence -- the min-ICL minus best-GBDT margin under each variant, across
                    all three evidence bases; the sign flips.

Reads `results/ablations/*/ablation_summary.csv` (written by
`ablate_reference_confound.py`) and writes
`results/ablations/figures/reference_confound.png`.

Usage::

    python experiments/make_ablation_figures.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Springer asks for 600 dpi on combination art; 170 is fine for reading on screen.
# Set TICE_FIG_DPI=600 before running to regenerate at submission resolution.
FIG_DPI = int(os.environ.get("TICE_FIG_DPI", "170"))

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tice.figio import save_figure  # noqa: E402

# Categorical slots 1-3, validated (light + dark) with the data-viz palette
# validator: lightness band, chroma floor, CVD separation and normal-vision
# floor all pass. The aqua slot warns on light-surface contrast (2.74:1), which
# obliges the direct value labels used on every bar below.
VARIANT = {
    "aure_grid": ("published rule", "#2a78d6"),
    "aure_common": ("common support", "#eb6834"),
    "aure_self": ("self-referenced", "#1baf7a"),
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASES = [
    # The three-seed primary benchmark leads: it is the basis the paper's headline and
    # its refutation both sit on. The pilot row is the seed-42 grid, whose published-rule
    # margin is +0.077; the +0.043 quoted in the abstract is the 10-seed average of the
    # same rule. Spelling out the basis keeps the two numbers from reading as a
    # contradiction.
    ("external_primary_3seed", "External primary\n44 ds x 3 seeds"),
    ("external_44", "External\n44 ds x 2 axes\n(seed 42)"),
    ("external_multiseed", "External subset\n12 ds x 3 seeds"),
    ("pilot", "Pilot\n5 ds x 6 axes\n(seed-42 grid)"),
]
# tabpfn_client is quota-limited and superseded by the local-weights `tabpfn` rows; on the
# primary basis it covers 141 of 264 cells and would drag min-ICL down for the wrong reason.
ICL = ("tabicl", "tabpfn", "tabpfn_client")
GBDT = ("catboost", "xgboost", "hist_gbdt", "catboost_tuned", "xgboost_tuned")


def _tidy(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_axisbelow(True)


def panel_lambda0(ax: plt.Axes, s: pd.DataFrame) -> None:
    # One series -> one neutral colour and no legend; the variant hues are
    # reserved for panels B/C so they never mean two things in one figure.
    d = s.sort_values("lambda0_fail_rate")
    ax.barh(d.model, d.lambda0_fail_rate, color="#898781", height=0.62)
    for m, v in zip(d.model, d.lambda0_fail_rate, strict=True):
        ax.text(v + 0.012, m, f"{v:.0%}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, max(0.55, d.lambda0_fail_rate.max() * 1.25))
    ax.set_xlabel("share of cells failing at $\\lambda$=0", fontsize=9, color=INK2)
    ax.set_title("A. Cells decided before any shift", fontsize=11, color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _tidy(ax)


def panel_variants(ax: plt.Axes, s: pd.DataFrame) -> None:
    d = s.sort_values("aure_grid", ascending=True)
    ys = range(len(d))
    h = 0.26
    for i, (col, (label, color)) in enumerate(VARIANT.items()):
        off = (i - 1) * h
        vals = d[col].fillna(0.0)
        ax.barh([y + off for y in ys], vals, height=h - 0.03, color=color, label=label)
        for y, v in zip(ys, vals, strict=True):
            if v > 0:
                ax.text(v + 0.002, y + off, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(d.model)
    ax.set_xlabel("AURE", fontsize=9, color=INK2)
    ax.set_title("B. Same runs, three failure rules", fontsize=11, color=INK, loc="left")
    ax.set_xlim(0, max(0.19, float(d[list(VARIANT)].max().max()) * 1.22))
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _tidy(ax)


def panel_margin(ax: plt.Axes, margins: pd.DataFrame) -> None:
    ax.axvline(0, color="#c3c2b7", lw=1.2, zorder=1)
    for yi, (_, label) in enumerate(BASES):
        row = margins[margins.basis == label]
        for col, (vlabel, color) in VARIANT.items():
            if col not in row or row[col].isna().all():
                continue
            ax.scatter(row[col], [yi], s=95, color=color, zorder=3,
                       edgecolor="#fcfcfb", linewidth=1.6,
                       label=vlabel if yi == 0 else None)
    ax.set_yticks(range(len(BASES)))
    ax.set_yticklabels([lbl for _, lbl in BASES], fontsize=8.5)
    ax.set_xlabel("min-ICL $-$ best-GBDT AURE", fontsize=9, color=INK2)
    ax.set_title("C. The margin flips sign", fontsize=11, color=INK, loc="left")
    # room under the lowest row for the polarity annotations, so they sit on the
    # zero line they refer to rather than floating at the top of the panel
    ax.set_ylim(-0.85, len(BASES) - 0.45)
    lo, hi = ax.get_xlim()
    pad = (hi - lo) * 0.03
    ax.text(pad, -0.7, "ICL ahead →", fontsize=8, color=MUTED, va="center")
    ax.text(-pad, -0.7, "← GBDT ahead", fontsize=8, color=MUTED, ha="right", va="center")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    _tidy(ax)


def collect_margins(root: Path) -> pd.DataFrame:
    rows = []
    for key, label in BASES:
        f = root / key / "ablation_summary.csv"
        if not f.exists():
            continue
        s = f.read_text(encoding="utf-8")
        d = pd.read_csv(f) if s.strip() else None
        if d is None:
            continue
        idx = d.set_index("model")
        icl = [m for m in ICL if m in idx.index]
        gbdt = [m for m in GBDT if m in idx.index]
        # Each basis must carry exactly one TabPFN arm. The local weights and the
        # quota-limited client cover different cell counts, so a basis holding both
        # would silently let the thinner arm supply min-ICL -- the single-family
        # scoring hazard of section 8, in the figure layer this time.
        if {"tabpfn", "tabpfn_client"} <= set(icl):
            raise SystemExit(
                f"[ablation-fig] {key} carries both tabpfn and tabpfn_client; "
                "min-ICL would be ambiguous. Drop one before plotting."
            )
        rec = {"basis": label}
        for col in VARIANT:
            if col in idx and icl and gbdt:
                v = idx[col]
                if not v.loc[icl].isna().all() and not v.loc[gbdt].isna().all():
                    rec[col] = v.loc[icl].min() - v.loc[gbdt].max()
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablations", type=Path, default=_ROOT / "results" / "ablations")
    # Panels A and B default to the three-seed primary benchmark: it is the basis the
    # paper's headline and its refutation both sit on, and the caption quotes its
    # numbers. The 12-dataset subset was the old default and produced a figure whose
    # panels disagreed with its own caption.
    p.add_argument("--basis", default="external_primary_3seed", help="basis for panels A/B")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    summary_f = args.ablations / args.basis / "ablation_summary.csv"
    if not summary_f.exists():
        print(f"[ablation-fig] missing {summary_f} -- run ablate_reference_confound.py first")
        return 1
    s = pd.read_csv(summary_f)
    margins = collect_margins(args.ablations)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
    panel_lambda0(axes[0], s)
    panel_variants(axes[1], s)
    panel_margin(axes[2], margins)
    label = dict(BASES)[args.basis] if args.basis in dict(BASES) else args.basis
    # The framing is the reference policy, not calibration: section 6.1 counts which
    # trigger fires and finds the relative criterion carries 95% of the zero-shift
    # failures against 5% for the absolute calibration bar.
    fig.suptitle(
        "The reference policy, not the calibration bar, decides the envelope"
        f"   (panels A-B: {label.replace(chr(10), ', ')})",
        fontsize=12.5, color=INK, x=0.006, ha="left", y=0.975,
    )
    # one figure-level legend for the shared variant encoding: repeating it per
    # panel collided with the value labels and the hues mean the same thing in B and C
    handles = [
        plt.Line2D([], [], marker="s", ls="", markersize=8, color=color, label=lab)
        for lab, color in VARIANT.values()
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK2,
               loc="upper right", bbox_to_anchor=(0.995, 1.005), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out = args.out or (args.ablations / "figures" / "reference_confound.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out, dpi=FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[ablation-fig] wrote {out}")
    print(margins.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
