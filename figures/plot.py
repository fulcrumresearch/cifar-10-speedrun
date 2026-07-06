#!/usr/bin/env python3
"""README figures from the `runners/` compute-honest summaries.

Consumes the flat-mirror summaries the harness writes to
`runners/results/runner_<target>.json` (schema: time.mean_s = compute-honest,
diagnostics.train_loop, acc_mean, ...). Writes the images into this folder.

  uv run python figures/plot.py

Figures:
  attribution.{png,svg}           hiverge -> curriculum -> e1 waterfall (README Fig. 1)
  attribution_cheating.{png,svg}  real speedup vs excluded exploits (README Fig. 2)
  frontier.{png,svg}              compute-honest hiverge vs e1 endpoints, decomposed
                                  into timed train-loop + charged prepare
  iso_curriculum.{png,svg}        curriculum vs matched full-32, per graph setting
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "runners" / "results"
PLOTS = REPO / "figures"
PLOTS.mkdir(exist_ok=True)
GATE = 0.94

TEAL, TEAL_D, TEAL_L, AMBER = "#1D9E75", "#0F6E56", "#5DCAA5", "#BA7517"
INK, MUTED, FAINT = "#222222", "#555555", "#8a8a8a"

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except Exception:
    pass
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
})


def _load(label):
    p = RESULTS / f"runner_{label}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ch(d):
    """(compute_honest_mean, std, train_loop_mean, prepare_mean, acc)."""
    return (d["time"]["mean_s"], d["time"]["std_s"],
            d["diagnostics"]["train_loop"]["mean_s"],
            d["diagnostics"]["prepare"]["mean_s"], d["acc_mean"])


def frontier():
    """Compute-honest hiverge vs e1, each split into timed train-loop (base) +
    charged prepare (top), so the bar total is the official compute-honest metric."""
    pts = [(lbl, name, c) for lbl, name, c in
           (("hiverge", "hiverge\n(7.65-ep baseline)", AMBER),
            ("e1", "e1\n(champion)", TEAL_D))]
    rows = [(name, c, _ch(d)) for lbl, name, c in pts if (d := _load(lbl))]
    if len(rows) < 2:
        print("frontier: need both hiverge + e1 summaries; skipping")
        return

    floor = 1.70
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = range(len(rows))
    for i, (name, c, (ch, std, tl, prep, acc)) in enumerate(rows):
        # timed train-loop segment (floor -> train_loop), then charged prepare on top
        ax.bar(i, tl - floor, bottom=floor, width=0.56, color=c, zorder=3)
        ax.bar(i, ch - tl, bottom=tl, width=0.56, color=c, alpha=0.42, zorder=3,
               hatch="////", edgecolor="white", linewidth=0)
        ax.errorbar(i, ch, yerr=std, fmt="none", ecolor=FAINT, lw=1.2, capsize=4, zorder=4)
        ax.text(i, ch + std + 0.006, f"{ch:.3f}s", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(i, floor + 0.012, f"acc {acc*100:.2f}%", ha="center", va="bottom",
                fontsize=8.5, color="white", zorder=5)
        ax.text(i, (tl + ch) / 2, f"prepare\n{prep*1000:.0f}ms", ha="center", va="center",
                fontsize=7.5, color=MUTED, zorder=5)

    ch_hv, ch_e1 = rows[0][2][0], rows[1][2][0]
    pct = 100 * (ch_hv - ch_e1) / ch_hv
    ax.annotate("", xy=(1, ch_e1), xytext=(1, ch_hv),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.3))
    ax.text(1.32, (ch_hv + ch_e1) / 2, f"{pct:.1f}% less time\n(−{ch_hv - ch_e1:.2f}s)", ha="left", va="center",
            fontsize=10, color=TEAL_D, fontweight="bold")

    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax.set_xlim(-0.6, 1.9)
    ax.set_ylim(floor, 2.04)
    ax.set_ylabel("time to 94% accuracy (s) — single A100, n=200")
    ax.set_title("CIFAR-10 frontier — hiverge → e1 (time to 94% accuracy)", loc="left")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    solid = plt.Rectangle((0, 0), 1, 1, color=MUTED)
    hatched = plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.42, hatch="////")
    ax.legend([solid, hatched], ["training loop", "per-run setup (reset+whiten+aug)"],
              loc="upper right", framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"frontier.{ext}")
    plt.close(fig)
    print(f"wrote frontier.png/.svg  (hiverge {ch_hv:.3f}s -> e1 {ch_e1:.3f}s, -{pct:.1f}%)")


# Combined 6-bar waterfall. LEFT half (legit, measured n=200): the REAL hiverge -> e1
# speedup -- hiverge -> + 28->32 resolution curriculum (plain torch.compile) -> + the
# deeper 24->28->32 curriculum = e1. Both curriculum moves are single-variable within
# a recipe (e5: 1.993->1.865; e1: 1.903->1.823); e1's systems stack itself is
# net-neutral-to-slower at a fixed schedule (see README Results ledger).
# RIGHT half (cheating, RED): how single-GPU metric/environment exploits would FAKE
# further below e1 -- bad timed region (e1's train-loop diagnostic, relocate off-loop),
# machine lottery (e1's train-loop MIN = luckiest host), thermal cooldown (~9 ms untimed
# sleep, documented). None is real efficiency, which is why compute-honest refuses them.
WATERFALL_LEGIT = [
    ("hiverge",           "hiverge\nbaseline",                              AMBER),
    ("e5_mild_s825_n200", "+ resolution curriculum\n(plain torch.compile)", TEAL),
    ("e1",                "+ deeper 24px curriculum\n= e1",                 TEAL_D),
]
WATERFALL_CHEATS = [
    ("− bad timed region\n(relocate off-loop)", 1.766, "#B23A48"),
    ("− machine lottery\n(luckiest host)",      1.726, "#B23A48"),
    ("− thermal cooldown\n(untimed sleep)",     1.717, "#B23A48"),
]


def attribution():
    """Clean 3-bar waterfall of the REAL hiverge -> e1 speedup: hiverge -> + resolution
    curriculum (plain torch.compile) -> + systems stack = e1. Curriculum is the lever,
    systems the small finish. Bars load from result JSONs (n=200)."""
    rows = [(name, c, _load(lbl)) for lbl, name, c in WATERFALL_LEGIT]
    if any(d is None for _, _, d in rows):
        print("attribution: missing legit cells; skipping")
        return
    vals = [d["time"]["mean_s"] for _, _, d in rows]
    names = [n for n, _, _ in rows]
    cols = [c for _, c, _ in rows]

    floor, top = min(vals) - 0.045, max(vals) + 0.032
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for i in range(len(vals)):
        ax.bar(i, vals[i] - floor, bottom=floor, width=0.54, color=cols[i], zorder=3)
        ax.text(i, vals[i] + 0.004, f"{vals[i]:.3f}s", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=INK)
        if i > 0:
            prev = vals[i - 1]
            ax.plot([i - 1 + 0.27, i - 0.27], [prev, prev], color=FAINT, lw=1, ls=(0, (4, 3)), zorder=4)
            ax.annotate("", xy=(i - 0.46, vals[i]), xytext=(i - 0.46, prev),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
            ax.text(i - 0.42, (prev + vals[i]) / 2, f"−{prev - vals[i]:.3f}s",
                    ha="left", va="center", fontsize=10.5, fontweight="bold", color=TEAL_D)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(vals) - 0.35)
    ax.set_ylim(floor, top)
    ax.set_ylabel("time to 94% (s) — single A100, n=200")
    ax.set_title("Where the hiverge → e1 speedup comes from", loc="left", fontsize=12)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.text(0.5, 0.008, "approximate — bars are different recipes (hiverge / a plain torch.compile "
             "curriculum recipe / e1), each at its certified compute-honest time (n=200)",
             ha="center", fontsize=7, color=FAINT)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"attribution.{ext}")
    plt.close(fig)
    print(f"wrote attribution.png/.svg  (hiverge {vals[0]:.3f} -> curriculum {vals[1]:.3f} -> e1 {vals[2]:.3f})")


def attribution_cheating():
    """Floating waterfall: each bar is the time that step SAVES (the delta), hanging in
    the descending staircase from the hiverge baseline. GREEN region = real speedup
    (curriculum + systems = hiverge->e1, measured n=200). RED region = single-GPU gaming
    below e1's honest frontier (e1 train-loop diagnostic / luckiest-host min / ~9 ms
    thermal cooldown). No title; regions labelled."""
    legit = [(name, c, _load(lbl)) for lbl, name, c in WATERFALL_LEGIT]
    if any(d is None for _, _, d in legit):
        print("attribution_cheating: missing legit cells; skipping")
        return
    names = [n for n, _, _ in legit] + [n for n, _, _ in WATERFALL_CHEATS]
    vals = [d["time"]["mean_s"] for _, _, d in legit] + [v for _, v, _ in WATERFALL_CHEATS]
    nl = len(legit)  # bars [0, nl) real speedup; [nl, end) gaming
    GREEN, RED = TEAL, "#B23A48"

    floor, top = min(vals) - 0.012, max(vals) + 0.03
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.axvspan(-0.5, nl - 0.5, color=GREEN, alpha=0.07, zorder=0)
    ax.axvspan(nl - 0.5, len(vals) - 0.5, color=RED, alpha=0.07, zorder=0)
    ax.axvline(nl - 0.5, color=FAINT, lw=1.1, ls=(0, (3, 3)), zorder=1)
    ax.text((nl - 1) / 2.0, top - 0.003, "real speedup", ha="center", va="top",
            fontsize=13, color=TEAL_D, fontweight="bold")
    ax.text((nl + len(vals) - 1) / 2.0, top - 0.003, "gaming", ha="center", va="top",
            fontsize=13, color=RED, fontweight="bold")

    # hiverge: the baseline (full bar, neutral — it is not a saving)
    ax.bar(0, vals[0] - floor, bottom=floor, width=0.62, color=MUTED, zorder=3)
    ax.text(0, vals[0] + 0.004, f"{vals[0]:.3f}s", ha="center", va="bottom",
            fontsize=10.5, fontweight="bold", color=INK)
    # every later bar = the time that step SAVES, hanging from the running total down to it
    for i in range(1, len(vals)):
        prev, cur = vals[i - 1], vals[i]
        col = GREEN if i < nl else RED
        ax.bar(i, prev - cur, bottom=cur, width=0.62, color=col, zorder=3, alpha=0.92)
        ax.plot([i - 1 + 0.31, i - 0.31], [prev, prev], color=FAINT, lw=1, ls=(0, (4, 3)), zorder=4)
        ax.text(i, (prev + cur) / 2, f"−{prev - cur:.3f}s", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=5)
    ax.text(nl - 1, vals[nl - 1] - 0.006, f"e1 {vals[nl - 1]:.3f}s", ha="center", va="top",
            fontsize=9, fontweight="bold", color=TEAL_D)
    ax.text(len(vals) - 1, vals[-1] - 0.006, f"{vals[-1]:.3f}s", ha="center", va="top",
            fontsize=9, fontweight="bold", color=RED)

    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=8, color=INK)
    ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_ylim(floor, top)
    ax.set_ylabel("time to 94% (s) — single A100, n=200")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.text(0.5, 0.008, "each bar = time that step saves; legit (green) measured n=200, cheats (red) = train-loop diagnostic / luckiest-host min / ~9 ms thermal cooldown.",
             ha="center", fontsize=7, color=FAINT)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"attribution_cheating.{ext}")
    plt.close(fig)
    print(f"wrote attribution_cheating.png/.svg  (floating; hiverge {vals[0]:.3f} -> e1 {vals[nl-1]:.3f} -> gamed {vals[-1]:.3f})")


# Iso-accuracy curriculum value by graph mode (the headline contingency result).
# For each graph mode we compare the progressive curriculum (full schedule) against
# full-resolution training SHORTENED to the same accuracy (S=252 steps). The bar gap
# is the curriculum's true value at matched accuracy; its sign flips with graphing.
ISO_MODES = [
    ("eager", "e1a_prog_eager_precomp", "iso_full32_s252_eager_n200", "no CUDA graphs"),
    ("per-step", "e1a_prog_perstep", "iso_full32_s252_perstep_n120", "per-step graphs"),
    ("mega", "e1a_prog_mega", "iso_full32_s252_mega_n200", "whole-run graph"),
]


def iso_curriculum():
    """Grouped bars: progressive curriculum vs accuracy-matched full-resolution,
    per graph mode. Positive gap (full-32 taller) = curriculum is faster = a win."""
    rows = []
    for mode, prog_lbl, iso_lbl, sub in ISO_MODES:
        dp, di = _load(prog_lbl), _load(iso_lbl)
        if dp and di:
            rows.append((mode, sub, dp["time"]["mean_s"], dp["time"]["std_s"],
                         di["time"]["mean_s"], di["time"]["std_s"]))
    if len(rows) < 2:
        print("iso_curriculum: need prog + iso cells; skipping")
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    w = 0.36
    for i, (mode, sub, pv, ps, iv, isd) in enumerate(rows):
        curriculum_wins = iv > pv  # full-32 slower => curriculum faster => win
        gap = pv - iv  # >0 means curriculum faster
        ax.bar(i - w / 2, pv, w, color=TEAL_D, zorder=3,
               label="resolution curriculum (284-step schedule)" if i == 0 else None)
        ax.bar(i + w / 2, iv, w, color=AMBER, zorder=3,
               label="full-32, shortened to 252 steps (same accuracy)" if i == 0 else None)
        ax.errorbar(i - w / 2, pv, yerr=ps, fmt="none", ecolor=FAINT, lw=1.1, capsize=3, zorder=4)
        ax.errorbar(i + w / 2, iv, yerr=isd, fmt="none", ecolor=FAINT, lw=1.1, capsize=3, zorder=4)
        ax.text(i - w / 2, pv + ps + 0.012, f"{pv:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(i + w / 2, iv + isd + 0.012, f"{iv:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
        vcol = TEAL_D if curriculum_wins else "#B23A48"
        verdict = "faster" if curriculum_wins else "slower"
        ax.text(i, max(pv, iv) + max(ps, isd) + 0.075,
                f"curriculum\n{abs(gap):.2f}s {verdict}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=vcol)

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{m}\n({s})" for m, s, *_ in rows], fontsize=10, color=INK)
    ax.set_ylim(1.6, 2.75)
    ax.set_ylabel("time to 94% accuracy (s) — single A100, lower is better")
    ax.set_title("At matched accuracy (~94.0%): resolution curriculum vs full resolution\n"
                 "Curriculum saves ~0.22 s with graphing (the standard setting); 'no graphs' is a control",
                 loc="left", fontsize=12)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="upper center", framealpha=0.9, ncol=1)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(PLOTS / f"iso_curriculum.{ext}")
    plt.close(fig)
    print(f"wrote iso_curriculum.png/.svg  ({len(rows)}/3 modes)")


if __name__ == "__main__":
    frontier()
    attribution()
    attribution_cheating()
    iso_curriculum()
