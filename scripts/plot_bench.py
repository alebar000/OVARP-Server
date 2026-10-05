"""
Figure for the paper: time to first audio (TTFA) by reply length, full-reply vs.
sentence-level scheduling, one small multiple per LLM:TTS provider pair.

Each point is a median over turns. The bar behind it spans the interquartile range,
and faint dots are the individual turns. Single y-axis per panel, shared across panels.

  python scripts/plot_bench.py bench_results/<run_id> [--metric ttfa_ms] [--out fig_latency.pdf]

Author: Alexander Barquero Elizondo, Ph.D. -- UCR, ECCI/CITIC
License: MIT
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODE_COLOR = {"full": "#2a78d6", "sentence": "#eb6834"}   # validated categorical slots 1-2
MODE_LABEL = {"full": "Full-reply synthesis", "sentence": "Sentence-level TTS"}
CLASSES = ["short", "medium", "long"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--metric", default="ttfa_ms")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = Path(a.run_dir)
    rows = [r for r in csv.DictReader(open(run / "trials.csv", encoding="utf-8"))
            if not r["error"] and r.get(a.metric) not in ("", "None", None)]
    pairs = sorted({(r["llm"], r["tts"]) for r in rows})
    modes = [m for m in ("full", "sentence") if any(r["mode"] == m for r in rows)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.3 * len(pairs) + 0.6, 2.4), sharey=True, squeeze=False)
    ymax = max(float(r[a.metric]) for r in rows) / 1000 * 1.08
    for ax, (llm, tts) in zip(axes[0], pairs):
        for mi, m in enumerate(modes):
            off = (mi - (len(modes) - 1) / 2) * 0.22
            xs, meds = [], []
            for ci, c in enumerate(CLASSES):
                v = [float(r[a.metric]) / 1000 for r in rows
                     if (r["llm"], r["tts"], r["mode"], r["length_class"]) == (llm, tts, m, c)]
                if not v:
                    continue
                x = ci + off
                ax.scatter([x] * len(v), v, s=6, color=MODE_COLOR[m], alpha=0.18, linewidths=0, zorder=1)
                ax.plot([x, x], [q(v, .25), q(v, .75)], color=MODE_COLOR[m], lw=4, alpha=0.45,
                        solid_capstyle="round", zorder=2)
                med = statistics.median(v)
                ax.scatter([x], [med], s=26, color=MODE_COLOR[m], edgecolors="white", linewidths=1.2, zorder=3)
                xs.append(x)
                meds.append(med)
            ax.plot(xs, meds, color=MODE_COLOR[m], lw=1.2, zorder=2, label=MODE_LABEL[m])
        ax.set_xticks(range(len(CLASSES)), [c.capitalize() for c in CLASSES])
        ax.set_title(f"LLM {llm} / TTS {tts}", color=INK, fontsize=8.5, loc="left")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("Requested reply length")
    axes[0][0].set_ylabel({"ttfa_ms": "Time to first audio (s)", "speech_end_ms": "End of speech (s)",
                           "done_ms": "All audio sent (s)"}.get(a.metric, a.metric))
    axes[0][-1].legend(frameon=False, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    out = Path(a.out) if a.out else run / f"fig_{a.metric}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print("Wrote", out)


if __name__ == "__main__":
    main()
