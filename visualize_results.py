"""
RAG Comparison Dashboard v2 — Fixed & Improved
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from pathlib import Path

DATA_FILE = Path("all_evaluations.json")
OUT_DIR   = Path("results_v2")
IND_DIR   = OUT_DIR / "individual"
OUT_DIR.mkdir(exist_ok=True)
IND_DIR.mkdir(exist_ok=True)

SYSTEMS = ["Baseline RAG", "GraphRAG", "LightRAG", "Hybrid RAG", "Blocks-DB"]
COLORS  = {
    "Baseline RAG": "#E53935",
    "GraphRAG":     "#1565C0",
    "LightRAG":     "#2E7D32",
    "Hybrid RAG":   "#6A1B9A",
    "Blocks-DB":    "#F57F17",
}
MARKERS = {
    "Baseline RAG": "o",
    "GraphRAG":     "s",
    "LightRAG":     "^",
    "Hybrid RAG":   "D",
    "Blocks-DB":    "P",
}
METRICS       = ["accuracy", "relevance", "completeness", "conciseness"]
METRIC_LABELS = ["Accuracy", "Relevance", "Completeness", "Conciseness"]
QTYPES        = ["factual", "explanation", "comparison", "reasoning"]
QTYPE_LABELS  = ["Factual", "Explanation", "Comparison", "Reasoning"]

IND_DPI = 180

def load():
    with open(DATA_FILE) as f:
        raw = json.load(f)
    averages, per_q, latency = {}, {}, {}
    for sys in SYSTEMS:
        val = raw[sys]
        items = val["items"]
        averages[sys] = {m: float(np.mean([i["scores"][m] for i in items])) for m in METRICS}
        averages[sys]["overall"] = float(val["overall"])
        latency[sys]  = float(val["avg_latency"])
        per_q[sys]    = items
    return averages, per_q, latency

def _spine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _save(fig, name, ind=True):
    path = IND_DIR / f"{name}.png"
    fig.savefig(path, dpi=IND_DPI, bbox_inches="tight", facecolor="white")
    print(f"  saved: {path}")
    plt.close(fig)


# ── FIX 1: Radar — y-axis starts at 2.5 so differences are visible ──────────
def chart_radar(averages):
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("white")

    N      = len(METRICS)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, size=12, fontweight="bold")
    ax.set_ylim(2.5, 5.2)                          # FIX: start at 2.5
    ax.set_yticks([3, 3.5, 4, 4.5, 5])
    ax.set_yticklabels(["3", "3.5", "4", "4.5", "5"], size=8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.4)

    for sys in SYSTEMS:
        scores = averages[sys]
        vals = [scores[m] for m in METRICS] + [scores[METRICS[0]]]
        ax.plot(angles, vals, "o-", linewidth=2.5, color=COLORS[sys], label=sys)
        ax.fill(angles, vals, alpha=0.10, color=COLORS[sys])

    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.2), fontsize=9, framealpha=0.8)
    ax.set_title("Radar Chart — All Metrics", size=14, fontweight="bold", pad=22)
    fig.tight_layout()
    _save(fig, "01_radar")


# ── FIX 2: Grouped bars — add value labels cleanly, consistent spacing ───────
def chart_grouped_bars(averages):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    x     = np.arange(len(METRICS))
    width = 0.14
    for i, sys in enumerate(SYSTEMS):
        offset = (i - 2) * width
        scores = [averages[sys][m] for m in METRICS]
        bars = ax.bar(x + offset, scores, width * 0.88,
                      label=sys, color=COLORS[sys], alpha=0.87,
                      edgecolor="white", linewidth=0.5)
        for bar, sc in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{sc:.1f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=COLORS[sys])

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=12)
    ax.set_ylim(0, 6.0)
    ax.set_ylabel("Score (1–5)", fontsize=11)
    ax.set_title("Performance by Metric", size=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _spine(ax)
    fig.tight_layout()
    _save(fig, "02_grouped_bar")


# ── FIX 3: Heatmap — text contrast fixed, colormap nicer ────────────────────
def chart_heatmap(averages):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    data = np.array([[averages[s][m] for m in METRICS] for s in SYSTEMS])
    cmap = LinearSegmentedColormap.from_list("rg", ["#d32f2f", "#fff176", "#388e3c"])
    im   = ax.imshow(data, cmap=cmap, vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.set_yticks(range(len(SYSTEMS)))
    ax.set_yticklabels(SYSTEMS, fontsize=11)
    for i in range(len(SYSTEMS)):
        for j in range(len(METRICS)):
            v  = data[i, j]
            tc = "white" if v >= 4.4 or v <= 2.0 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=tc)
    plt.colorbar(im, ax=ax, label="Score (1–5)", shrink=0.8)
    ax.set_title("Heatmap — Score Overview", size=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_heatmap")


# ── FIX 4: Overall ranking — cleaner, latency as annotation not inside bar ──
def chart_overall_bar(averages, latency):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    overalls = [averages[s]["overall"] for s in SYSTEMS]
    x = np.arange(len(SYSTEMS))
    bars = ax.bar(x, overalls, color=[COLORS[s] for s in SYSTEMS],
                  alpha=0.87, edgecolor="white", width=0.55)

    for i, (bar, val, sys) in enumerate(zip(bars, overalls, SYSTEMS)):
        # Score on top
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=COLORS[sys])
        # Latency inside bar (white text)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{latency[sys]:.1f}s", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SYSTEMS, fontsize=10)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("Overall Average Score (1–5)", fontsize=11)
    ax.set_title("Overall Ranking  (latency shown inside bars)", size=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _spine(ax)
    fig.tight_layout()
    _save(fig, "04_overall_ranking")


# ── FIX 5: Question type bars — add value labels ────────────────────────────
def chart_qtype_bars(per_q):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    x     = np.arange(len(QTYPES))
    width = 0.14
    for i, sys in enumerate(SYSTEMS):
        offset = (i - 2) * width
        scores = []
        for qt in QTYPES:
            typed = [item["avg_score"] for item in per_q[sys] if item["type"] == qt]
            scores.append(np.mean(typed) if typed else 0)
        bars = ax.bar(x + offset, scores, width * 0.88,
                      label=sys, color=COLORS[sys], alpha=0.87,
                      edgecolor="white", linewidth=0.5)
        for bar, sc in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                    f"{sc:.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=COLORS[sys])

    ax.set_xticks(x)
    ax.set_xticklabels(QTYPE_LABELS, fontsize=12)
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("Average Score (1–5)", fontsize=11)
    ax.set_title("Score by Question Type", size=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _spine(ax)
    fig.tight_layout()
    _save(fig, "05_score_by_qtype")


# ── FIX 6: Per-question lines — cleaner, highlight interesting questions ─────
def chart_per_question(per_q):
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")

    systems = list(per_q.keys())
    q_ids   = list(range(1, 21))

    for sys in systems:
        scores = [item["avg_score"] for item in per_q[sys]]
        ax.plot(q_ids, scores, marker=MARKERS[sys], linewidth=1.6,
                markersize=4.5, color=COLORS[sys], label=sys, alpha=0.9)

    # Shade the "interesting" questions where systems diverge
    divergent = [5, 6, 9, 11, 19, 20]
    for q in divergent:
        ax.axvspan(q - 0.4, q + 0.4, alpha=0.07, color="grey")

    ax.set_xticks(q_ids)
    ax.set_xticklabels([str(i) for i in q_ids], fontsize=8)
    ax.set_ylim(1.0, 5.6)
    ax.set_xlabel("Question #", fontsize=11)
    ax.set_ylabel("Score (1–5)", fontsize=11)
    ax.set_title("Per-Question Scores  (grey bands = high-variance questions)",
                 size=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.8, ncol=2)
    ax.grid(alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    _spine(ax)
    fig.tight_layout()
    _save(fig, "06_per_question_scores")


# ── FIX 7: Violin — y-axis starts at 1.5, better width ─────────────────────
def chart_violin(per_q):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    all_scores = [[item["avg_score"] for item in per_q[s]] for s in SYSTEMS]
    parts = ax.violinplot(all_scores, positions=range(1, len(SYSTEMS)+1),
                          showmedians=True, showextrema=True, widths=0.55)
    for pc, sys in zip(parts["bodies"], SYSTEMS):
        pc.set_facecolor(COLORS[sys]); pc.set_alpha(0.50); pc.set_edgecolor(COLORS[sys])
    for part_name in ("cmedians", "cbars", "cmaxes", "cmins"):
        if part_name in parts:
            parts[part_name].set_color("black"); parts[part_name].set_linewidth(1.2)

    rng = np.random.default_rng(42)
    for i, (scores, sys) in enumerate(zip(all_scores, SYSTEMS)):
        jitter = rng.uniform(-0.10, 0.10, len(scores))
        ax.scatter([i+1+j for j in jitter], scores, s=26,
                   color=COLORS[sys], alpha=0.85, zorder=3,
                   edgecolors="white", lw=0.4)

    ax.set_xticks(range(1, len(SYSTEMS)+1))
    ax.set_xticklabels(SYSTEMS, fontsize=10)
    ax.set_ylim(1.5, 5.6)                          # FIX: start at 1.5 not 1
    ax.set_ylabel("Score (1–5)", fontsize=11)
    ax.set_title("Score Distribution (Violin + Points)", size=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _spine(ax)
    fig.tight_layout()
    _save(fig, "07_score_distribution")


# ── FIX 8: Improvement vs baseline — cleaner labels, ns bar explanation ─────
def chart_diff_baseline(per_q):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    baseline = [item["avg_score"] for item in per_q["Baseline RAG"]]
    systems  = [s for s in SYSTEMS if s != "Baseline RAG"]
    diffs, pvals, cohens = [], [], []
    for sys in systems:
        scores = [item["avg_score"] for item in per_q[sys]]
        diff_arr = np.array(scores) - np.array(baseline)
        diff   = np.mean(diff_arr)
        t, p   = stats.ttest_rel(scores, baseline)
        d_val  = diff / np.std(diff_arr, ddof=1)
        diffs.append(diff); pvals.append(p); cohens.append(d_val)

    bars = ax.bar(np.arange(len(systems)), diffs,
                  color=[COLORS[s] for s in systems],
                  alpha=0.87, edgecolor="white", width=0.55)

    for bar, p, d, sys in zip(bars, pvals, cohens, systems):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{sig}\nd={d:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=COLORS[sys])

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(systems)))
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Score Difference vs Baseline RAG", fontsize=11)
    ax.set_title("Improvement vs Baseline  (* p<.05  ** p<.01  *** p<.001)",
                 size=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    # footnote explaining ns
    ax.text(0.99, 0.02,
            "ns = not significant (p > .05)",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            color="grey", style="italic")
    _spine(ax)
    fig.tight_layout()
    _save(fig, "08_improvement_vs_baseline")


# ── FIX 9: Quality vs Latency — fixed label positioning ─────────────────────
def chart_quality_latency(averages, latency):
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    for sys in SYSTEMS:
        ax.scatter(latency[sys], averages[sys]["overall"],
                   s=340, color=COLORS[sys], marker=MARKERS[sys],
                   zorder=5, edgecolors="white", linewidths=2.0)

    # dx = log10 multiplier for x, dy = absolute y offset, ha = text alignment
    label_cfg = {
        "Baseline RAG": dict(dx=-0.30, dy=-0.07, ha="right"),
        "Blocks-DB":    dict(dx= 0.10, dy= 0.05, ha="left"),
        "Hybrid RAG":   dict(dx=-0.08, dy= 0.05, ha="right"),
        "LightRAG":     dict(dx= 0.08, dy= 0.05, ha="left"),
        "GraphRAG":     dict(dx= 0.06, dy=-0.07, ha="left"),
    }
    for sys, cfg in label_cfg.items():
        x, y = latency[sys], averages[sys]["overall"]
        ax.text(x * (10 ** cfg["dx"]), y + cfg["dy"], sys,
                fontsize=9.5, fontweight="bold", color=COLORS[sys],
                ha=cfg["ha"], va="center")

    ax.set_xscale("log")
    ax.set_xlim(2.5, 28)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))
    ax.xaxis.set_major_locator(ticker.FixedLocator([3, 4, 5, 6, 8, 10, 13, 18, 25]))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(3.4, 5.15)
    ax.set_xlabel("Avg Latency (s) — log scale", fontsize=11)
    ax.set_ylabel("Overall Score (1–5)", fontsize=11)
    ax.set_title("Quality vs. Latency  (top-left = ideal)", size=14, fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    _spine(ax)
    fig.tight_layout()
    _save(fig, "09_quality_vs_latency")


# ── FIX 10: Stacked metric profile — unchanged logic, cleaner style ──────────
def chart_metric_profiles(averages):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")

    core_m = ["accuracy", "relevance", "completeness"]
    clrs   = ["#1976D2", "#F57C00", "#388E3C"]
    y      = np.arange(len(SYSTEMS))
    left   = np.zeros(len(SYSTEMS))

    for m, label, c in zip(core_m, ["Accuracy", "Relevance", "Completeness"], clrs):
        vals = [averages[s][m] / 3 for s in SYSTEMS]
        ax.barh(y, vals, 0.45, left=left, label=label, color=c, alpha=0.82, edgecolor="white")
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels(SYSTEMS, fontsize=11)
    ax.set_xlabel("Cumulative Score (Acc + Rel + Com each /3)", fontsize=10)
    ax.set_title("Stacked Metric Profile  (Acc + Rel + Com)", size=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _spine(ax)
    fig.tight_layout()
    _save(fig, "10_stacked_metric_profile")


# ── FIX 11: Quality-latency quadrant — better quadrant lines & labels ────────
def chart_quadrant(averages, latency):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    xs = [latency[s] for s in SYSTEMS]
    ys = [averages[s]["overall"] for s in SYSTEMS]
    med_x = np.median(xs)
    med_y = np.median(ys)

    ax.axvline(med_x, color="grey", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(med_y, color="grey", linestyle="--", linewidth=1, alpha=0.5)

    # Quadrant labels
    ax.text(0.02, 0.98, "Fast & High Quality\n✓ Best",
            transform=ax.transAxes, fontsize=8, color="#2e7d32", va="top", fontweight="bold",
            bbox=dict(fc="#e8f5e9", ec="#2e7d32", boxstyle="round,pad=0.3", alpha=0.85))
    ax.text(0.65, 0.98, "Slow & High Quality\n~ Expensive",
            transform=ax.transAxes, fontsize=8, color="#e65100", va="top",
            bbox=dict(fc="#fff3e0", ec="#e65100", boxstyle="round,pad=0.3", alpha=0.85))
    ax.text(0.02, 0.06, "Fast & Low Quality\n~ Limited",
            transform=ax.transAxes, fontsize=8, color="#757575", va="bottom",
            bbox=dict(fc="#f5f5f5", ec="#757575", boxstyle="round,pad=0.3", alpha=0.85))
    ax.text(0.65, 0.06, "Slow & Low Quality\n✗ Worst",
            transform=ax.transAxes, fontsize=8, color="#b71c1c", va="bottom",
            bbox=dict(fc="#ffebee", ec="#b71c1c", boxstyle="round,pad=0.3", alpha=0.85))

    # Tuned offsets for NEW latency values (no 237s outlier)
    offsets = {
        "Baseline RAG": (-55, 8),
        "GraphRAG":     (10, -14),
        "LightRAG":     (10,  6),
        "Hybrid RAG":   (-80, 8),
        "Blocks-DB":    (-72, -14),
    }
    for sys, xi, yi in zip(SYSTEMS, xs, ys):
        ax.scatter(xi, yi, s=280, color=COLORS[sys], marker=MARKERS[sys],
                   zorder=5, edgecolors="white", linewidths=1.8)
        ox, oy = offsets.get(sys, (8, 4))
        ax.annotate(sys, (xi, yi), xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=COLORS[sys])

    ax.set_xscale("log")
    ax.set_xlabel("Avg Latency (s) — log scale", fontsize=11)
    ax.set_ylabel("Overall Score (1–5)", fontsize=11)
    ax.set_title("Quality-Latency Quadrant\n(top-left = best trade-off)", size=14, fontweight="bold")
    ax.set_ylim(3.4, 5.3)
    ax.grid(alpha=0.2, linestyle="--")
    _spine(ax)
    fig.tight_layout()
    _save(fig, "11_quality_latency_quadrant")


def main():
    print("Loading data...")
    averages, per_q, latency = load()

    print("Generating improved individual charts...")
    chart_radar(averages)
    chart_grouped_bars(averages)
    chart_heatmap(averages)
    chart_overall_bar(averages, latency)
    chart_qtype_bars(per_q)
    chart_per_question(per_q)
    chart_violin(per_q)
    chart_diff_baseline(per_q)
    chart_quality_latency(averages, latency)
    chart_metric_profiles(averages)
    chart_quadrant(averages, latency)

    print(f"\nDone! → {IND_DIR}/")

if __name__ == "__main__":
    main()