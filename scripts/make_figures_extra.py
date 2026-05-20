"""Regenerate Figures 2, 3, 5, 8 as clean matplotlib figures.

Brings the screenshot-style Figs 2/3 into the same visual language as the
other plots, fixes the Fig 5 callout overlap and the Fig 8 capacity-ceiling
/ x-tick collision, and drops the redundant embedded titles (the numbered
caption is printed below each figure in the manuscript).
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
RES = ROOT / "results"
DPI = 400

# ---------- Figure 2: HDFS log excerpt (code-panel style) ----------
def fig2():
    lines = [
        "081109 20:36:15 INFO  DataXceiver      Receiving block blk_-1608999687919862906",
        "081109 20:38:07 INFO  PacketResponder  PacketResponder 1 for blk_38865049064139660 terminating",
        "081109 20:40:05 INFO  FSNamesystem     BLOCK* allocateBlock: blk_-1608999687919862906",
        "081109 20:40:15 INFO  DataXceiver      Receiving block blk_7503483334202473044",
        "081109 20:41:06 INFO  PacketResponder  Received blk_7503483334202473044 (67108864 B)",
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # clean, journal-style listing: plain white panel with a simple thin rule frame
    # (no rounded corners, no drop shadow, no grey screenshot chrome)
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90,
        facecolor="white", edgecolor="#999999", linewidth=0.8))
    ax.text(0.05, 0.87, "HDFS.log", family="monospace", fontsize=8.5,
            color="#555555", va="center", ha="left")
    ax.plot([0.04, 0.96], [0.81, 0.81], color="#cccccc", lw=0.7)
    y = 0.70
    for ln in lines:
        ax.text(0.05, y, ln, family="monospace", fontsize=7.2, color="#222222",
                va="center", ha="left")
        y -= 0.135
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(FIGS / "Figure_2.png", dpi=DPI, facecolor="white")
    plt.close(fig)

# ---------- Figure 3: BlockId -> Label sample (booktabs table) ----------
def fig3():
    rows = [
        ("0", "blk_-1608999687919862906", "Normal"),
        ("1", "blk_7503483334202473044",  "Normal"),
        ("2", "blk_-3544583377289625738", "Anomaly"),
        ("3", "blk_-9073992586687739851", "Normal"),
        ("4", "blk_7854771516489510726",  "Normal"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # tight central column block (leaves margins L/R)
    x_idx, x_blk, x_lab = 0.16, 0.24, 0.74
    L, R = 0.12, 0.90
    y_hdr = 0.84; dy = 0.125
    y_top, y_mid = 0.92, 0.78
    # header
    for x, c in [(x_idx, "#"), (x_blk, "BlockId"), (x_lab, "Label")]:
        ax.text(x, y_hdr, c, fontsize=10, fontweight="bold", va="center", ha="left")
    # booktabs rules
    ax.plot([L, R], [y_top, y_top], color="#333333", lw=1.4)   # top rule
    ax.plot([L, R], [y_mid, y_mid], color="#333333", lw=1.0)   # under header
    y = y_mid - 0.04
    for idx, blk, lab in rows:
        y -= dy
        anom = (lab == "Anomaly")
        ax.text(x_idx, y, idx, family="monospace", fontsize=9, va="center", ha="left", color="#444")
        ax.text(x_blk, y, blk, family="monospace", fontsize=9, va="center", ha="left", color="#222")
        ax.text(x_lab, y, lab, fontsize=9, va="center", ha="left",
                fontweight="bold" if anom else "normal",
                color="#B22222" if anom else "#222")
    ax.plot([L, R], [y - dy*0.6, y - dy*0.6], color="#333333", lw=1.4)  # bottom rule
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    fig.savefig(FIGS / "Figure_3.png", dpi=DPI, facecolor="white")
    plt.close(fig)

# ---------- Figure 4: HDFS Bi-LSTM training history + confusion matrix ----------
def fig4():
    m = json.load(open(RES / "hdfs_metrics.json"))
    h = m["history"]
    cm = m["confusion_matrix"]            # session-level [[TN,FP],[FN,TP]]
    n_test = m.get("n_test_sessions", sum(sum(r) for r in cm))
    best = m["best_epoch"]
    epochs = list(range(1, len(h["accuracy"]) + 1))
    BLUE, RED = "#1f5fa8", "#d62728"

    # side-by-side (landscape) so it matches the caption's "(left) ... (right)"
    # and stays short enough to never overflow a page top.
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    # --- (a) twin-axis training history ---
    ax2 = ax1.twinx()
    ax1.plot(epochs, h["accuracy"], color=BLUE, lw=1.8, label="train acc")
    ax1.plot(epochs, h["val_accuracy"], color=BLUE, lw=1.6, ls="--", label="val acc")
    ax2.plot(epochs, h["loss"], color=RED, lw=1.8, label="train loss")
    ax2.plot(epochs, h["val_loss"], color=RED, lw=1.6, ls="--", label="val loss")
    ax1.axvline(best, color="#999999", ls=":", lw=1.1)
    ax1.text(best + 0.15, ax1.get_ylim()[0] + 0.0015, f"ep {best}",
             color="#888888", fontsize=8, va="bottom")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy", color=BLUE)
    ax2.set_ylabel("Loss", color=RED)
    ax1.tick_params(axis="y", labelcolor=BLUE); ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("(a) Bi-LSTM training history (HDFS)")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="lower right", ncol=1, fontsize=7.5, framealpha=0.95)

    # --- (b) confusion matrix ---
    ax3.set_title(f"(b) Confusion matrix (n = {n_test:,})")
    labels = ["Normal", "Anomalous"]
    row_tot = [sum(r) for r in cm]
    for i in range(2):
        for j in range(2):
            frac = cm[i][j] / row_tot[i] if row_tot[i] else 0
            ax3.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                          facecolor=plt.cm.Blues(0.15 + 0.8 * frac), edgecolor="white", lw=2))
            ax3.text(j + 0.5, 1 - i + 0.5, f"{cm[i][j]:,}\n({frac*100:.2f}%)",
                     ha="center", va="center", fontsize=9.5,
                     color="white" if frac > 0.5 else "#222222")
    ax3.set_xlim(0, 2); ax3.set_ylim(0, 2)
    ax3.set_xticks([0.5, 1.5]); ax3.set_xticklabels(labels)
    ax3.set_yticks([1.5, 0.5]); ax3.set_yticklabels(labels)
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
    ax3.set_aspect("equal"); ax3.tick_params(length=0)
    for sp in ax3.spines.values(): sp.set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "Figure_4.png", dpi=DPI, facecolor="white")
    plt.close(fig)

# ---------- Figure 5: GUIDE per-class P/R/F1 ----------
def fig5():
    d = json.load(open(RES / "guide_metrics.json"))["per_class"]
    # support both old (Benign/Suspicious/Malicious) and new canonical Microsoft labels
    if "0_BenignPositive" in d:
        order = ["0_BenignPositive", "1_FalsePositive", "2_TruePositive"]
        names = ["BenignPositive", "FalsePositive", "TruePositive"]
        weak_idx, weak_txt = 1, "FalsePositive recall\nis the weak point"
    else:
        order = ["0_Benign", "1_Suspicious", "2_Malicious"]
        names = ["Benign", "Suspicious", "Malicious"]
        weak_idx, weak_txt = 1, "Suspicious recall\nis the weak point"
    P = [d[k]["precision"] for k in order]
    R = [d[k]["recall"] for k in order]
    F = [d[k]["f1-score"] for k in order]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    b1 = ax.bar(x-w, P, w, label="Precision", color="#4C78A8")
    b2 = ax.bar(x,   R, w, label="Recall",    color="#F58518")
    b3 = ax.bar(x+w, F, w, label="F1-score",  color="#54A24B")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.012,
                    f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    # text-only callout (no arrow) centered in the whitespace above the
    # Suspicious group — the wording names "recall" so no pointer is needed,
    # and this cannot overlap any bar or value label
    ax.text(x[weak_idx], 0.95, weak_txt,
            ha="center", va="top", fontsize=8, color="#B22222")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_5.png", dpi=DPI, facecolor="white")
    plt.close(fig)

# ---------- Figure 8: pipeline scaling ----------
def fig8():
    s = json.load(open(RES / "stress_test_reps.json"))["summary"]
    rates = [100, 250, 500, 1000]
    keys = ["100eps", "250eps", "500eps", "1000eps"]
    p50 = [s[k]["p50_mean"] for k in keys]; p50e = [s[k]["p50_std"] for k in keys]
    p95 = [s[k]["p95_mean"] for k in keys]; p95e = [s[k]["p95_std"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.errorbar(rates, p50, yerr=p50e, marker="o", capsize=3, color="#4C78A8", label="p50 latency")
    ax.errorbar(rates, p95, yerr=p95e, marker="s", capsize=3, color="#E45756", label="p95 latency")
    ax.axvline(958, color="#888888", linestyle="--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    # annotation placed in open space to the LEFT of the dashed line so it does
    # not overlap the line or the right-most p95 marker
    ax.text(720, ax.get_ylim()[1]*0.45, "mean achieved rate\n(958 events/s)",
            rotation=90, ha="center", va="center", fontsize=8, color="#555555")
    ax.set_xticks(rates); ax.set_xticklabels([str(r) for r in rates])
    ax.set_xlabel("Target throughput (events/s, log scale)")
    ax.set_ylabel("Ingest-to-ML-verdict latency (ms, log scale)")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "Figure_8.png", dpi=DPI, facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    import sys
    only = sys.argv[1:] or ["2", "3", "4", "5", "8"]
    if "2" in only: fig2()
    if "3" in only: fig3()
    if "4" in only: fig4()
    if "5" in only: fig5()
    if "8" in only: fig8()
    print(f"regenerated Figures {', '.join(only)}")
