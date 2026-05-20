"""Regenerate Figure 1 (the end-to-end architecture diagram) cleanly.

Replaces the old hand-drawn version that had labels overflowing the boxes.
Boxes are sized to their text; the Logstash<->ML-inference call labels sit
in the gap between the two boxes, not on top of them.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "Figure_1.png"

# palette
C = dict(ingest="#FCE3B7", broker="#BfE3E0", proc="#D7C6F0", ml="#F3C9CE",
         store="#BFE0C2", viz="#FBE7A1", soar="#F2B27A", play="#F2B27A")

def box(ax, x, y, w, h, text, fc, fs=11, bold=True):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor="#333333", facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=5)

def arrow(ax, p, q, color="#222222", style="-", lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, linestyle=style,
        shrinkA=2, shrinkB=2, zorder=1))

fig, ax = plt.subplots(figsize=(6.6, 7.4))
ax.set_xlim(0, 10); ax.set_ylim(2.7, 12.6); ax.axis("off")

# rows (y) and the main flow column (x=4.3)
xc = 4.3
yF, yK, yL, yE, yV, yP = 11.8, 10.2, 8.6, 7.0, 5.4, 3.6
bw, bh = 2.0, 0.9

# data plane
box(ax, xc-1.15, yF, bw, bh, "Filebeat", C["ingest"])
box(ax, xc+1.15, yF, bw, bh, "Suricata", C["ingest"])
box(ax, xc, yK, 2.4, bh, "Apache Kafka", C["broker"])
# detection
box(ax, xc, yL, 2.2, bh, "Logstash", C["proc"])
ml_x, ml_w = 7.9, 2.9
box(ax, ml_x, yL, ml_w, 1.05, "ML Inference\n(XGBoost / Bi-LSTM)", C["ml"], fs=9.5)
box(ax, xc, yE, 2.6, bh, "Elasticsearch", C["store"])
# soar / viz
box(ax, xc-1.3, yV, 2.0, bh, "Kibana", C["viz"])
box(ax, xc+1.3, yV, 2.0, bh, "Wazuh AR", C["soar"])
for dx, label in [(-2.6, "iptables\nDROP"), (0.0, "TheHive\ncase"), (2.6, "Account\ndisable")]:
    box(ax, xc+dx, yP, 1.9, 0.95, label, C["play"], fs=9)

# arrows down the main flow
arrow(ax, (xc-1.15, yF-bh/2), (xc-0.4, yK+bh/2))
arrow(ax, (xc+1.15, yF-bh/2), (xc+0.4, yK+bh/2))
arrow(ax, (xc, yK-bh/2), (xc, yL+bh/2))
arrow(ax, (xc, yL-bh/2), (xc, yE+bh/2))
arrow(ax, (xc, yE-bh/2), (xc-1.3, yV+bh/2))
arrow(ax, (xc, yE-bh/2), (xc+1.3, yV+bh/2))
for dx in (-2.6, 0.0, 2.6):
    arrow(ax, (xc+1.3, yV-bh/2), (xc+dx, yP+0.95/2))

# Logstash <-> ML inference (request / response), labels ABOVE/BELOW the arrows in the gap
lr = xc + 1.1            # logstash right edge
mll = ml_x - ml_w/2      # ml left edge
arrow(ax, (lr, yL+0.18), (mll, yL+0.18), color="#7A3FB0", style=(0,(4,2)))
arrow(ax, (mll, yL-0.18), (lr, yL-0.18), color="#7A3FB0", style=(0,(4,2)))
midx = (lr+mll)/2
ax.text(midx, yL+0.5, "POST /predict", ha="center", va="bottom",
        fontsize=8, style="italic", color="#7A3FB0")
ax.text(midx, yL-0.5, "ml_label, ml_score", ha="center", va="top",
        fontsize=8, style="italic", color="#7A3FB0")

# section bands on the far left, fully inside the canvas
for y0, y1, name in [(yK-0.6, yF+0.6, "Data plane"),
                     (yE-0.6, yL+0.6, "Detection"),
                     (yP-0.6, yV+0.6, "SOAR")]:
    ax.text(0.55, (y0+y1)/2, name, ha="center", va="center", rotation=90,
            fontsize=10, fontweight="bold", color="#555555")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT, dpi=400, facecolor="white")
print(f"wrote {OUT}")
