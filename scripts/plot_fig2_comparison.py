"""Draw our Phase 1 test results beside the paper's Fig. 2, in the same layout.

    python scripts/plot_fig2_comparison.py [RESULTS_JSON] [OUT_SVG]

Defaults: msm/analyses/phase1-analysis-test-pooled/results.json ->
docs/figures/phase1-fig2-comparison.svg. Writes a self-contained SVG with no
plotting dependencies. Run from the repo root.

Layout, arm order, colours and the outlined on-target bars follow Fig. 2 of
Li et al. (arXiv 2605.02087). The paper's bar heights are the value labels
printed on its figure; its error bars (±1 SEM over 4 training seeds) are not
reported numerically and are not drawn. Our error bars are question-clustered
95% bootstrap CIs for one released adapter per arm, so the two panels'
uncertainties are not comparable.
"""
import json
import os
import sys
from html import escape

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "msm/analyses/phase1-analysis-test-pooled/results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/figures/phase1-fig2-comparison.svg"

# (adapter label in results.json, legend text, fill) in Fig. 2's order.
ARMS = [
    ("baseline", "Baseline", "#e2e2e2"),
    ("cheese-aft", "AFT (cheese)", "#bdbdbd"),
    ("pro-affordability-spec-msm", "MSM (pro-affordability)", "#afcbeb"),
    ("pro-america-spec-msm", "MSM (pro-America)", "#e8a998"),
    ("pro-affordability-spec-msm-cheese-aft", "MSM (pro-affordability) + AFT (cheese)", "#45709b"),
    ("pro-america-spec-msm-cheese-aft", "MSM (pro-America) + AFT (cheese)", "#a83e2c"),
]
EVALS = [("affordability", "Pro-affordability Eval"), ("america", "Pro-America Eval")]
# Fig. 2 outlines the arms whose spec matches the eval.
ON_TARGET = {
    "affordability": {"pro-affordability-spec-msm", "pro-affordability-spec-msm-cheese-aft"},
    "america": {"pro-america-spec-msm", "pro-america-spec-msm-cheese-aft"},
}
# Fig. 2 value labels (also the targets table in src/msm_repro/README.md).
PAPER = {
    "affordability": [0.23, 0.32, 0.38, 0.28, 0.48, 0.29],
    "america": [0.38, 0.36, 0.36, 0.52, 0.38, 0.55],
}

res = json.load(open(RESULTS))
ours = {e: [res["by_eval"][e]["adapters"][a]["aligned_rate_all"]["value"] for a, _, _ in ARMS] for e, _ in EVALS}
n_q = {e: res["by_eval"][e]["n_questions"] for e, _ in EVALS}

INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e6e6e6"
W, H = 1180, 445
PANEL_W, PLOT_H, TOP, LEFT = 400, 300, 70, 70
GAP = 60  # between panels
YMAX = 0.6
BAR, BAR_GAP, GROUP_GAP = 21, 2, 46

out = []
add = out.append
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    'font-family="Helvetica, Arial, sans-serif">')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')


def text(x, y, s, size=12, anchor="middle", fill=INK, weight="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
        f'font-weight="{weight}">{escape(s)}</text>')


def panel(x0, title, subtitle, values, cis):
    y_of = lambda v: TOP + PLOT_H * (1 - v / YMAX)  # noqa: E731
    text(x0 + PANEL_W / 2, TOP - 38, title, 15, weight="bold")
    text(x0 + PANEL_W / 2, TOP - 20, subtitle, 11.5, fill=MUTED)
    for i in range(7):
        v = i / 10
        y = y_of(v)
        add(f'<line x1="{x0}" x2="{x0 + PANEL_W}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        text(x0 - 8, y + 4, f"{v:.1f}", 11, anchor="end", fill=MUTED)
    add(f'<line x1="{x0}" x2="{x0 + PANEL_W}" y1="{y_of(0):.1f}" y2="{y_of(0):.1f}" stroke="{MUTED}" stroke-width="1"/>')
    group_w = len(ARMS) * BAR + (len(ARMS) - 1) * BAR_GAP
    start = x0 + (PANEL_W - 2 * group_w - GROUP_GAP) / 2
    for g, (ev, ev_name) in enumerate(EVALS):
        gx = start + g * (group_w + GROUP_GAP)
        for i, (arm, legend, fill) in enumerate(ARMS):
            v = values[ev][i]
            bx = gx + i * (BAR + BAR_GAP)
            by = y_of(v)
            outline = ' stroke="#000000" stroke-width="2"' if arm in ON_TARGET[ev] else ""
            add(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BAR}" height="{y_of(0) - by:.1f}" fill="{fill}"{outline}>'
                f'<title>{escape(legend)}, {escape(ev_name)}: {v:.3f}</title></rect>')
            label_y = by - 5
            if cis is not None:
                lo, hi = cis[ev][i]
                cx = bx + BAR / 2
                add(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y_of(lo):.1f}" y2="{y_of(hi):.1f}" stroke="{INK}" stroke-width="1.2"/>')
                for yy in (y_of(lo), y_of(hi)):
                    add(f'<line x1="{cx - 3.5:.1f}" x2="{cx + 3.5:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{INK}" stroke-width="1.2"/>')
                label_y = y_of(hi) - 5
            text(bx + BAR / 2, label_y, f"{v:.2f}", 9.5)
        text(gx + group_w / 2, y_of(0) + 18, ev_name, 12)


paper_x = LEFT
ours_x = LEFT + PANEL_W + GAP
panel(paper_x, "Paper (Li et al., Fig. 2)", "values as labelled in the figure; 4-seed SEM bars not drawn", PAPER, None)
panel(
    ours_x,
    "Ours: released adapters, test split",
    "greedy, 256 tokens, both option orders; error bars: 95% CI over questions",
    ours,
    {e: [res["by_eval"][e]["adapters"][a]["aligned_rate_all"]["ci"] for a, _, _ in ARMS] for e, _ in EVALS},
)
ylab_y = TOP + PLOT_H / 2
add(f'<text transform="translate({LEFT - 44},{ylab_y}) rotate(-90)" font-size="12" text-anchor="middle" '
    f'fill="{INK}">Value-aligned preference rate</text>')

# Legend.
lx, ly = ours_x + PANEL_W + 28, TOP + 40
for i, (arm, legend, fill) in enumerate(ARMS):
    y = ly + i * 30
    add(f'<rect x="{lx}" y="{y - 11}" width="16" height="12" fill="{fill}"/>')
    parts = legend.split(" + ")
    text(lx + 24, y, parts[0] + (" +" if len(parts) > 1 else ""), 11.5, anchor="start")
    if len(parts) > 1:
        text(lx + 24, y + 14, parts[1], 11.5, anchor="start")
add(f'<rect x="{lx}" y="{ly + 6 * 30 - 1}" width="16" height="12" fill="#ffffff" stroke="#000000" stroke-width="2"/>')
text(lx + 24, ly + 6 * 30 + 10, "outline: spec matches eval", 11, anchor="start", fill=MUTED)

foot = (f"Ours: aligned_rate_all (non-choices count as not aligned); {n_q['affordability']} affordability and "
        f"{n_q['america']} America test questions, each asked in both option orders; one released seed per arm. "
        "Paper: 4 training seeds.")
text(LEFT, H - 34, foot, 11, anchor="start", fill=MUTED)
dec = res["by_eval"]["affordability"]["adapters"]["pro-america-spec-msm"]["decided_rate"]["value"]
others = min(res["by_eval"][e]["adapters"][a]["decided_rate"]["value"] for e, _ in EVALS for a, _, _ in ARMS
             if (e, a) != ("affordability", "pro-america-spec-msm"))
text(LEFT, H - 18, f"MSM (pro-America) on the affordability eval chooses an option in only {dec:.1%} of responses "
     f"(every other cell \u2265 {others:.1%}), which lowers its rate under this metric.", 11, anchor="start", fill=MUTED)
add("</svg>")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
open(OUT, "w").write("\n".join(out) + "\n")
print(f"wrote {OUT}")
