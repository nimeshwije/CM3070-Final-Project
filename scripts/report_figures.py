"""Every figure in the final report, generated in one consistent style.

I got tired of figures drifting apart visually as I regenerated them one by
one, so this script makes all of them in a single run with one shared
rcParams block and colour palette.

Run from anywhere:   python scripts/report_figures.py
Reads  reports/final_experiments.json (made by scripts/report_experiments.py)
Writes reports/figures/*.png
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # i.e. the evo-advisor folder
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from advisor.data import fetch_prices
from advisor.genome import Genome
from advisor.backtest import run_backtest
from advisor.evaluation import evaluate_split

OUT = os.path.join(ROOT, "reports", "figures"); os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(ROOT, "reports", "final_experiments.json")))
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.dpi": 170, "savefig.bbox": "tight", "savefig.facecolor": "white"})
champ = Genome.from_dict(D["A_champion"]["genome"])
aapl_snap = fetch_prices("AAPL", start="2015-01-01", end="2024-11-29").prices
aapl_full = fetch_prices("AAPL", start="2015-01-01").prices
ev = evaluate_split(aapl_snap, champ, 0.7, cost=0.001)

def pct(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

# ---------- Fig 4.1 GA convergence ----------
h = pd.DataFrame(D["A_champion"]["history"])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(h["generation"], h["best_fitness"], color=BLUE, lw=2, label="Best fitness")
ax.plot(h["generation"], h["mean_fitness"], color=ORANGE, lw=1.6, label="Population mean")
ax.set_xlabel("Generation"); ax.set_ylabel("Walk-forward fitness\n(mean − 0.5·std of per-fold Sharpe)")
ax.axhline(0, color="#c9c8c2", lw=0.8)
ax.annotate(f"best {h['best_fitness'].iloc[-1]:.2f}", (h["generation"].iloc[-1], h["best_fitness"].iloc[-1]),
            xytext=(-40, 8), textcoords="offset points", color=MUTED, fontsize=9)
ax.legend(loc="lower right"); fig.savefig(f"{OUT}/fig4_1_convergence.png"); plt.close(fig)

# ---------- Fig 4.2 equity in/out of sample (snapshot) ----------
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4))
for ax, res, rng, title in ((axes[0], ev.train_result, ev.train_range, "In-sample (training window)"),
                            (axes[1], ev.test_result, ev.test_range, "Out-of-sample (held-out window)")):
    ax.plot(res.equity.index, res.equity.values, color=BLUE, lw=1.8, label="Evolved strategy")
    ax.plot(res.benchmark_equity.index, res.benchmark_equity.values, color=ORANGE, lw=1.4, label="Buy & hold")
    ax.set_title(f"{title}: {rng[0]} to {rng[1]}", fontsize=10, loc="left", color=INK)
    ax.set_ylabel("Growth of 1.0 (net of costs)")
    m = res.metrics
    ax.text(0.01, 0.97, f"Strategy: Sharpe {m['sharpe']:.2f}, max DD {m['max_drawdown']:.0%}\n"
            f"Buy & hold: Sharpe {m['benchmark_sharpe']:.2f}, max DD {m['benchmark_max_drawdown']:.0%}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
axes[0].legend(loc="center left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig4_2_equity.png"); plt.close(fig)

# ---------- Fig 5.1 OOS drawdown ----------
fig, ax = plt.subplots(figsize=(7.2, 3.2))
eq, beq = ev.test_result.equity, ev.test_result.benchmark_equity
dd, bdd = eq / eq.cummax() - 1, beq / beq.cummax() - 1
ax.fill_between(dd.index, dd.values, 0, color=BLUE, alpha=0.35, lw=0, label="Evolved strategy")
ax.plot(bdd.index, bdd.values, color=ORANGE, lw=1.4, label="Buy & hold")
ax.set_ylabel("Drawdown from peak"); pct(ax); ax.legend(loc="lower left")
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_title(f"Out-of-sample window {ev.test_range[0]} to {ev.test_range[1]}", fontsize=10, loc="left")
fig.savefig(f"{OUT}/fig5_1_drawdown.png"); plt.close(fig)

# ---------- Fig 5.2 multi-seed ----------
c = pd.DataFrame(D["C_multiseed"])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=False)
for ax, col, title in ((axes[0], "oos_sharpe", "Out-of-sample Sharpe ratio"), (axes[1], "is_sharpe", "In-sample Sharpe ratio")):
    for i, (cfg, colr) in enumerate((("prototype", ORANGE), ("hardened", BLUE))):
        v = c[c.config == cfg][col].values
        jitter = (np.random.RandomState(1).rand(len(v)) - 0.5) * 0.18
        ax.scatter(np.full(len(v), i) + jitter, v, color=colr, s=34, zorder=3, edgecolor="white", lw=0.8)
        ax.hlines(v.mean(), i - 0.22, i + 0.22, color=colr, lw=2.4, zorder=4)
        ax.text(i + 0.26, v.mean(), f"mean {v.mean():.2f}\nsd {v.std(ddof=1):.2f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Prototype\nconfiguration", "Hardened\nengine"])
    ax.set_xlim(-0.5, 1.9); ax.set_title(title, fontsize=10, loc="left"); ax.axhline(0, color="#c9c8c2", lw=0.8)
axes[0].axhline(c["bh_sharpe"].iloc[0], color=GREY, lw=1, ls="--")
axes[0].text(-0.45, c["bh_sharpe"].iloc[0] + 0.02, "buy & hold", color=MUTED, fontsize=8)
fig.suptitle("Ten independent GA runs (seeds 1–10), AAPL, same data and split", fontsize=10, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_2_multiseed.png"); plt.close(fig)

# ---------- Fig 5.3 forward test: full OOS to 2026 with cut-off line ----------
warm = champ.long_window + 5
start = pd.Timestamp(ev.test_range[0])
idx = aapl_full.index.get_indexer([start], method="bfill")[0]
chunk = aapl_full.iloc[idx - warm:]
res = run_backtest(chunk, champ, 0.001)
r = res.daily_returns[res.daily_returns.index >= start]; b = chunk.pct_change().fillna(0)[chunk.index >= start]
eqf, beqf = (1 + r).cumprod(), (1 + b).cumprod()
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(eqf.index, eqf.values, color=BLUE, lw=1.8, label="Evolved strategy (rule frozen Dec 2021)")
ax.plot(beqf.index, beqf.values, color=ORANGE, lw=1.4, label="Buy & hold")
cut = pd.Timestamp("2024-11-29")
ax.axvline(cut, color=GREY, lw=1, ls="--"); ax.axvspan(cut, eqf.index[-1], color="#f0efec", zorder=0)
ax.text(cut, ax.get_ylim()[1] * 0.98, "  data unseen when the\n  rule was evolved and\n  the draft was written", va="top", fontsize=8, color=MUTED)
ax.set_ylabel("Growth of 1.0 (net of costs)"); ax.legend(loc="upper left", fontsize=8.5)
fig.savefig(f"{OUT}/fig5_3_forward.png"); plt.close(fig)

# ---------- Fig 5.4 cross-ticker dumbbells (Sharpe and max drawdown) ----------
g = [r for r in D["G_cross_ticker"] if r["ticker"] != "SNDK"]
g = sorted(g, key=lambda r: r["test"]["benchmark_total_return"])
SINGLE = {"ADBE", "PODD", "WBD"}
names = [r["ticker"] + (" *" if r["ticker"] in SINGLE else "") for r in g]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.6), sharey=True)
y = np.arange(len(g))
for ax, key, bkey, title in ((axes[0], "sharpe", "benchmark_sharpe", "Out-of-sample Sharpe ratio"),
                             (axes[1], "max_drawdown", "benchmark_max_drawdown", "Out-of-sample maximum drawdown")):
    s = [r["test"][key] for r in g]; bh = [r["test"][bkey] for r in g]
    ax.hlines(y, bh, s, color="#d6d5cf", lw=2, zorder=1)
    ax.scatter(bh, y, color=ORANGE, s=40, zorder=3, label="Buy & hold", edgecolor="white")
    ax.scatter(s, y, color=BLUE, s=40, zorder=3, label="Evolved rule", edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_title(title, fontsize=10, loc="left")
    ax.axvline(0, color="#c9c8c2", lw=0.8)
pct(axes[1], "x"); axes[0].legend(loc="lower right", fontsize=8.5)
fig.text(0.01, -0.02, "Tickers ordered by buy-and-hold return over the test window (weakest at top). "
         "Unmarked tickers share ONE rule evolved jointly on all eleven (multi-asset mode); * = rule evolved on that ticker alone. 70/30 split, 2015–2026.", fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_4_cross_ticker.png"); plt.close(fig)

# ---------- Fig 5.5 rule transfer ----------
hrows = sorted(D["H_transfer"], key=lambda r: r["benchmark_sharpe"])
fig, ax = plt.subplots(figsize=(7.2, 3.9))
y = np.arange(len(hrows)); s = [r["sharpe"] for r in hrows]; bh = [r["benchmark_sharpe"] for r in hrows]
ax.hlines(y, bh, s, color="#d6d5cf", lw=2, zorder=1)
ax.scatter(bh, y, color=ORANGE, s=40, zorder=3, label="Buy & hold", edgecolor="white")
ax.scatter(s, y, color=BLUE, s=40, zorder=3, label="AAPL rule, unchanged", edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels([r["ticker"] for r in hrows]); ax.axvline(0, color="#c9c8c2", lw=0.8)
ax.set_xlabel("Sharpe ratio, 2021-12-08 to 2024-11-29, net of costs"); ax.legend(loc="lower right", fontsize=8.5)
fig.savefig(f"{OUT}/fig5_5_transfer.png"); plt.close(fig)

# ======================= schematic diagrams =======================
def box(ax, x, y, w, h, text, fc="#eef4fc", ec=BLUE, fs=8.5, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(ax, x1, y1, x2, y2, text="", color=MUTED, fs=7.5, tx=0, ty=0.06):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=color, lw=1.1))
    if text: ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", fontsize=fs, color=MUTED)
def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, 10); ax.set_ylim(0, 10 * h / w); ax.axis("off"); ax.grid(False); return fig, ax

# ---------- Fig 3.1 architecture ----------
fig, ax = canvas(8.6, 5.0)
ax.add_patch(Rectangle((0.15, 3.0), 9.7, 2.7, fc="#f7f9fd", ec="#c9d6ea", lw=1, ls="--"))
ax.text(0.3, 5.45, "TRAINING SYSTEM  (administrator only; runs offline)", fontsize=8.5, color=BLUE, fontweight="bold")
ax.add_patch(Rectangle((0.15, 0.15), 9.7, 2.45, fc="#fdf8f5", ec="#ecd3c6", lw=1, ls="--"))
ax.text(0.3, 2.38, "ONLINE ADVISOR  (any user; loads five integers, fetches recent prices, emits a signal)", fontsize=8.5, color=ORANGE, fontweight="bold")
bw, bh, yb = 2.05, 1.05, 3.2
box(ax, 0.35, yb, bw, bh, "Data ingestion\nyfinance → CSV cache\n→ synthetic fallback", fs=7.8)
box(ax, 2.75, yb, bw, bh, "Evolutionary engine\nGA over 5-gene genome,\nwalk-forward fitness", fs=7.8)
box(ax, 5.15, yb, bw, bh, "Backtest & evaluation\n1-day lag, 10 bp costs,\nout-of-sample vs B&H", fs=7.8)
box(ax, 7.55, yb, bw, bh, "Rule artifact\nmodels/<ticker>.json\ngenome · metrics · GA log", fc="#fff8e6", ec="#c98500", fs=7.8)
box(ax, 0.35, 4.45, 9.25, 0.7, "Entry points: admin web area (login, CSRF, lockout; background job with live progress)  or  command line  →  one shared train_rule() pipeline", fc="white", ec=GREY, fs=7.2)
for x in (2.4, 4.8, 7.2): arrow(ax, x, yb + bh/2, x + 0.35, yb + bh/2)
arrow(ax, 3.95, 4.45, 3.95, yb + bh + 0.02)
yo = 0.5
box(ax, 0.35, yo, bw, bh, "Recent prices\n(same ingestion module,\ncached)", fs=7.8)
box(ax, 2.75, yo, bw, bh, "Rule engine\ndeterministic BUY /\nHOLD / SELL + reasons", fc="#fdf1ec", ec=ORANGE, fs=7.8)
box(ax, 5.15, yo, bw, bh, "Explanation layer\nlocal LLM (Ollama) →\nguardrail → template", fc="#fdf1ec", ec=ORANGE, fs=7.8)
box(ax, 7.55, yo, bw, bh, "Web interface\nFlask page + JSON API,\n'Do you hold it?'", fc="#fdf1ec", ec=ORANGE, fs=7.8)
for x in (2.4, 4.8, 7.2): arrow(ax, x, yo + bh/2, x + 0.35, yo + bh/2)
arrow(ax, 8.575, yb, 3.8, yo + bh + 0.02, "", color="#c98500")
ax.text(6.3, 2.5, "loads the evolved rule", fontsize=7.5, color="#c98500")
fig.savefig(f"{OUT}/fig3_1_architecture.png"); plt.close(fig)

# ---------- Fig 3.2 rule as state machine + position table ----------
fig, ax = canvas(8.4, 3.4)
box(ax, 0.6, 1.4, 2.3, 1.3, "IN CASH\n(flat, earning 0)", fc="#f0efec", ec=GREY, bold=True)
box(ax, 5.1, 1.4, 2.3, 1.3, "IN MARKET\n(long the asset)", fc="#eef4fc", ec=BLUE, bold=True)
ax.add_patch(FancyArrowPatch((2.9, 2.45), (5.1, 2.45), arrowstyle="-|>", mutation_scale=12, color=BLUE, lw=1.3, connectionstyle="arc3,rad=-0.25"))
ax.text(4.0, 3.45, "ENTER when  SMA(short) > SMA(long)  AND  RSI < buy threshold", ha="center", fontsize=8.5, color=BLUE)
ax.add_patch(FancyArrowPatch((5.1, 1.65), (2.9, 1.65), arrowstyle="-|>", mutation_scale=12, color=ORANGE, lw=1.3, connectionstyle="arc3,rad=-0.25"))
ax.text(4.0, 0.55, "EXIT when  SMA(short) < SMA(long)  OR  RSI > sell threshold", ha="center", fontsize=8.5, color=ORANGE)
ax.text(4.0, 0.15, "Genome = (short window, long window, RSI period, RSI buy, RSI sell); champion = (5, 222, 30, 49, 76)", ha="center", fontsize=8, color=MUTED)
# the little position-action table on the right of the state machine
tx, ty = 7.75, 3.25
ax.text(tx, ty, "Position-aware action", fontsize=8.5, fontweight="bold", color=INK)
rows = [("rule", "holds?", "action"), ("in market", "yes", "HOLD"), ("in market", "no", "BUY"), ("in cash", "yes", "SELL"), ("in cash", "no", "HOLD")]
for i, r in enumerate(rows):
    for j, cell in enumerate(r):
        ax.text(tx + [0, 0.85, 1.65][j], ty - 0.4 - i * 0.4, cell, fontsize=7.2, color=INK if i else MUTED, fontweight="bold" if (i and j == 2) else "normal")
fig.savefig(f"{OUT}/fig3_2_state_machine.png"); plt.close(fig)

# ---------- Fig 3.3 recommendation + guardrail flow ----------
fig, ax = canvas(8.6, 3.8)
Yc = 2.9; bw, bh = 1.95, 1.15
box(ax, 0.25, Yc, 1.6, bh, "Rule engine\n(deterministic)", fc="#eef4fc", ec=BLUE, bold=True, fs=8)
box(ax, 2.15, Yc, bw, bh, "Decision object\naction · stance · reasons\nindicators · rule text", fc="#fff8e6", ec="#c98500", fs=7.6)
box(ax, 4.4, Yc, bw, bh, "Prompt: 'restate, do\nnot reconsider' →\nlocal LLM (llama3.2)", fc="#fdf1ec", ec=ORANGE, fs=7.6)
box(ax, 6.65, Yc, 2.0, bh, "Guardrail check\nlength · names the action\n· no rival action", fc="#fdf1ec", ec=ORANGE, fs=7.6)
box(ax, 4.4, 0.45, bw, 1.0, "Template explanation\nbuilt from reasons only", fc="#f0efec", ec=GREY, fs=7.6)
box(ax, 6.65, 0.45, 2.0, 1.0, "Shown to user, with\ndisclaimer; source labelled\n('LLM' or 'rule-based')", fc="white", ec=INK, fs=7.6)
arrow(ax, 1.85, Yc + bh/2, 2.15, Yc + bh/2); arrow(ax, 4.1, Yc + bh/2, 4.4, Yc + bh/2); arrow(ax, 6.35, Yc + bh/2, 6.65, Yc + bh/2)
arrow(ax, 7.65, Yc, 7.65, 1.47); ax.text(7.75, 2.2, "pass", fontsize=7.5, color=MUTED)
arrow(ax, 6.65, Yc + 0.15, 6.35, 1.3); ax.text(5.55, 2.3, "fail, empty or\nOllama unreachable", fontsize=7.2, color=MUTED, ha="center")
arrow(ax, 3.1, Yc, 4.4, 1.0); ax.text(3.75, 2.15, "LLM disabled", fontsize=7.2, color=MUTED)
arrow(ax, 6.35, 0.95, 6.65, 0.95)
ax.text(0.25, 1.45, "The model never sees prices and cannot\nchange the decision: it receives a finished\nDecision object, and its output is checked\nagainst that object before display.", fontsize=7.8, color=MUTED, va="top")
fig.savefig(f"{OUT}/fig3_3_guardrail_flow.png"); plt.close(fig)

# ---------- Fig 4.4 walk-forward fitness schematic ----------
fig, ax = canvas(7.6, 2.6)
ax.text(0.2, 3.0, "Full history 2015-01-02 → 2024-11-29 (2,495 trading days), split chronologically 70 / 30", fontsize=8.5, color=INK)
x0, w = 0.3, 6.6
for k in range(4):
    ax.add_patch(Rectangle((x0 + k * w / 4, 1.6), w / 4 - 0.04, 0.9, fc=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"][k], ec="white"))
    ax.text(x0 + k * w / 4 + w / 8, 2.05, f"fold {k+1}\nSharpe s{k+1}", ha="center", va="center", fontsize=8, color=INK if k < 2 else "white")
ax.add_patch(Rectangle((x0 + w + 0.1, 1.6), 2.9, 0.9, fc="#fdf1ec", ec=ORANGE, lw=1.2))
ax.text(x0 + w + 0.1 + 1.45, 2.05, "held-out test window\nnever seen during search", ha="center", va="center", fontsize=8, color=INK)
ax.text(x0 + w / 2, 1.3, "training window (GA searches here)", ha="center", fontsize=8, color=MUTED)
ax.text(x0, 0.55, "fitness  =  mean(s1..s4)  −  0.5 · std(s1..s4)  −  sparsity penalty if trades/year < 2      (costs of 10 bp per side charged inside every fold)",
        fontsize=8.5, color=INK)
ax.text(x0, 0.15, "A rule must perform in every sub-period, not once; indicators warm up on preceding data, so no fold sees the future.", fontsize=8, color=MUTED)
fig.savefig(f"{OUT}/fig4_4_walkforward.png"); plt.close(fig)
print("done", sorted(os.listdir(OUT)))
