"""Step 4: compute all numbers and figures for the poster.

Usage: python scripts/04_analyze.py

Inputs : data/test_suite_final.csv, data/natural_negation.csv, results/predictions.csv
Outputs: results/summary.md          <- human-readable tables with every number for the poster
         results/results.json        <- the same numbers, machine-readable
         results/fig_heatmap.pdf     <- poster Figure 1 (drop into the LaTeX folder)
         results/fig_dumbbell.pdf    <- poster Figure 2
         results/errors_to_code.csv  <- errors for the hand-coded error taxonomy (step 5)

Conventions
- LLM: descriptive numbers are the MEAN over the three prompts (range reported separately);
  significance tests use the majority vote of the three prompts.
- Confidence intervals: 95% percentile bootstrap, 1,000 resamples, resampling SEEDS
  (so the five perturbations of one seed stay together).
- McNemar (exact): per model x phenomenon, pairing each seed with its perturbed version;
  Bonferroni-corrected over all 15 tests.
"""
import json
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from common import DATA, RESULTS, PHENOMENA, PHENOMENON_LABELS, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)
B = 1000
MODELS = ["VADER", "RoBERTa", "LLM"]

suite = pd.read_csv(DATA / "test_suite_final.csv", dtype=str)
preds = pd.read_csv(RESULTS / "predictions.csv", dtype={"item_id": str})
nat_path = DATA / "natural_negation.csv"
nat = pd.read_csv(nat_path, dtype=str) if nat_path.exists() else None

# ------------------------------------------------------------------ correctness table
gold = pd.concat([suite[["item_id", "seed_id", "phenomenon", "gold", "text"]],
                  nat.assign(seed_id="")[["item_id", "seed_id", "phenomenon", "gold", "text"]]
                  if nat is not None else None], ignore_index=True)
df = preds.merge(gold, on="item_id", how="inner")
df["correct"] = (df["pred"] == df["gold"]).astype(float)

# per-prompt LLM rows keep prompt ids; build (a) LLM mean-correctness and (b) majority vote
llm = df[df["model"] == "LLM"]
prompts = sorted(llm["prompt_id"].unique())
if len(llm):
    vote = (llm.assign(p=(llm["pred"] == "pos").astype(int))
               .groupby(["item_id", "source", "seed_id", "phenomenon", "gold", "text"], as_index=False)["p"].mean())
    vote["pred"] = np.where(vote["p"] > 0.5, "pos", "neg")
    vote["correct"] = (vote["pred"] == vote["gold"]).astype(float)
    vote["model"], vote["prompt_id"] = "LLM", "vote"
    llm_mean = (llm.groupby(["item_id", "source", "seed_id", "phenomenon", "gold", "text"], as_index=False)["correct"].mean())
    llm_mean["model"], llm_mean["prompt_id"] = "LLM", "mean"
base = df[df["model"] != "LLM"].assign(prompt_id="-")
desc = pd.concat([base, llm_mean] if len(llm) else [base], ignore_index=True)       # for descriptive stats
test = pd.concat([base, vote] if len(llm) else [base], ignore_index=True)           # for tests / error coding
models = [m for m in MODELS if m in desc["model"].unique()]


def boot_ci(frame, stat, by_seed=True):
    """95% percentile CI; resamples seeds (clusters) if by_seed, else items."""
    if by_seed:
        groups = [g for _, g in frame.groupby("seed_id")]
        vals = [stat(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(B)]
    else:
        vals = [stat(frame.sample(len(frame), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


res = {"n_seeds": int(suite["seed_id"].nunique()), "n_suite_items": int(len(suite)),
       "n_natural": int(len(nat)) if nat is not None else 0, "llm_prompts": prompts, "models": {}}
agreement = RESULTS / "agreement.json"
if agreement.exists():
    res["agreement"] = json.loads(agreement.read_text())

S = desc[desc["source"] == "suite"]
for m in models:
    d = S[S["model"] == m]
    seed = d[d["phenomenon"] == "seed"]
    pert = d[d["phenomenon"].isin(PHENOMENA)]
    r = {"seed_acc": seed["correct"].mean() * 100, "perturbed_acc": pert["correct"].mean() * 100}
    r["drop"] = r["seed_acc"] - r["perturbed_acc"]
    r["drop_ci"] = [x * 100 for x in boot_ci(d, lambda f: f[f.phenomenon == "seed"].correct.mean()
                                                   - f[f.phenomenon.isin(PHENOMENA)].correct.mean())]
    r["error_by_phenomenon"] = {}
    for p in PHENOMENA:
        dp = d[d["phenomenon"] == p]
        err = (1 - dp["correct"].mean()) * 100
        ci = [100 - x * 100 for x in boot_ci(dp, lambda f: f.correct.mean())][::-1]
        r["error_by_phenomenon"][p] = {"error": err, "ci": ci, "n": int(len(dp))}
    if nat is not None:
        dn = desc[(desc["source"] == "natural") & (desc["model"] == m)]
        if len(dn):
            r["natural_acc"] = dn["correct"].mean() * 100
            r["natural_acc_ci"] = [x * 100 for x in boot_ci(dn, lambda f: f.correct.mean(), by_seed=False)]
    res["models"][m] = r

# LLM prompt sensitivity (mean and range across prompts)
if len(llm):
    per_prompt = {}
    for pid in prompts:
        d = llm[(llm["prompt_id"] == pid) & (llm["source"] == "suite")]
        per_prompt[pid] = {"seed_acc": d[d.phenomenon == "seed"].correct.mean() * 100,
                           "perturbed_acc": d[d.phenomenon.isin(PHENOMENA)].correct.mean() * 100}
    pa = [v["perturbed_acc"] for v in per_prompt.values()]
    res["llm_prompt_sensitivity"] = {"per_prompt": per_prompt, "perturbed_acc_range": [min(pa), max(pa)]}

# McNemar per model x phenomenon (seed vs its perturbation), Bonferroni
tests = []
T = test[test["source"] == "suite"]
for m in models:
    d = T[T["model"] == m]
    seed_ok = d[d.phenomenon == "seed"].set_index("seed_id")["correct"]
    for p in PHENOMENA:
        pert_ok = d[d.phenomenon == p].set_index("seed_id")["correct"]
        j = pd.concat([seed_ok.rename("s"), pert_ok.rename("p")], axis=1).dropna()
        tab = [[((j.s == 1) & (j.p == 1)).sum(), ((j.s == 1) & (j.p == 0)).sum()],
               [((j.s == 0) & (j.p == 1)).sum(), ((j.s == 0) & (j.p == 0)).sum()]]
        pval = mcnemar(tab, exact=True).pvalue
        tests.append({"model": m, "phenomenon": p, "b_seed_right_pert_wrong": int(tab[0][1]),
                      "c_seed_wrong_pert_right": int(tab[1][0]), "p": pval})
tdf = pd.DataFrame(tests)
tdf["p_bonferroni"] = (tdf["p"] * len(tdf)).clip(upper=1)
tdf["significant"] = tdf["p_bonferroni"] < 0.05
res["mcnemar"] = tdf.to_dict(orient="records")

(RESULTS / "results.json").write_text(json.dumps(res, indent=2, default=float))

# ------------------------------------------------------------------ errors to hand-code
err = test[(test["source"] == "suite") & test["phenomenon"].isin(PHENOMENA) & (test["correct"] == 0)]
err = err[["item_id", "phenomenon", "text", "gold", "model", "pred"]].sort_values(["model", "phenomenon", "item_id"])
err = err.assign(type_A="", type_B="", type_final="", comment="")
out_err = RESULTS / "errors_to_code.csv"
if out_err.exists():
    print(f"NOTE: {out_err} exists and was NOT overwritten (it may contain your coding). "
          f"Delete it to regenerate.")
else:
    err.to_csv(out_err, index=False)

# ------------------------------------------------------------------ summary.md
def f(x): return f"{x:.1f}"
L = ["# Results summary (auto-generated by 04_analyze.py)", ""]
if "agreement" in res:
    a = res["agreement"]
    L += [f"**Annotator agreement:** Cohen's kappa = {a['kappa']}, raw agreement = {a['raw_agreement']*100:.1f}% "
          f"(n = {a['n_items']}, {a['n_disagreements']} disagreements resolved by discussion)", ""]
L += [f"Seeds: {res['n_seeds']}, suite items: {res['n_suite_items']}, natural items: {res['n_natural']}", "",
      "## Accuracy: seeds vs perturbed (poster Figure 2)", "",
      "| Model | Seed acc. | Perturbed acc. | Drop | 95% CI of drop | Natural acc. |", "|---|---|---|---|---|---|"]
for m in models:
    r = res["models"][m]
    nat_s = f"{f(r['natural_acc'])} [{f(r['natural_acc_ci'][0])}, {f(r['natural_acc_ci'][1])}]" if "natural_acc" in r else "-"
    L.append(f"| {m} | {f(r['seed_acc'])} | {f(r['perturbed_acc'])} | {f(r['drop'])} | "
             f"[{f(r['drop_ci'][0])}, {f(r['drop_ci'][1])}] | {nat_s} |")
L += ["", "## Error rate (%) per phenomenon, with 95% CI (poster Figure 1)", "",
      "| Phenomenon | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
for p in PHENOMENA:
    cells = []
    for m in models:
        e = res["models"][m]["error_by_phenomenon"][p]
        cells.append(f"{f(e['error'])} [{f(e['ci'][0])}, {f(e['ci'][1])}]")
    L.append(f"| {PHENOMENON_LABELS[p]} | " + " | ".join(cells) + " |")
if "llm_prompt_sensitivity" in res:
    ps = res["llm_prompt_sensitivity"]
    L += ["", "## LLM prompt sensitivity", "",
          "| Prompt | Seed acc. | Perturbed acc. |", "|---|---|---|"]
    for pid, v in ps["per_prompt"].items():
        L.append(f"| {pid} | {f(v['seed_acc'])} | {f(v['perturbed_acc'])} |")
    L.append(f"\nPerturbed accuracy range across prompts: {f(ps['perturbed_acc_range'][0])}-{f(ps['perturbed_acc_range'][1])}")
L += ["", "## McNemar tests (seed vs perturbed, exact, Bonferroni x%d)" % len(tdf), "",
      "| Model | Phenomenon | seed right/pert wrong | seed wrong/pert right | p (corr.) | sig. |", "|---|---|---|---|---|---|"]
for t in res["mcnemar"]:
    L.append(f"| {t['model']} | {PHENOMENON_LABELS[t['phenomenon']]} | {t['b_seed_right_pert_wrong']} | "
             f"{t['c_seed_wrong_pert_right']} | {t['p_bonferroni']:.4f} | {'yes' if t['significant'] else 'no'} |")
L += ["", f"Errors exported for hand-coding: {len(err)} rows -> results/errors_to_code.csv"]
(RESULTS / "summary.md").write_text("\n".join(L))

# ------------------------------------------------------------------ figures (poster style)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
fam = "Lato" if any("Lato" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": fam, "font.size": 25, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
W, NAVY, GREY = 6.8, "#273c75", "#aab4c3"
names = {"VADER": "VADER", "RoBERTa": "RoBERTa", "LLM": "Qwen"}

v = np.array([[res["models"][m]["error_by_phenomenon"][p]["error"] for m in models] for p in PHENOMENA])
# Drawn as vector rectangles rather than imshow: imshow embeds a raster bitmap, which
# fell below the exam's 150 PPI rule for pixel images at A1 size. Rectangles stay vector.
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors, cm
fig, ax = plt.subplots(figsize=(W, 2.98))
norm = mcolors.Normalize(vmin=0, vmax=max(85, v.max()))
cmap = cm.get_cmap("Blues") if hasattr(cm, "get_cmap") else plt.get_cmap("Blues")
for i in range(len(PHENOMENA)):
    for j in range(len(models)):
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=cmap(norm(v[i, j])),
                               edgecolor="white", linewidth=5, zorder=1))
        ax.text(j, i, f"{v[i, j]:.0f}%", ha="center", va="center", fontsize=26, fontweight="bold",
                color="white" if v[i, j] > 40 else "#1c1f24", zorder=2)
ax.set_xlim(-.5, len(models) - .5); ax.set_ylim(len(PHENOMENA) - .5, -.5)
ax.set_xticks(range(len(models)), [names[m] for m in models]); ax.xaxis.tick_top()
ax.set_yticks(range(len(PHENOMENA)), [PHENOMENON_LABELS[p] for p in PHENOMENA])
ax.tick_params(length=0, pad=10)
for sp in ax.spines.values(): sp.set_visible(False)
fig.tight_layout(pad=.2); fig.savefig(RESULTS / "fig_heatmap.pdf"); plt.close(fig)

seed_a = [res["models"][m]["seed_acc"] for m in models]
pert_a = [res["models"][m]["perturbed_acc"] for m in models]
fig, ax = plt.subplots(figsize=(W, 2.95))
y = np.arange(len(models))[::-1]
lo = min(pert_a + seed_a)
for yi, s, p in zip(y, seed_a, pert_a):
    ax.plot([p, s], [yi, yi], color=GREY, lw=7, zorder=1, solid_capstyle="round")
    ax.text(max(s, p) + 2.5, yi, f"−{s - p:.1f}" if s >= p else f"+{p - s:.1f}", va="center",
            fontsize=25, color=NAVY, fontweight="bold")
ax.scatter(seed_a, y, s=380, color="white", edgecolor=NAVY, linewidth=3, zorder=2, label="Original")
ax.scatter(pert_a, y, s=380, color=NAVY, zorder=3, label="Perturbed")
ax.set_yticks(y, [names[m] for m in models]); ax.set_xlim(max(0, lo - 8), 112)
ax.set_xlabel("Accuracy (%)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e3e7ee", lw=1.2); ax.set_axisbelow(True); ax.set_ylim(-.75, len(models) - .25)
fig.legend(loc="upper center", ncol=2, frameon=False, handletextpad=.1, columnspacing=1.2, bbox_to_anchor=(.58, 1.0))
fig.tight_layout(pad=.2, rect=(0, 0, 1, .80)); fig.savefig(RESULTS / "fig_dumbbell.pdf"); plt.close(fig)

print((RESULTS / "summary.md").read_text())
print("\nFigures: results/fig_heatmap.pdf, results/fig_dumbbell.pdf -> copy into the poster folder.")
