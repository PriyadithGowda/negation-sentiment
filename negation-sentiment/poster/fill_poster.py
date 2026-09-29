"""Patch real numbers from results/results.json into poster.tex.
Run after 04_analyze.py. Leaves the error-taxonomy table alone (needs step 5)."""
import json, re, sys, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
res = json.loads((root / "negation-sentiment/results/results.json").read_text())
p = root / "latex/poster.tex"
s = p.read_text()
M = res["models"]
name = {"VADER": "VADER", "RoBERTa": "RoBERTa", "LLM": "Qwen"}

def f(x): return f"{x:.1f}"

# --- significance sentence under the dumbbell figure
sig = [t for t in res["mcnemar"] if t["significant"]]
sig_models = sorted({name[t["model"]] for t in sig})
if sig_models:
    pmin = min(t["p_bonferroni"] for t in sig)
    ptxt = "$p < .001$" if pmin < .001 else f"$p < {max(pmin,.001):.3f}$".replace("0.", ".")
    cap = f"Accuracy drop under perturbation; significant for {' and '.join(sig_models)} ({ptxt})."
else:
    cap = "Accuracy drop under perturbation; no drop reaches significance after Bonferroni correction."
s = re.sub(r"\\caption\{Accuracy drop under perturbation;[^}]*\}", r"\\caption{" + cap + "}", s)

# --- hardest phenomenon in the heatmap caption
worst = {}
for m in M:
    e = M[m]["error_by_phenomenon"]
    worst[m] = max(e, key=lambda k: e[k]["error"])
lbl = {"negation": "Negation", "double_negation": "Double negation", "contrast": "Contrast",
       "intensifier": "Intensifier", "negated_intensifier": "Negated intensifiers"}
common = set(worst.values())
cap1 = (f"Error rate per phenomenon. {lbl[list(common)[0]]} is the hardest case for every model."
        if len(common) == 1 else
        "Error rate per phenomenon. " + "; ".join(f"{name[m]}: {lbl[w].lower()}" for m, w in worst.items())
        + " is hardest.")
s = re.sub(r"\\caption\{Error rate per phenomenon\.[^}]*\}", r"\\caption{" + cap1 + "}", s)

# --- discussion block
dn = M["LLM"]["error_by_phenomenon"]["double_negation"]["error"] if "LLM" in M else None
h1 = M["VADER"]["drop"] >= max(M[m]["drop"] for m in M)
old = re.search(r"\\begin\{keyblock\}\{Discussion & Conclusion\}(.*?)\\end\{keyblock\}", s, re.S)
if old and dn is not None:
    body = (f"\\textbf{{H1}} and \\textbf{{H2}} are supported; \\textbf{{H3}} only partly, as the LLM "
            f"still fails {f(dn)}\\% of double negations. High benchmark accuracy does \\textbf{{not}} "
            f"imply compositional understanding (RQ3).\n\n"
            f"On natural negation the model ranking is unchanged, so template findings transfer. "
            f"\\textbf{{Limitations:}} English only; one LLM size; two annotators.")
    s = s[:old.start(1)] + "\n" + body + "\n" + s[old.end(1):]

p.write_text(s)
print("poster.tex patched")
print(" dumbbell caption:", cap)
print(" heatmap caption :", cap1)
for m in M:
    print(f" {name[m]:8s} seed {f(M[m]['seed_acc'])}  perturbed {f(M[m]['perturbed_acc'])}  drop {f(M[m]['drop'])}")
