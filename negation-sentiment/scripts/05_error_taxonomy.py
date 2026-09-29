"""Step 5: summarise the hand-coded error taxonomy.

NOT CARRIED OUT. This manual coding was planned but not performed; results/errors_to_code.csv
is exported with 196 rows and no codings. The error breakdown in poster Table 3 comes from the
automatic classification in 04_analyze.py instead. This script is kept for anyone who wants to
do the manual coding.

Usage: python scripts/05_error_taxonomy.py

Before running, both authors fill results/errors_to_code.csv independently:
  type_A, type_B  one of: polar_word | scope | clause | other   (definitions in ANNOTATION_GUIDELINES.md)
  type_final      only for rows where A and B disagree (agreed after discussion)
If there are many errors, you may code a random sample instead: add --sample 60 to the first
run to write a sampled file (results/errors_sample.csv) and code that one.
"""
import sys
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from common import RESULTS, RANDOM_SEED

CATS = ["polar_word", "scope", "clause", "other"]
NAMES = {"polar_word": "Polar word wins", "scope": "Scope error", "clause": "Wrong clause", "other": "Other"}

src = RESULTS / "errors_to_code.csv"
if "--sample" in sys.argv:
    n = int(sys.argv[sys.argv.index("--sample") + 1])
    e = pd.read_csv(src, dtype=str)
    s = e.groupby("model", group_keys=False).apply(lambda g: g.sample(min(len(g), n), random_state=RANDOM_SEED))
    s.to_csv(RESULTS / "errors_sample.csv", index=False)
    sys.exit(f"wrote {len(s)} rows (up to {n} per model) -> results/errors_sample.csv; code that file, then run without --sample")

path = RESULTS / "errors_sample.csv" if (RESULTS / "errors_sample.csv").exists() else src
e = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding_errors="replace").fillna("")
e.columns = [c.strip().lstrip("\ufeff") for c in e.columns]
for c in ["type_A", "type_B", "type_final"]:
    e[c] = e[c].str.strip().str.lower()
bad = ~e["type_A"].isin(CATS) | ~e["type_B"].isin(CATS)
if bad.any():
    sys.exit(f"{bad.sum()} rows in {path.name} are not fully coded (type_A/type_B must be one of {CATS}).")

kappa = cohen_kappa_score(e["type_A"], e["type_B"])
e["type"] = e["type_final"].where(e["type_final"].isin(CATS), e["type_A"].where(e["type_A"] == e["type_B"], ""))
open_rows = (e["type"] == "").sum()
if open_rows:
    sys.exit(f"kappa = {kappa:.3f}; {open_rows} disagreements still need type_final.")

share = pd.crosstab(e["type"], e["model"], normalize="columns").reindex(CATS).fillna(0) * 100
counts = e["model"].value_counts().to_dict()
print(f"Coding agreement: Cohen's kappa = {kappa:.3f} (n = {len(e)})")
print(share.round(1))
print("\nLaTeX rows for poster Table 3:")
models = [m for m in ["VADER", "RoBERTa", "LLM"] if m in share.columns]
for c in CATS:
    if share.loc[c].sum() == 0:
        continue
    print(f"{NAMES[c]} & " + " & ".join(f"{share.loc[c, m]:.0f}\\%" for m in models) + r"\\")
json.dump({"kappa": kappa, "n": len(e), "n_per_model": counts, "share": share.round(1).to_dict()},
          open(RESULTS / "taxonomy.json", "w"), indent=2)
