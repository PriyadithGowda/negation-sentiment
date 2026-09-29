"""Step 2: check the hand-written test suite and compute inter-annotator agreement.

Usage: python scripts/02_agreement.py

Reads data/test_suite.csv (columns label_A, label_B filled independently by the two authors).
- reports Cohen's kappa overall and per phenomenon
- writes data/disagreements.csv (resolve them together and write the result into 'gold')
- rows where A == B get gold filled automatically
- once every row has a gold label, writes data/test_suite_final.csv
"""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from common import DATA, RESULTS, LABELS

path = DATA / "test_suite.csv"
df = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding_errors="replace").fillna("")
df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
for c in ["text", "label_A", "label_B", "gold"]:
    df[c] = df[c].str.strip()
for c in ["label_A", "label_B", "gold"]:
    df[c] = df[c].str.lower()

problems = []
if (df["text"] == "").any():
    problems.append(f"{(df['text'] == '').sum()} rows have no text yet")
for c in ["label_A", "label_B"]:
    bad = ~df[c].isin(LABELS)
    if bad.any():
        problems.append(f"{bad.sum()} rows have a missing/invalid {c} (use pos or neg)")
if problems:
    print("Not ready:\n  - " + "\n  - ".join(problems))
    sys.exit(1)

kappa = cohen_kappa_score(df["label_A"], df["label_B"])
per = {p: round(cohen_kappa_score(g["label_A"], g["label_B"]), 3) if g["label_A"].nunique() + g["label_B"].nunique() > 2 else None
       for p, g in df.groupby("phenomenon")}
agree = (df["label_A"] == df["label_B"]).mean()
print(f"Cohen's kappa = {kappa:.3f}   raw agreement = {agree:.1%}   n = {len(df)}")
for p, k in per.items():
    print(f"  {p:22s} kappa = {k}")

auto = (df["gold"] == "") & (df["label_A"] == df["label_B"])
df.loc[auto, "gold"] = df.loc[auto, "label_A"]
dis = df[df["label_A"] != df["label_B"]]
dis.to_csv(DATA / "disagreements.csv", index=False)
df.to_csv(path, index=False)

import json
json.dump({"kappa": round(float(kappa), 3), "raw_agreement": round(float(agree), 3), "n_items": int(len(df)),
           "n_disagreements": int(len(dis))}, open(RESULTS / "agreement.json", "w"), indent=2)

missing_gold = ~df["gold"].isin(LABELS)
if missing_gold.any():
    print(f"\n{missing_gold.sum()} disagreements need a gold label. Discuss them (data/disagreements.csv), "
          "write the agreed label into the 'gold' column of test_suite.csv, and run this script again.")
    sys.exit(0)

df.to_csv(DATA / "test_suite_final.csv", index=False)
print(f"\nAll items have gold labels -> data/test_suite_final.csv ({len(df)} rows)")
