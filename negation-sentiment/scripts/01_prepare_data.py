"""Step 1: select seed candidates and the natural-negation subset from SST-2.

Usage
  python scripts/01_prepare_data.py candidates   # writes data/seed_candidates.csv, data/natural_negation.csv
  (you now mark keep=1 for 60 positive + 60 negative seeds in seed_candidates.csv)
  python scripts/01_prepare_data.py skeleton     # writes data/test_suite.csv to fill in by hand

Only the SST-2 *validation* split is used, because the RoBERTa model was fine-tuned
on the training split (using training sentences would leak).
"""
import sys
import pandas as pd
from common import DATA, RANDOM_SEED, PHENOMENA, NEGATION_RE, CONTRAST_RE


def load_sst2_validation() -> pd.DataFrame:
    from datasets import load_dataset
    try:
        ds = load_dataset("stanfordnlp/sst2", split="validation")
    except Exception:
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
    df = ds.to_pandas()[["idx", "sentence", "label"]]
    df["label"] = df["label"].map({1: "pos", 0: "neg"})
    df["sentence"] = df["sentence"].str.strip()
    df["n_tokens"] = df["sentence"].str.split().str.len()
    return df


def candidates():
    df = load_sst2_validation()
    has_neg = df["sentence"].str.contains(NEGATION_RE)
    has_contrast = df["sentence"].str.contains(CONTRAST_RE)

    # Seed candidates: short, no negation, no contrast marker. 100 per class so you can
    # discard unclear ones and still keep 60 + 60.
    pool = df[~has_neg & ~has_contrast & df["n_tokens"].between(4, 12)]
    seeds = (pool.groupby("label", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), 100), random_state=RANDOM_SEED)))
    seeds = seeds.rename(columns={"idx": "sst2_idx", "sentence": "text"})
    seeds.insert(0, "keep", "")
    seeds.insert(1, "seed_id", [f"S{i:03d}" for i in range(len(seeds))])
    seeds.to_csv(DATA / "seed_candidates.csv", index=False)

    # Natural subset: sentences that already contain negation (gold = SST-2 label).
    nat = df[has_neg]
    per_class = 75
    nat = (nat.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_class), random_state=RANDOM_SEED)))
    nat = nat.rename(columns={"idx": "sst2_idx", "sentence": "text", "label": "gold"})
    nat.insert(0, "item_id", [f"N{i:03d}" for i in range(len(nat))])
    nat["phenomenon"] = "natural"
    nat[["item_id", "sst2_idx", "phenomenon", "text", "gold"]].to_csv(DATA / "natural_negation.csv", index=False)

    print(f"seed candidates: {len(seeds)} ({seeds['label'].value_counts().to_dict()}) -> data/seed_candidates.csv")
    print(f"natural negation: {len(nat)} ({nat['gold'].value_counts().to_dict()}) -> data/natural_negation.csv")
    print("Next: open seed_candidates.csv, put 1 in 'keep' for 60 clear positive and 60 clear negative seeds.")


def skeleton():
    # sep=None lets pandas detect ',' or ';' (German Excel saves CSV with ';')
    seeds = pd.read_csv(DATA / "seed_candidates.csv", dtype=str, sep=None, engine="python",
                        encoding_errors="replace")
    seeds.columns = [c.strip().lstrip("\ufeff") for c in seeds.columns]
    if "keep" not in seeds.columns:
        sys.exit(f"No 'keep' column found. Columns are: {list(seeds.columns)}")
    mark = seeds["keep"].fillna("").str.strip().str.lower()
    kept = seeds[mark.isin(["1", "1.0", "x", "yes", "y", "true"])]
    counts = kept["label"].value_counts().to_dict()
    print(f"kept seeds: {len(kept)} {counts}")
    if len(kept) == 0:
        filled = mark[mark != ""]
        sys.exit("No seeds marked keep=1. "
                 + (f"Values found in 'keep': {filled.unique()[:10].tolist()}" if len(filled)
                    else "The 'keep' column is empty - did you upload the edited file and save it as CSV?"))
    if len(kept) != 120 or counts.get("pos", 0) != counts.get("neg", 0):
        print("WARNING: aim for exactly 60 pos + 60 neg seeds. Continuing anyway.")
    rows = []
    for _, s in kept.iterrows():
        rows.append(dict(item_id=f"{s.seed_id}-seed", seed_id=s.seed_id, phenomenon="seed",
                         text=s.text, label_A="", label_B="", gold="", notes=""))
        for p in PHENOMENA:
            rows.append(dict(item_id=f"{s.seed_id}-{p}", seed_id=s.seed_id, phenomenon=p,
                             text="", label_A="", label_B="", gold="", notes=""))
    out = DATA / "test_suite.csv"
    if out.exists():
        sys.exit(f"{out} already exists - not overwriting your work.")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")
    print("Next: write the perturbed sentences (see ANNOTATION_GUIDELINES.md), then each of you "
          "labels every row independently in label_A / label_B.")


if __name__ == "__main__":
    {"candidates": candidates, "skeleton": skeleton}[sys.argv[1] if len(sys.argv) > 1 else "candidates"]()
