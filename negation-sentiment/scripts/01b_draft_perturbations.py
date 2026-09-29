"""Step 1d (optional): fill data/test_suite.csv with RULE-BASED DRAFT perturbations.

Usage
  python scripts/01b_draft_perturbations.py                 # write drafts into data/test_suite.csv
  python scripts/01b_draft_perturbations.py --drop-incomplete   # also remove seeds the rules could not handle

Every generated sentence is a DRAFT: the `notes` column says DRAFT. You must read each one,
fix what sounds wrong, and label everything yourself (that is what makes the suite valid).
Rows the rules cannot handle are left empty for you to write by hand.

Method (state this in the poster/appendix): perturbations were generated with deterministic
rules over a hand-built adjective/antonym lexicon and then manually corrected and labelled
by both authors.
"""
import re
import sys
import pandas as pd
from common import DATA, PHENOMENA

# --- lexicon: adjective -> antonym of the opposite polarity -------------------
ANTONYM = {
    # positive -> negative
    "charming": "charmless", "good": "bad", "great": "awful", "funny": "unfunny",
    "entertaining": "boring", "enjoyable": "tedious", "engaging": "dull", "warm": "cold",
    "clever": "clumsy", "smart": "stupid", "witty": "witless", "beautiful": "ugly",
    "gorgeous": "ugly", "solid": "shaky", "compelling": "tiresome", "powerful": "feeble",
    "moving": "unmoving", "delightful": "dreary", "poignant": "hollow", "fresh": "stale",
    "thrilling": "dull", "lively": "lifeless", "satisfying": "frustrating", "memorable": "forgettable",
    "brilliant": "inept", "thoughtful": "mindless", "elegant": "clumsy", "tender": "harsh",
    "sweet": "sour", "remarkable": "unremarkable", "inventive": "derivative", "honest": "phony",
    "absorbing": "tedious", "impressive": "unimpressive", "amusing": "tiresome", "heartfelt": "hollow",
    "likable": "unlikable", "creative": "unimaginative", "exciting": "dull", "affecting": "unaffecting",
    "well-made": "shoddy", "well-crafted": "sloppy", "well-honed": "shapeless", "hip": "square",
    "stunning": "drab", "lyrical": "leaden", "satirical": "toothless", "deep": "shallow",
    "meaningful": "pointless", "dense": "thin", "jolly": "grim", "intriguing": "dreary",
    "competent": "incompetent", "earnest": "cynical", "quiet": "noisy", "unforgettable": "forgettable",
    "pure": "muddled", "elliptical": "obvious", "brooding": "breezy", "intense": "tepid",
    "magnificent": "wretched", "harmless": "harmful", "anarchic": "tame", "exquisite": "crude",
    "unsentimental": "mawkish", "light": "heavy", "healthy": "sickly", "provocative": "bland",
    "seductive": "repellent", "significant": "trivial", "accessible": "impenetrable",
    "uplifting": "depressing", "lovely": "hideous", "thought-provoking": "mindless",
    "surreal": "mundane", "economical": "bloated", "regal": "common", "subtle": "heavy-handed",
    "spontaneous": "laboured", "original": "derivative", "sharp": "blunt", "hilarious": "unfunny",
    # negative -> positive
    "bad": "good", "awful": "great", "dull": "engaging", "boring": "entertaining",
    "tedious": "enjoyable", "lifeless": "lively", "stupid": "smart", "forgettable": "memorable",
    "tiresome": "compelling", "clumsy": "elegant", "shallow": "profound", "predictable": "surprising",
    "slow": "brisk", "pointless": "meaningful", "unfunny": "funny", "banal": "original",
    "trite": "original", "cliched": "original", "flat": "vivid", "mediocre": "excellent",
    "painful": "pleasant", "annoying": "charming", "messy": "tidy", "weak": "strong",
    "lazy": "careful", "inept": "skilful", "unpleasant": "pleasant", "sloppy": "careful",
    "hollow": "heartfelt", "derivative": "inventive", "pretentious": "unpretentious",
    "incoherent": "coherent", "tepid": "spirited", "silly": "thoughtful", "crude": "refined",
    "maddening": "satisfying", "meaningless": "meaningful", "unwatchable": "watchable",
    "bleak": "sunny", "desperate": "assured", "shapeless": "well-honed", "wacky": "restrained",
    "forced": "effortless", "heavy-handed": "subtle", "inoffensive": "daring", "insipid": "flavourful",
    "implausible": "believable", "vulgar": "tasteful", "coarse": "refined", "inconsistent": "consistent",
    "excessive": "restrained", "profane": "gentle", "ridiculous": "convincing", "shoddy": "well-made",
    "overlong": "brisk", "whiny": "good-natured", "unfocused": "focused", "underdeveloped": "fully realised",
    "laughable": "convincing", "arthritic": "nimble", "indifferent": "passionate", "valueless": "valuable",
    "grim": "cheerful", "drab": "stunning", "leaden": "lyrical", "thin": "dense",
    "sickly": "healthy", "mawkish": "unsentimental", "repellent": "seductive", "trivial": "significant",
    "impenetrable": "accessible", "depressing": "uplifting", "hideous": "lovely", "mindless": "thoughtful",
    "mundane": "surreal", "bloated": "economical", "muddled": "pure", "blunt": "sharp",
    "harsh": "tender", "sour": "sweet", "stale": "fresh", "ugly": "beautiful", "feeble": "powerful",
    "witless": "witty", "charmless": "charming", "frustrating": "satisfying", "unremarkable": "remarkable",
    "phony": "honest", "unimpressive": "impressive", "unlikable": "likable", "unimaginative": "creative",
    "dreary": "delightful", "shaky": "solid", "cold": "warm", "unmoving": "moving",
}
ADJECTIVES = set(ANTONYM)
INTENSIFIERS = {"very", "really", "so", "too", "extremely", "utterly", "quite", "truly", "rather"}
COPULA = {"is", "are", "was", "were", "'s", "'re", "feels", "seems", "looks", "remains", "becomes"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "one", "another", "his", "her", "its", "their"}
CONTRAST_CLAUSES = {   # clause of the OPPOSITE polarity, placed before "but"
    "pos": ["the premise is thin", "the pacing drags", "the plot makes little sense",
            "the dialogue is clumsy", "the ending falls flat"],
    "neg": ["the cast is likable", "the photography is lovely", "the score is lovely",
            "the idea is promising", "the opening is strong"],
}


def tokens(s):
    return s.split()


def first_adjective(toks):
    """index of the first known adjective that is not already intensified"""
    for i, t in enumerate(toks):
        w = t.strip(",.!?;:").lower()
        if w in ADJECTIVES and (i == 0 or toks[i - 1].lower() not in INTENSIFIERS):
            return i, w
    return None, None


def add_not(text):
    """negate the sentence: after a copula, or in front of a noun/adjective phrase"""
    toks = tokens(text)
    for i, t in enumerate(toks):
        if t.strip(",.").lower() in COPULA:
            return " ".join(toks[:i + 1] + ["not"] + toks[i + 1:])
    first = toks[0].strip(",.").lower()
    if first in DETERMINERS or first in ADJECTIVES or first in INTENSIFIERS:
        return "not " + text
    return None


def make_drafts(text, label):
    """returns dict phenomenon -> draft sentence (or None when the rules do not apply)"""
    toks = tokens(text)
    idx, adj = first_adjective(toks)
    out = {p: None for p in PHENOMENA}

    out["negation"] = add_not(text)

    if idx is not None:
        strong = toks[idx].strip(",.!?;:")
        intens = toks[:idx] + ["very", strong] + toks[idx + 1:]
        out["intensifier"] = " ".join(intens)
        out["negated_intensifier"] = add_not(" ".join(intens))
        # "double negation": negate the antonym, which preserves the seed's polarity
        swapped = " ".join(toks[:idx] + [ANTONYM[adj]] + toks[idx + 1:])
        out["double_negation"] = add_not(swapped)

    clauses = CONTRAST_CLAUSES["pos" if label == "pos" else "neg"]
    clause = clauses[hash(text) % len(clauses)]
    out["contrast"] = f"{clause} , but {text}"
    return out


def main():
    path = DATA / "test_suite.csv"
    df = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding_errors="replace").fillna("")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    seeds = df[df["phenomenon"] == "seed"].set_index("seed_id")

    # seed polarity: from seed_candidates.csv (the SST-2 label)
    cand = pd.read_csv(DATA / "seed_candidates.csv", dtype=str, sep=None, engine="python",
                       encoding_errors="replace").fillna("")
    cand.columns = [c.strip().lstrip("﻿") for c in cand.columns]
    polarity = dict(zip(cand["seed_id"], cand["label"]))

    filled = blank = 0
    for i, row in df.iterrows():
        if row["phenomenon"] == "seed" or row["text"].strip():
            continue
        sid = row["seed_id"]
        drafts = make_drafts(seeds.loc[sid, "text"], polarity.get(sid, "pos"))
        d = drafts.get(row["phenomenon"])
        if d:
            df.at[i, "text"] = d
            df.at[i, "notes"] = "DRAFT - check wording and label"
            filled += 1
        else:
            df.at[i, "notes"] = "write by hand"
            blank += 1

    if "--drop-incomplete" in sys.argv:
        bad = df[(df["phenomenon"] != "seed") & (df["text"].str.strip() == "")]["seed_id"].unique()
        df = df[~df["seed_id"].isin(bad)]
        print(f"dropped {len(bad)} seeds the rules could not fully handle")

    df.to_csv(path, index=False)
    n_seeds = df[df['phenomenon'] == 'seed'].shape[0]
    print(f"drafted {filled} sentences, {blank} left empty; suite now has {n_seeds} seeds / {len(df)} rows")
    print("NEXT: read every DRAFT row, fix the wording, then both of you label all rows "
          "(label_A / label_B) independently.")


if __name__ == "__main__":
    main()
