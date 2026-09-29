"""Step 3: run all models on the test suite and the natural-negation subset.

Usage (Colab with a T4 GPU recommended):
  python scripts/03_run_models.py                     # all three models
  python scripts/03_run_models.py --models vader roberta
  python scripts/03_run_models.py --llm Qwen/Qwen2.5-3B-Instruct   # if Llama access is not granted

Writes results/predictions.csv in long format:
  item_id, source (suite|natural), model, prompt_id, pred, score
"""
import argparse
import pandas as pd
from common import DATA, RESULTS, ROBERTA_ID, LLM_ID, LLM_PROMPTS


def load_items() -> pd.DataFrame:
    suite = pd.read_csv(DATA / "test_suite_final.csv", dtype=str)
    suite["source"] = "suite"
    parts = [suite[["item_id", "source", "text"]]]
    nat_path = DATA / "natural_negation.csv"
    if nat_path.exists():
        nat = pd.read_csv(nat_path, dtype=str)
        nat["source"] = "natural"
        parts.append(nat[["item_id", "source", "text"]])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- VADER
def run_vader(items):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t)["compound"] for t in items["text"]]
    # Binary decision: compound >= 0 -> pos. (VADER's usual +-0.05 neutral band is
    # collapsed because the task is binary; state this in the poster/appendix.)
    return pd.DataFrame({"item_id": items["item_id"], "source": items["source"], "model": "VADER",
                         "prompt_id": "-", "pred": ["pos" if s >= 0 else "neg" for s in scores],
                         "score": scores})


# ---------------------------------------------------------------- RoBERTa
def run_roberta(items, batch_size=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ROBERTA_ID)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_ID).to(dev).eval()
    probs = []
    texts = items["text"].tolist()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, return_tensors="pt").to(dev)
            probs += torch.softmax(model(**enc).logits, -1)[:, 1].tolist()   # P(label 1 = positive)
    return pd.DataFrame({"item_id": items["item_id"], "source": items["source"], "model": "RoBERTa",
                         "prompt_id": "-", "pred": ["pos" if p >= 0.5 else "neg" for p in probs],
                         "score": probs})


# ---------------------------------------------------------------- LLM
def run_llm(items, model_id, batch_size=16):
    """Zero-shot classification by comparing the log-probability of the answers
    'positive' vs 'negative' as the first generated token (no free-text parsing needed)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    def first_ids(words):
        ids = {tok.encode(w, add_special_tokens=False)[0] for w in words}
        return sorted(ids)
    pos_ids = first_ids(["positive", "Positive", " positive", " Positive"])
    neg_ids = first_ids(["negative", "Negative", " negative", " Negative"])
    assert not set(pos_ids) & set(neg_ids), "answer tokens overlap - check tokenizer"

    out = []
    texts = items["text"].tolist()
    for pid, template in LLM_PROMPTS.items():
        prompts = [tok.apply_chat_template([{"role": "user", "content": template.format(text=t)}],
                                           tokenize=False, add_generation_prompt=True) for t in texts]
        margins = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                logp = torch.log_softmax(model(**enc).logits[:, -1, :].float(), -1)
                lp_pos = torch.logsumexp(logp[:, pos_ids], -1)
                lp_neg = torch.logsumexp(logp[:, neg_ids], -1)
                margins += (lp_pos - lp_neg).tolist()
        out.append(pd.DataFrame({"item_id": items["item_id"], "source": items["source"], "model": "LLM",
                                 "prompt_id": pid, "pred": ["pos" if m >= 0 else "neg" for m in margins],
                                 "score": margins}))
        print(f"  LLM prompt {pid} done")
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vader", "roberta", "llm"])
    ap.add_argument("--llm", default=LLM_ID)
    args = ap.parse_args()

    items = load_items()
    print(f"{len(items)} items ({items['source'].value_counts().to_dict()})")
    frames = []
    if "vader" in args.models:
        frames.append(run_vader(items)); print("VADER done")
    if "roberta" in args.models:
        frames.append(run_roberta(items)); print("RoBERTa done")
    if "llm" in args.models:
        frames.append(run_llm(items, args.llm)); print(f"LLM ({args.llm}) done")

    new = pd.concat(frames, ignore_index=True)
    path = RESULTS / "predictions.csv"
    if path.exists():   # keep predictions of models not re-run this time
        old = pd.read_csv(path, dtype={"item_id": str})
        new = pd.concat([old[~old["model"].isin(new["model"].unique())], new], ignore_index=True)
    new.to_csv(path, index=False)
    (RESULTS / "run_info.txt").write_text(f"models={args.models}\nllm={args.llm}\nroberta={ROBERTA_ID}\n")

    # Sanity check: accuracy on the unmodified seeds should be high. If RoBERTa is near 0%
    # its label mapping is flipped.
    suite = pd.read_csv(DATA / "test_suite_final.csv", dtype=str)
    seeds = suite[suite["phenomenon"] == "seed"][["item_id", "gold"]]
    chk = new.merge(seeds, on="item_id")
    print("\nSeed accuracy (sanity check):")
    print(chk.assign(ok=chk["pred"] == chk["gold"]).groupby(["model", "prompt_id"])["ok"].mean().round(3))
