"""Shared settings and helpers for the negation-sentiment experiment."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

RANDOM_SEED = 42

# The five perturbation types (+ the unmodified seed). Order is used in tables/figures.
PHENOMENA = ["negation", "double_negation", "contrast", "intensifier", "negated_intensifier"]
PHENOMENON_LABELS = {
    "negation": "Negation",
    "double_negation": "Double neg.",
    "contrast": "Contrast",
    "intensifier": "Intensifier",
    "negated_intensifier": "Neg. intens.",
}

LABELS = ("pos", "neg")

# Negation cues used to (a) exclude seeds that are already negated and
# (b) find naturally negated SST-2 sentences. SST-2 is tokenised, e.g. "does n't".
NEGATION_RE = re.compile(
    r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely|cannot)\b|n't\b",
    re.IGNORECASE,
)
CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet)\b", re.IGNORECASE)

# ---- Models -------------------------------------------------------------
ROBERTA_ID = "textattack/roberta-base-SST-2"          # RoBERTa-base fine-tuned on SST-2
LLM_ID = "Qwen/Qwen2.5-3B-Instruct"                   # ungated, no licence request needed
LLM_FALLBACK_ID = "meta-llama/Llama-3.2-3B-Instruct"  # gated: only if access is granted

# Three paraphrased zero-shot prompts (the poster reports mean and range across them).
LLM_PROMPTS = {
    "p1": "Classify the sentiment of the following movie review sentence as positive or negative.\n"
          "Sentence: {text}\nAnswer with one word: positive or negative.",
    "p2": "Is the opinion expressed in this sentence positive or negative?\n"
          "\"{text}\"\nReply with exactly one word (positive/negative).",
    "p3": "You are a sentiment classifier. Read the text and decide whether its overall sentiment "
          "is positive or negative.\nText: {text}\nSentiment:",
}
