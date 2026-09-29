# Not Bad, Not Good? Sentiment Models under Negation

Code and data for our poster in *Trends in Natural Language Processing* (SoSe 2026, Universität Trier).
Authors: Priyadith Hosahalli Nagesh, Vismaya Basavaraju Mamatha.

We test whether sentiment models handle negation, double negation, contrast and intensifiers.
We compare three model families on the same minimal sentence pairs, plus a set of naturally negated SST-2 sentences.

| Model | Family | Checkpoint |
|---|---|---|
| VADER | lexicon + rules | `vaderSentiment` |
| RoBERTa-base | fine-tuned on SST-2 | `textattack/roberta-base-SST-2` |
| Qwen2.5-3B-Instruct | zero-shot LLM, 3 prompts | `Qwen/Qwen2.5-3B-Instruct` |

## The test suite

`data/test_suite_final.csv` — **263 items from 44 SST-2 seed sentences** (22 positive, 22 negative).
Each seed has five perturbations, so every item is a minimal pair of its seed:

| Phenomenon | n | Example (from the negative seed *a sometimes tedious film .*) | Gold |
|---|---|---|---|
| seed | 44 | a sometimes tedious film . | neg |
| negation | 44 | **not** a sometimes tedious film . | pos |
| double_negation | 43 | **not** a sometimes **enjoyable** film . | neg |
| contrast | 44 | the idea is promising , **but** a sometimes tedious film . | neg |
| intensifier | 44 | a sometimes **very** tedious film . | neg |
| negated_intensifier | 44 | **not** a sometimes **very** tedious film . | pos |

Plus `data/natural_negation.csv` — **149 SST-2 validation sentences that already contain
negation** (75 neg, 74 pos), to check whether the template findings transfer to real text.

### How the suite was built and validated

1. Seed sentences drawn from the SST-2 **validation** split only (RoBERTa was fine-tuned on
   the training split, so training sentences would leak).
2. The 220 perturbations drafted by a rule-based script over a hand-built antonym lexicon
   (`scripts/01b_draft_perturbations.py`). Seeds the rules could not cover were dropped,
   leaving 44. *Caveat:* the choice of contrast clause uses Python's `hash()`, which is
   randomised per interpreter run, so re-running the generator can pick a different clause.
   The sentences actually used are fixed in `data/test_suite_final.csv`.
3. **Both authors checked all 264 drafted rows.** 34 were flagged (12.9%): 33 sentences
   reworded, 1 gold label corrected. Three further sentences in the S073 group were then
   shortened to match a rewording, so 36 sentences differ from the draft in all. Every row —
   draft text, reviewer verdict, reviewer's replacement, final text — is in
   `data/verification_record.csv`.
4. **One item excluded** — `S049-double_negation` (*as tasteful as it is not banal .*), where
   the antonym substitution reversed the polarity instead of preserving it. This is the item
   whose gold label the reviewers corrected: rather than relabel it, it was dropped, because a
   double-negation item that flips polarity breaks the minimal-pair design. Hence 43 rather
   than 44 double-negation items, and 263 rather than 264 items overall.
5. **Agreement:** a random 80-item sample was labelled independently and blind by both
   authors before any discussion: **Cohen's κ = 1.00, raw agreement 80/80**
   (`data/agreement_sample_labelled.csv`, `results/agreement.json`). On 4 of those 80 items
   both authors independently assigned a label *different* from the one the construction rule
   predicted; the rule-derived labels were not shown in the sample files.

## Pipeline

The suite is final, so only steps 3–5 remain. Open `notebooks/run_on_colab.ipynb`
(it clones this repository) or run from the repository root:

| Step | Command | Who | Output |
|---|---|---|---|
| 3 | `python scripts/03_run_models.py` (needs a GPU for the LLM) | script | `results/predictions.csv` |
| 4 | `python scripts/04_analyze.py` | script | `results/summary.md`, poster figures, `errors_to_code.csv` |

Steps 1–2 built the suite. They are kept in `scripts/` to document how it was built, but they
**cannot be re-run from this repository**: their working files (`data/seed_candidates.csv`,
`data/test_suite.csv`) are intermediate and are not committed — only the finished suite and the
verification record are. See [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md) for the rules
that were used.

**Not carried out:** step 5 (`scripts/05_error_taxonomy.py`) was planned as a manual
error coding by both authors. It was **not** performed — `results/errors_to_code.csv`
is exported but empty of codings. The error breakdown reported on the poster comes from
the automatic classification described below instead. The script is kept in `scripts/`
for anyone who wants to do the manual coding.

## Setup

```bash
pip install -r requirements.txt
```

**Model access:** Qwen2.5-3B-Instruct is ungated, so no Hugging Face token is needed. Llama 3.2 was considered first but is gated behind a licence request, so the ungated Qwen model was used instead.

**Hardware:** VADER and RoBERTa run on a CPU in minutes. The LLM needs a GPU; a free Colab T4 takes about 10 minutes for 3 prompts × 412 sentences (263 suite + 149 natural).

## Method notes (use these in the poster and appendix)

- **Data:** seeds and natural sentences come from the SST-2 *validation* split only. RoBERTa was fine-tuned on the training split, so training sentences would leak.
- **VADER:** the compound score ≥ 0 counts as positive. The usual ±0.05 neutral band is dropped because the task is binary.
- **LLM:** zero-shot, one of 3 paraphrased prompts (see `scripts/common.py`). The label is whichever answer, *positive* or *negative*, gets the higher probability as the first answer token. This avoids parsing free text, and no answer can be invalid. Descriptive numbers are the mean over the prompts; significance tests use the majority vote.
- **Statistics:** 95% bootstrap confidence intervals (1,000 resamples, clustered by seed). Exact McNemar tests compare each seed with its perturbation, per model and phenomenon, with Bonferroni correction over 15 tests.
- **Agreement:** Cohen's kappa is computed on the blind labels, before any discussion.
- **Perturbations:** generated by rules over a hand-built adjective/antonym lexicon (`scripts/01b_draft_perturbations.py`), then checked and corrected by both authors; seeds the rules could not cover were dropped. "Double negation" is realised as a negated antonym (*charming* -> *not charmless*), which keeps the seed's polarity. "Contrast" prepends a clause of the opposite polarity followed by *but*, leaving the seed sentence itself untouched; the clause after *but* decides the gold label, so contrast **preserves** the seed's label. Only `negation` and `negated_intensifier` flip it.
- **Known limitation of the suite:** in one seed (`S085`) the seed sentence already contained *very*, so its intensifier item adds a second one on a different adjective. The minimal-pair property still holds.
- **Error taxonomy (poster Table 3):** computed automatically, not hand-coded. Every wrong prediction on a perturbed item falls into exactly one of three classes: *operator ignored* (the model repeats the label it gave the unmodified seed, although the operator flips the gold label), *already wrong on the seed* (the model also got the unperturbed sentence wrong), or *over-applied / other*. See `results/error_types_auto.csv` and `results/error_types_auto.json`.

## Reproducibility

The random seed is fixed (42) in `scripts/common.py`. The exact model checkpoints are named in `scripts/common.py` and in the table at the top of this file.

**Provenance of `results/predictions.csv`:** the models were run on Colab with the code of
`scripts/03_run_models.py` executed inline in the notebook rather than by invoking the script,
so `results/run_info.txt` was not written. The `pred` column is the model output for all 412
items. The `score` column is recorded for the 263 suite items only; it is empty for the 149
natural-negation items, whose raw scores were not retained. No result in `results/` depends on
`score` — `04_analyze.py` reads only `pred`. All hand-checked data — the final suite (`data/test_suite_final.csv`), the row-by-row verification record (`data/verification_record.csv`) and the blind agreement sample (`data/agreement_sample_labelled.csv`) — are committed to the repository, along with the raw model predictions (`results/predictions.csv`).
