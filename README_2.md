# Not Bad, Not Good? Sentiment Models under Negation

Code, data and results for our research poster in *Trends in Natural Language Processing*
(3. Parallelgruppe, SoSe 2026), Universität Trier.

**Priyadith Hosahalli Nagesh · Vismaya Basavaraju Mamatha**

> **Everything lives in the [`negation-sentiment/`](negation-sentiment) folder.**
> To reproduce: `git clone` this repository, then `cd negation-sentiment/negation-sentiment`.

## What this is

We test whether sentiment models actually handle negation, or whether they lean on polar
words. We built a **minimal-pair test suite of 263 items from 44 SST-2 seed sentences**:
each seed appears unmodified and under five perturbations (negation, double negation,
contrast, intensifier, negated intensifier), so every item differs from its seed by one
operator only. Three model families are compared on the identical items, plus 149
naturally negated SST-2 sentences.

## Headline result

| Model | Seed acc. | Perturbed acc. | Drop |
|---|---|---|---|
| VADER (lexicon + rules) | 70.5 % | 63.0 % | −7.4 (n.s.) |
| RoBERTa-base (fine-tuned on SST-2) | **100 %** | 78.5 % | **−21.5** |
| Qwen2.5-3B-Instruct (zero-shot, 3 prompts) | 95.5 % | 68.3 % | **−27.1** |

RoBERTa classifies **every** unmodified seed correctly and still loses 21.5 points under
perturbation; **74 % of its errors repeat the label it gave the unmodified seed**. High
benchmark accuracy does not imply compositional understanding. Negated intensifiers are
near chance for all three models (44–50 % error).

Full numbers, confidence intervals and McNemar tests:
[`negation-sentiment/results/summary.md`](negation-sentiment/results/summary.md).

## Where things are

| Path | Contents |
|---|---|
| `negation-sentiment/scripts/` | data preparation, perturbation generator, model inference, statistics |
| `negation-sentiment/data/` | the final test suite, the natural-negation subset, the agreement sample, the verification record |
| `negation-sentiment/results/` | raw predictions, summary tables, poster figures |
| `negation-sentiment/notebooks/` | Colab notebook that runs the whole experiment |
| `negation-sentiment/poster/` | LaTeX source of the poster and appendix |

## Reproducing

```bash
cd negation-sentiment
pip install -r requirements.txt
python scripts/03_run_models.py   # needs a GPU for the LLM
python scripts/04_analyze.py
```

Or open `notebooks/run_on_colab.ipynb` in Google Colab with a T4 runtime.

Qwen2.5-3B-Instruct is ungated, so no Hugging Face token is required. Random seed fixed at
42. Seeds and natural sentences come from the SST-2 **validation** split only, because the
RoBERTa checkpoint was fine-tuned on the training split.

## Annotation

The 220 perturbations were drafted by a rule-based script over a hand-built antonym
lexicon, then checked row by row by both authors. 34 of the 264 drafted rows were flagged:
33 were reworded and kept, and the 34th was excluded because the antonym substitution had
reversed the polarity instead of preserving it — hence 263 items, not 264. Three further
sentences in the same seed group were shortened to match a rewording, so 36 sentences
differ from the draft in all. Every row is documented in
[`data/verification_record.csv`](negation-sentiment/data/verification_record.csv).

On an 80-item random sample of the drafts, labelled independently and blind by both authors
before the verification pass, **Cohen's κ = 1.00** (80/80). On four of those items both
authors departed from the label the construction rule implied; all four were among the 34
the verification pass then flagged.

The error breakdown on the poster is computed automatically (see
[the README](negation-sentiment/README.md#method-notes-use-these-in-the-poster-and-appendix));
the manual error-coding step that was originally planned was not carried out.
