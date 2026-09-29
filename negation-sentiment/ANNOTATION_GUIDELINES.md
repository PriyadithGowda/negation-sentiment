# Annotation guidelines

These rules keep the test suite consistent. Follow them strictly and cite them in the appendix.

## 1. Choosing seeds (`data/seed_candidates.csv`)

Put `1` in the `keep` column for **60 positive and 60 negative** sentences (in the end 44 seeds survived the perturbation rules). Keep a sentence only if:

- its sentiment is **clear** without context (skip neutral, mixed or sarcastic sentences)
- it has a **polar word you can negate**, usually an adjective or verb (*charming*, *fails*, *boring*)
- the original SST-2 label is correct in your judgement

If a seed later turns out impossible to perturb naturally, replace it with another candidate.

## 2. Writing the five perturbations (`data/test_suite.csv`)

Change **as little as possible**. Keep the seed's words, domain and style (SST-2 is lower-case and tokenised; you may write normal text, just be consistent).

| Phenomenon | Rule | Seed → perturbation | Gold |
|---|---|---|---|
| `negation` | Negate the main polar predicate with *not / n't / never* | *a charming film* → *not a charming film* | flips |
| `double_negation` | Negate an antonym formed with a negative prefix or word: *not un-X, not without X, not in-X, never fails to* | *a charming film* → *a film not without charm* | same as seed |
| `contrast` | Prepend a clause of **opposite** polarity, then *but* + seed. The clause after *but* decides the overall sentiment | *a charming film* → *the plot is thin, but it is a charming film* | same as seed |
| `intensifier` | Add *very / really / extremely / truly* to the polar word | *a charming film* → *a very charming film* | same as seed |
| `negated_intensifier` | *not very / not really / not that* + polar word | *a charming film* → *not a very charming film* | flips (weakened) |

Checks for every sentence:

- grammatical and natural enough that a native speaker could write it
- no new polar words except the contrast clause
- the whole sentence has a clear gold label under the rules above; if not, rewrite it

Split the writing: each of you writes the perturbations for half of the seeds. Then swap and check each other's sentences before labelling.

## 2b. Checking the drafts

`scripts/01b_draft_perturbations.py` fills most rows with rule-based drafts (marked DRAFT in `notes`). Go through every row and:

- fix anything ungrammatical or unnatural (the rules are simple and sometimes clumsy)
- make sure only the intended operator changed
- rewrite the sentence if the draft changed the meaning in an unintended way
- delete the DRAFT note once you have checked the row

Split the seeds between you, then swap and check each other's corrections. Note in the appendix how many drafts you had to correct: that number is evidence of careful work.

## 3. Labelling (blind) — CARRIED OUT ON A SAMPLE, NOT EVERY ROW

> **What actually happened.** The plan below was to double-label all 264 rows and compute
> kappa over the lot. Instead, both authors blind-labelled a **random 80-item sample** of the
> drafts (that is the kappa on the poster: **1.00**, 80/80), and all 264 rows went through a
> single-pass verification instead of a second independent labelling — each row marked ok/not-ok
> with a replacement sentence where needed (`data/verification_record.csv`). Because of this,
> `scripts/02_agreement.py` below computes a different quantity from the reported kappa and
> cannot be re-run from this repository: it needs `data/test_suite.csv` with `label_A`/`label_B`
> columns, which is an intermediate file and is not committed. The instructions below are kept
> for reference.

Each of you labels **every row** (`pos` or `neg`) **without seeing the other's labels**:

1. Make two copies of `test_suite.csv`, one per person.
2. Label your copy in your own column (`label_A` or `label_B`). Judge the sentence as a reader would, not by the rule table.
3. Paste both columns into the shared `test_suite.csv` and run `python scripts/02_agreement.py`.
4. Discuss each disagreement in `data/disagreements.csv` and write the agreed label into `gold`.

Report Cohen's kappa **before** discussion. That is the number on the poster.

## 4. Error taxonomy (`results/errors_to_code.csv`) — NOT CARRIED OUT

> This manual coding step was planned but **not performed**. `errors_to_code.csv` is
> exported and left uncoded. The error breakdown on the poster is computed automatically
> instead (see README, "Error taxonomy"). The instructions below are kept for reference.

For each model error, choose one category. Code independently first (`type_A`, `type_B`), then agree on `type_final` for disagreements.

| Code | Name | Definition | Example |
|---|---|---|---|
| `polar_word` | Polar word wins | The prediction follows the sentiment of a strongly polar word and ignores the operator acting on it | *not a charming film* → pos |
| `scope` | Scope error | The operator is noticed but applied to the wrong part of the sentence, or two operators are not combined (double negation) | *not without charm* → neg |
| `clause` | Wrong clause | In contrast sentences, the prediction follows the clause before *but* | *the plot is thin, but charming* → neg |
| `other` | Other | Anything else (tokenisation, label noise, unclear) | |

If there are more than about 80 errors per model, code a random sample: `python scripts/05_error_taxonomy.py --sample 60`.
