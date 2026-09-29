# Natural-negation subset: real results (run 26.09.2026, Colab T4)

149 SST-2 validation sentences that already contain a negation cue (75 neg / 74 pos).
Models: VADER; textattack/roberta-base-SST-2; Qwen/Qwen2.5-3B-Instruct (3 zero-shot prompts).

| Model | Accuracy | 95% CI | acc on neg | acc on pos |
|---|---|---|---|---|
| VADER | 59.1 | [51.0, 66.4] | 42.7 | 75.7 |
| RoBERTa (fine-tuned) | 94.0 | [89.9, 97.3] | 97.3 | 90.5 |
| Qwen prompt 1 | 90.6 | [85.9, 95.3] | 96.0 | 85.1 |
| Qwen prompt 2 | 92.6 | [88.6, 96.6] | 96.0 | 89.2 |
| Qwen prompt 3 | 79.2 | [72.5, 85.2] | 100.0 | 58.1 |
| Qwen majority vote | 90.6 | | | |

Qwen mean over prompts: 87.5 (range 79.2-92.6). The three prompts agree on 81.2% of items.

Notes for the poster:
- VADER is below chance on negated NEGATIVE sentences (42.7%): it follows the polar word and
  misses the operator, exactly the failure mode H1 predicts.
- Prompt 3 pushes Qwen towards "negative" (100% on neg, 58% on pos): zero-shot LLM results are
  prompt-dependent, which is why we report mean and range over three prompts.
- RoBERTa is strongest here but was fine-tuned on SST-2 train, i.e. the same domain and style.
- Bootstrap CI endpoints are Monte-Carlo estimates (1,000 resamples); they move by a few
  tenths of a point between runs. The authoritative values are the ones in
  `results/results.json`, written by `04_analyze.py`.
