# Plan to the deadline (30.09.2026, end of day CET)

> **This is the original working plan, kept for the record. What actually happened differs
> in two ways:** the suite ended up at **44 seeds / 263 items** (not 60+60 seeds / 720 rows),
> because seeds the perturbation rules could not cover were dropped; and **step 5, the manual
> error coding, was not carried out** — the poster's error breakdown is computed automatically
> (see the README). The finished state of the project is described in
> [README.md](README.md).

| Date | Task | Who | Done when |
|---|---|---|---|
| Mon 21.09 | Request Llama 3.2 access on Hugging Face. Create the GitHub repo and push this folder | both | access requested, repo exists |
| Mon 21.09 | Step 1a: run `01_prepare_data.py candidates` | P | CSVs exist |
| Tue 22.09 | Step 1b: choose 60 pos + 60 neg seeds; step 1c: skeleton | both | `test_suite.csv` has 720 rows |
| Tue 22 – Wed 23.09 | Write perturbations (60 seeds each), then check each other's | P: S000–S0xx, V: rest | no empty `text` |
| Thu 24.09 | Blind labelling, `02_agreement.py`, resolve disagreements | both | `test_suite_final.csv` exists, kappa noted |
| Thu 24.09 | Step 3: run the models on Colab | P | `predictions.csv`; seed accuracy looks sane |
| Fri 25.09 | Step 4: analysis; step 5: code the errors independently | both | `summary.md`, `taxonomy.json` |
| Sat 26.09 | Put the real numbers and figures into the poster; rewrite claims to match the results | both | no placeholder values left |
| Sun 27.09 | Appendix PDF: full reference list plus **signed** integrity declarations (both) | both | appendix PDF |
| Mon 28.09 | Final checks (below), QR code, make the repo public | both | checklist ticked |
| Tue 29.09 | Upload the ZIP to STUD.IP (`posters` folder, your seminar group) | one of you | uploaded, 1 day of buffer left |

## If the results differ from the sample poster

That is expected: the sample numbers are invented. Rewrite the hypothesis verdicts (supported, partly, not supported) to match the real data. A refuted hypothesis explained well is still a good result.

## Final checklist before upload

- [ ] Watermark block removed from `poster.tex`; every number comes from `results/summary.md`
- [ ] University logo and a working QR code link to the public repo; URL in the footer correct
- [ ] Poster PDF page size is A1 (594 × 841 mm): `pdfinfo poster.pdf`
- [ ] All text at least 24 pt (the template is; re-check anything you add)
- [ ] Appendix PDF: complete references (mainly academic papers) and both signed declarations
- [ ] ZIP contains **exactly two PDFs** and is named `{id1}_{id2}.zip`
- [ ] AI use disclosed as the integrity declaration requires
