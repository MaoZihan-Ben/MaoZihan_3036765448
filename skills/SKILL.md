---
name: card-krueger-did-replication
description: Rebuild every table and figure of the Card and Krueger (1994) minimum-wage difference-in-differences replication from the raw store-level survey data, and re-check each reproduced value against the value printed in the paper. Use when the tables in outputs/ need to be regenerated, or when the replication has to be verified against the published results.
---

# Card and Krueger (1994) minimum-wage replication

## What this skill does

Rebuilds the published tables and the wage-distribution figure of Card, D., and A. B. Krueger
(1994), "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and
Pennsylvania", American Economic Review 84(4), 772-793, from the raw store-level survey file, and
writes a value-by-value comparison against the numbers printed in the paper. One command produces
all of it; no step has to be run by hand and no file has to be edited.

## Inputs

- `data/raw/public.csv` - the raw store-level survey file, 410 stores and 46 columns. This is the
  only input the workflow needs.
- The published benchmarks are stored inside `code/run_all.py` itself, so no external copy of the
  paper is required to run the comparison.

## Outputs

- `data/processed/cleaned_did.csv` - the cleaned analysis data set used by every regression.
- `outputs/tables/table2_means.csv`, `table3_did.csv`, `table4_employment_regressions.csv`,
  `table5_robustness.csv`, `table6_other_outcomes.csv`, `table7_prices.csv` - the replicated
  tables, matching the paper's tables of the same numbers.
- `outputs/tables/comparison_with_paper.csv` - every reproduced value beside its published value,
  with the difference and a pass or flag marker.
- `outputs/tables/extra_permutation_test.csv`, `extra_permutation_draws.csv` - the randomization
  inference extension.
- `outputs/figures/figure1_wage_distribution.png`, `extra_permutation_test.png` - the figures.

## Procedure

1. Confirm that `data/raw/public.csv` exists.
2. From the repository root, run:

```
python code/run_all.py
```

3. The script resolves its own directories from its location, so the working directory does not
   have to be set; running from the root keeps the printed paths readable.
4. The script writes only to `data/processed/` and to `outputs/`. It never writes to the raw data
   file, so the run can be repeated as often as needed.
5. Read the console summary and the comparison table to check the run.

## Steps inside run_all.py

| Function | Step |
| --- | --- |
| `load_raw` | Traverses the raw file and reports shape, fields, splits and missing values |
| `clean` | Builds the derived variables and writes the cleaned analysis file |
| `table2` | State means (the paper's Table 2) |
| `table3` | Difference-in-differences table (Table 3) |
| `table4` | Reduced-form employment regressions (Table 4) |
| `table5` | Robustness checks (Table 5) |
| `table6` | Other outcomes (Table 6) |
| `table7` | Full-meal prices (Table 7) |
| `figure1` | Starting-wage distribution (Figure 1) |
| `extra_permutation` | Randomization inference extension, not in the paper |
| `compare` | Writes the comparison against the published values |
| `not_reproducible` | Lists the items that cannot be rebuilt with this data |

## Expected result

The console ends with a line in the form `92 items compared: 91 pass`. The comparison table has
92 rows, 91 of them marked `OK`, and exactly one flagged: the New Jersey indicator in the
robustness row that treats temporarily closed stores as permanently closed, where the replication
gives 2.028 against 2.20 in the paper. The gap coefficient in that same row reproduces at 14.424
against a published 14.420 and the other New Jersey values in the table all match, so the
published cell is inconsistent with the rest of its own row; the value is reported as it stands.
A run that flags any other row has not reproduced the documented results.

Values that confirm a correct run: regression sample 357, balanced sample 384, raw
difference-in-differences 2.75, New Jersey indicator with controls 2.304 with a standard error of
1.196.

## Reference material in this folder

- `README.md` - how the project is set up and how to run it.
- `AI_USE_DISCLOSURE.md` - how AI tools were used and how the output was checked.
- `report.pdf` - the research report.

## Notes

- The analysis is deterministic. The extension uses a fixed random seed, so repeated runs give
  identical files on any machine and from any working directory.
- Five items cannot be reproduced with the available data and are listed at the end of the run:
  the sampling frame of the paper's Table 1, the Newark and Camden zip-code subsamples, the
  McDonald's entry analysis, the Current Population Survey evidence, and the cash-register
  instrumental-variables check.
