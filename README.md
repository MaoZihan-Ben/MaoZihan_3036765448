Replication of Card and Krueger (1994): Minimum Wages and Employment

This repo reproduces Card and Krueger (1994), "Minimum Wages and Employment: A Case Study of the
Fast-Food Industry in New Jersey and Pennsylvania", American Economic Review 84(4), 772-793. It
rebuilds the paper's tables and the wage figure, puts every number side by side with the published
one, says which parts cannot be redone with the data we have, and adds one extra analysis of my own.

How to run it. Clone or download the repo and then, from the repository root, run
`python code/run_all.py`. That is the whole thing: one command, nothing else to prepare. You need
Python with pandas, numpy, statsmodels and matplotlib (I ran it on Python 3.13, Anaconda). The
script finds all its folders from its own location, so it does not matter where the repo sits; run
it from the root just so the printed paths read nicely. It writes only into `data/processed/` and
`outputs/`, and it never touches the original data file, so you can re-run it as often as you like.

What is in here. `code/run_all.py` is the only script: the entire workflow lives in that one file,
with one function per step, and that single script is the reusable workflow the brief asks for.
`data/raw/public.csv` is the original survey data, 410 stores and 46 columns, and it is the only
input the analysis needs. `data/processed/cleaned_did.csv` is the cleaned analysis file the script
writes. Everything the script produces lands in `outputs/tables` and `outputs/figures`. The report
is `report.docx` and the AI statement is `AI_USE_DISCLOSURE.md`.

The data. `public.csv` comes with the course as package 01. It has 331 New Jersey stores and 79 in eastern
Pennsylvania, each surveyed twice, in February and March 1992 and again in November and December
1992. I leave the raw file alone, and the cleaned file is rebuilt from it on every run.

Checking that it worked. The console prints a summary line ending in "92 items compared: 91 pass".
The comparison table `outputs/tables/comparison_with_paper.csv` has 92 rows, 91 of them marked OK,
and exactly one flagged: the New Jersey dummy in the robustness row that treats temporarily closed
stores as permanently closed, where I get 2.028 and the paper prints 2.20. If any other row is
flagged, something went wrong. A few numbers worth eyeballing: the regression sample is 357, the
balanced sample is 384, the raw difference-in-differences is 2.75, and the New Jersey dummy with
controls comes out at 2.304 with a standard error of 1.196.

Which file is which. `table2_means` is the state-means table, `table3_did` the
difference-in-differences table, `table4` to `table7` the employment regressions, the robustness
checks, the other outcomes and the prices, and `figure1_wage_distribution` the wage distribution.
`comparison_with_paper.csv` has everything side by side, and the two `extra_permutation` files with
their figure are the part that is not in the paper.

How close it is. 91 of the 92 values match to the rounding used in the paper. The exception is the
cell above: the gap coefficient in that same row comes out at 14.424 against a published 14.420,
the standard error is 1.219 against 1.21, and the other New Jersey dummies in that table all match,
so the printed 2.20 does not look consistent with the rest of its own row. I report it as it is
rather than nudging anything. Five things cannot be redone with this data, and the script prints
them at the end: the sampling frame and response rates of Table 1, the Newark and Camden zip-code
subsamples, the McDonald's entry numbers, the CPS state evidence, and the cash-register
instrumental-variables check.

The extra bit. On top of the replication I added a placebo test for the main estimate. I reshuffle
the state labels inside the sample 1,000 times and re-run the same regression each time, keeping
the number of stores fixed. The observed coefficient is 2.304, the placebo coefficients average
0.015 with a standard deviation of 1.196, and only 28 out of the 1,000 reach the observed value, a
one-sided p-value of 0.028. So the New Jersey gain does not look like noise, and the seed is fixed
so it reproduces exactly.

Everything is relative to the script's own location, there are no hardcoded paths, and the results
come out identical on every machine and in every directory.
