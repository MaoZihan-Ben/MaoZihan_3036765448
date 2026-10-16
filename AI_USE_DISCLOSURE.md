 AI-Use Disclosure

I used an AI assistant on this project, mainly to write code and to help me read the paper. The ideas, the strategy and the decisions were mine: I chose the paper, I set out the difference-in-differences design, I defined the analysis sample and the treatment of closed stores, and I decided how to read the results and what to report. The three written documents, this statement, the README and the report, are my own writing; I used the assistant for drafting support and rewrote the text myself. What follows is where the assistant was used, how I checked what it produced, and which of its suggestions I corrected or threw out.

 Where and how AI was used

Three things. First, pages were turned into images and the published numbers were read off them to serve as benchmarks. Second, the code. The assistant drafted `code/run_all.py` from a plan I laid out, working from my instructions about the order of the steps, the definition of each variable and the choice of estimator, which I took from the way the paper organises its own analysis. Third, Everything that set the direction was mine, and the code was written against those decisions rather than in place of them.

 How I checked the AI-generated code, analysis and writing

The main check is automatic. The script keeps a record of every published benchmark and writes a comparison file in which each reproduced number sits next to the published one with a pass or flag marker. Of the 92 compared values, 91 pass, and the single exception is described below.

I also read the benchmarks twice. They were read off images of the published tables and, wherever the paper reports the same quantity in two places, checked against each other. That caught one misreading of mine: the change in the share of stores with a register open at eleven o'clock was first put down as -0.20, which contradicted the New Jersey indicator printed in the same table, and a second look gave 0.17, which fits.

I re-ran the code to make sure it gives the same answer every time, from a cleared output state and from different working directories, and the outputs came out identical. The two sample sizes match the paper as well: the regression sample has 357 observations and the balanced sample has 384.

 Important AI errors or suggestions that I corrected or rejected

 replacement call that would have wiped out real values. An early version used a replace function to turn a zero into a missing value; that would also have wiped non-missing zeros. I had it changed to a masked assignment.

variable name with a decimal point inside it, which broke the parser when the regression formula was read. The variable was renamed.

A robustness row that came out identical to the row above it. The row treating temporarily closed stores as permanently closed gave exactly the same numbers as the balanced-sample row, because the sample definition excluded those stores and the row was doing nothing. I caught that and had the sample definition fixed, after which the row reproduced the published values.

An outcome variable that treated missing answers as zeros. The meal-program block was computed with missing responses sitting in the denominator, which threw off every number in that block. Masking them out brought it into exact agreement with the paper.


