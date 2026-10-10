# 0030. Judge calibration reports quadratic weighted kappa

- **Status:** Accepted
- **Date:** 2026-10-10

## Context

The judge scores each variant from 1 to 5 on the brand anchors (0029). The
architecture asks for hand scores on a 15-output sample, and for a script that
reports agreement with the judge. Low agreement means the rubric should be
tuned before anyone treats the judge mean as the quality number. The sample
has to be stable in git, and CI has to check it without calling a model.

## Decision

`evals/calibration/` holds 15 YAML files, five briefs from each brand: the
first five cases of brightleaf, ledgerly and voltride. Each file is one
authored variant for that brief, plus a hand score for every rubric criterion
and a note that says why. The copy was written for this sample so the 1–5
scale is actually used, including deliberate misses. It is not a saved
baseline or pipeline run.

`brandforge calibrate` loads those files and calls `score_row` once per
output, on the judge tier, with the same prompt the eval uses. The hand
scores and the note are not part of the row the judge sees. The script
prints:

- Quadratic weighted Cohen's kappa on the criterion integers. The scale is
  always 1–5, including levels the sample did not use. A gap of 1 has weight
  1/16. A gap of 4 has weight 1.
- Exact agreement, agreement within one point, and the mean absolute error.
- The mean signed error (judge minus human), so a judge that sits one point
  high is visible.
- The same figures per criterion and per brand, and the absolute error of the
  row mean, which is the number promptfoo shows.
- Every pair that differs by two points or more.

Kappa below 0.60 is the low band (below the usual "substantial" line). The
verdict says to tune the rubric anchors. The command still exits 0. A low
figure is the result the sample exists to surface. Status 1 is reserved for
a sample that will not load, or a judge that scored nothing. Kappa is
undefined when the scores sit on a single level, and the verdict says so
instead of printing 1.

Unit tests pass a fake gateway. CI does not run `brandforge calibrate`.

## Alternatives considered

- **Hand-score live model output and commit that.** The sample would change
  whenever the prompts changed, and the task could not land without a paid
  run. The judge scores the text it is shown. Authored text that names the
  brief is a stable probe. A later results file (BF-38) can still be scored
  by hand if a live sample is wanted.
- **Unweighted kappa, or Pearson correlation, as the headline number.** A
  5-versus-1 miss should count more than a 5-versus-4 miss. Correlation can
  stay high when the judge is shifted by a point. Signed error is printed
  beside kappa so the direction of a shift is still visible.
- **Compute the weights only on the levels that appear.** A sample that only
  hits 4 and 5 would then treat a one-point gap as total disagreement.
  The rubric has five levels, so the weights use five levels.
- **Fail the command when kappa is under 0.60.** That would turn a measurement
  into a gate. The smoke eval (BF-38) is the CI gate. Calibration reports.
- **Put 0.60 in config.** It is a property of this comparison, not a knob
  for a deployment. Changing it is a change to this decision.
- **One overall hand score per output.** The judge's integers are per
  criterion. Agreement on those integers shows which anchor drifted. The row
  mean is reported as well, because that is the eval's headline score.

## Consequences

- **Gained:** `brandforge calibrate` reports agreement between the hand
  scores and the judge, on the same grader the eval uses. pytest checks the
  arithmetic and the 15 files with no model call.
- **Trade-off accepted:** the 15 outputs are fixtures. They cover the scale
  on purpose, so they are harsher than a random draw of pipeline copy.
- **Trade-off accepted:** the row handed to the judge uses the baseline
  shape, with `flagged` null. The judge does not read that field. The files
  are not claimed to be baseline runs.
- **Trade-off accepted:** a one-point gap is counted in the summary and left
  off the gap list. The list is for misses of two points or more.
