# Reproduction scope for the revised manuscript

| Component | Bundled evidence / executable support | Boundary |
|---|---|---|
| Original E1–E9 | Original core predictions, selected configuration/selection records, summaries and saved draws; existing verifier | Some old results remain frozen summaries without bundled predictions; verifier reports the count |
| Original raw OULAD and E5A | Existing explicit raw-OULAD and UCI training launchers retained | Selected pipelines only; raw OULAD ZIP supplied separately |
| E10A | 57 paired same-family files, saved draws, cohort metadata | Four-landmark and two-landmark common cohorts differ; no refit implied |
| E10B | 6 paired contrasts, 12 train/test configurations and predictions, prepared inputs, portable per-model refit | Day-28 source parameters fixed; preprocessing refit per horizon; year split retrospective |
| E11 | Dynamic/common predictions, all four arms, multiclass scores and confusion counts, 7 saved paired comparisons; original execution source | Internal only; original Day-28 models and full archive needed to rerun whole pipeline; dual withdrawal branch shared |
| E12 | Course calendar audit, source eligibility, original/proxy scores, proxy predictions, trials/selection/calibration metadata, saved paired comparisons; execution source | Proxy dates only; 0/30/60 lags select identical sources. No verified label-release dates; full original archive needed for complete source search replay |
| E13 | Final corrected-v2 9 internal + 17 course fits, prepared inputs/order, parameters, predictions and 6 paired contrasts; portable refit | Excludes superseded initial ordering; retains course 33. No retuning, no dropout dates |
| E14 | Matched raw/Platt/prior probabilities, 5,000 global student-cluster draws, 8 loss contrasts | OULAD only; prior constant is fold-specific source prevalence; UCI aggregate CI unavailable |
| SI S23 | Original full/reduced ablation table with original intervals; deterministic sign reversal | Source table verification, not new bootstrap or model fit |
| SI S32 | Four deterministic differences between synchronized saved paired gains | Newly packaged reconstruction matches manuscript rounding. Original revision script named in Word was not located; `revision.py:derived` is the replacement implementation, not a historical original |

## Commands and dependencies

`python reproduce.py verify` checks old core plus all added saved evidence. It does not
train any model. `python revision.py verify` checks only added evidence.
`python revision.py replay-fixed --job ...` independently refits the requested E10B/E13
model from bundled prepared data; `list-fixed` enumerates 38 jobs. It compares exact
record identities and probabilities against original outputs. It does not refit all
38 models implicitly, tune parameters, or generate new intervals.

The eight `code/revision_*.py` mirrors retain original paths/import interfaces for
method transparency. Do not launch them as standalone one-click programs. Their
original layout is `research_project/scripts`, `research_project/experiments_new`,
and `research_project/archived_E1_E9/{outputs,work,raw_data}`. The reduced public package
does not contain all fitted original E2/E4/E6 models or every raw historical input.
The portable wrapper remaps only bundled inputs, never executes an archived absolute
path, and writes new files exclusively below the reviewer work directory.

## Interpretation and provenance

The 79 contrasts retain 158,000 existing horizon draws; E14 retains 5,000 global draws.
No additional bootstrap results were generated during packaging. Confidence intervals
condition on fixed fitted models/splits and do not include redoing model selection.
New controls are revision-stage validation/sensitivity analyses, not historically
preregistered endpoints. Original multiplicity sensitivity applies only to its stated
original test families. S32 is explicitly post-hoc with no multiplicity adjustment.

E13's 26 superseded fits and 12,000 superseded draws are excluded from this release.
They remain in the author's archive. Early E12 CSV `not_run_initial_batch` flags are
historical audit timestamps; final completion is recorded separately in
`revision_outputs/audit/E10_E14_final_validation.json`. Calendar-proxy completion does
not establish actual label availability. Original E1–E9 output files were not revised.

The public files support substantive reviewer checking and selected model refitting;
they do not claim complete raw-data-to-every-table reproduction of all experiments.
Neither original Word document is distributed or modified. Their hashes appear in
`revision_source_map.json`. Some source metadata preserve historical author-machine
paths as provenance; portable commands resolve bundled relative paths instead.
