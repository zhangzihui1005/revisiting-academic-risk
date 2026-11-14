# Temporal gains in academic-risk prediction

Reproduction code and frozen evidence, **v5.0 (2026-11-14)**, for *Temporal gains in academic-risk prediction depend on outcome definitions and evaluation conditions*.

This version adds the reviewer-revision analyses to the unchanged historical E1–E14 scientific evidence. The original commands remain available. The revision extension contains complete-risk-set year transfer and calendar-proxy controls, same-family three-feature comparisons, nested development stability, intercept-only and Platt calibration diagnostics, capacity/lead-time summaries, and the numerical audit of the historical LR refit.

## Start here

Use Python 3.12 and an isolated environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python reproduce.py check
python reviewer.py verify --workdir _reviewer
```

`reviewer.py` joins the `reviewer_v5.part*` transport files, verifies SHA256, and restores human-readable source and frozen evidence. It recomputes metrics and intervals from **existing predictions and saved draws**, checks nested split separation, and performs no model fitting or new bootstrap. Use `restore` instead of `verify` to unpack with Python's standard library only. Do not edit the extracted frozen evidence; use a separate destination for new runs.

The restored `_reviewer/` tree contains `code/` (nine readable Python files), `tables/` (21 full-precision CSVs), `reference_calibration/`, `fullrisk/`, `stability/`, `calibration/`, `inputs/`, `PROTOCOL.md` and `SOURCE_INDEX.json`. Individual prediction rows, locked parameters, source search results, membership IDs, random states and saved draws are included. Raw interaction logs and model binaries are omitted. Historical source audits remain clearly dated historical records.

## Rebuild the revision controls from raw OULAD

Obtain the original public OULAD ZIP from the [official UCI record](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset). The launcher enforces SHA256 `f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4`.

```sh
python _reviewer/code/reproduce_core.py --raw "/path/to/OULAD.zip" --destination _new_run --stage all
python _reviewer/code/reproduce_calibration_core.py --prepared-root _new_run --destination _new_lomo
python _reviewer/code/reproduce_calibration_diagnostics.py --destination _new_diagnostics
```

These are **explicit computational runs**. The first rebuilds Day28/56 features and reruns the new complete-risk-set and nested-development controls; `--stage prepare`, `fullrisk`, and `stability` can be run separately. The second refits the fourteen historical selected LOMO LR workflows and reconstructs source OOF/calibration from rebuilt features; it does not repeat all historical hyperparameter searches. The third recomputes diagnostics and bootstrap from bundled historical OOF/target probabilities. Use fresh output destinations. Exact results can depend on library and platform versions; compare to the frozen tables and record deviations.

Historical saved-result verification and fixed-fit replay are still available with `reproduce.py` and `revision.py` (see below). This is not a one-command reconstruction of every historical search. No unperformed replay is claimed.

## Map to the revised article

| Revised evidence | Restored data / source |
| --- | --- |
| Main complete-risk-set results; Tables S33, S35–S36 | `tables/fullrisk_temporal_contrasts.csv`, `fullrisk_capacity_metrics.csv`, `representation_contrasts.csv`; `code/fullrisk.py` |
| Screening coverage and withdrawal lead time | `tables/waiting_time_coverage.csv`, `KEY_capacity_and_leadtime.csv` |
| Nested development stability | `tables/nested_development_*.csv`; `code/stability.py` |
| Calibration and simple probability baselines | `tables/calibration_*.csv`, `historical_class_weight_audit.csv`; `code/calibration_analysis.py` |
| Reviewer numerical checks | `tables/R1_*.csv`; frozen audit records |
| Calendar assumptions | `tables/calendar_date_bounds_and_assumptions.csv`; `PROTOCOL.md` |
| Historical E1–E14 and original figures | Existing `results/`, `docs/`, `evidence/`; prior commands below |

The numbered historical crosswalk in `docs/MANUSCRIPT_CROSSWALK.csv` describes the pre-reviewer manuscript and is not the numbering of the November revision. The new results are exploratory analyses of previously inspected data. Historical E11 is internal-only; the new complete-risk-set year-transfer analyses are separate. Calendar proxies do not verify actual label availability. Conditional failure and complete-risk-set Fail target different populations. Intervals are pointwise and conditional on fitted models, and overlapping development folds are not independent replications. There is no prospective intervention test.

## License and citation

Authors' original code and accompanying software documentation are licensed under **MIT** (`LICENSE`). Original datasets and derived data retain **CC BY 4.0** attribution requirements; see `LICENSING.md`. Manuscript files, referee reports, passwords, and raw interaction-log archives are not part of this release.

Citation metadata are in `CITATION.cff` and `.zenodo.json`. Cite the version-specific Zenodo DOI displayed on the release record once it has been archived. The article DOI is not yet available and is not invented here.

## Historical v4.0 interface reference

The text below documents the prior release, retained for command compatibility. Its references to "current manuscript" and numbering are historical; the v5.0 scope above governs the reviewer-revision extension. The original no-software-license statement below is superseded by the MIT license granted in v5.0.

# Temporal gains in academic-risk prediction

Reviewer reproduction package **v4.0**, aligned with the revised manuscript:
**Temporal gains in academic-risk prediction depend on outcome definitions and evaluation conditions**.

Repository: https://github.com/zhangzihui1005/revisiting-academic-risk

This release retains the E1–E9 evidence subset and adds final E10–E14 evidence.
It provides executable saved-result verification and selected training replays.
It is **not a one-command reconstruction of every historical E1–E14 experiment**.
See [coverage and limitations](docs/REPRODUCTION_SCOPE.md) before running archived scripts.
Manuscript documents, raw OULAD logs, and model binaries are omitted.

## Quick verification (no training)

Use Python 3.12 and an isolated environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python reproduce.py check
python reproduce.py verify --workdir _review
python reproduce.py preflight --workdir _preflight
```

`check` uses only the Python standard library. `verify` restores the bundled evidence,
recomputes core E1–E9 metrics, 79 added paired horizon comparisons, 38 fixed-fit metric
rows, E11 full-risk metrics/confusions, E12 proxy metrics, and E14 loss contrasts.
Intervals are recovered from **existing saved draws**, not resampled.
S32 contrasts are reconstructed only after checking synchronized saved draws and RNG states.
`preflight` checks hashes, split isolation, feature/configuration hashes, and corrected
UCI training order without fitting models. Use a fresh directory for each preflight run.

Reports are written to `_review/VERIFICATION.json`, `_review/REVISION_VERIFICATION.json`,
and accompanying verified CSVs. Omitted old prediction rows are counted explicitly.

## Fixed-configuration replay (explicit training)

Prepared train/test feature tables, membership/order, parameters, seeds, and original
predictions are bundled for 12 E10B and 26 final E13 fits. List the exact jobs:

```bash
python revision.py list-fixed --workdir _fixed
python revision.py replay-fixed --job E10B_internal_LR_D28 --workdir _fixed
python revision.py replay-fixed --job E10B_internal_LR_D56 --workdir _fixed
python revision.py replay-fixed --job E13_internal_H0_strict_LR --workdir _fixed
```

Each explicit command refits one model with the original model factory/preprocessing,
then compares every prediction with its frozen reference (absolute tolerance 1e-7).
It performs no Optuna search, target-driven selection, or bootstrap. Replays start from
prepared feature tables; they do not independently reconstruct raw OULAD features.
Versions/platforms can affect numerical reproducibility; differences fail visibly.

The original supported workflows remain available:

```bash
python reproduce.py raw-oulad --oulad /path/to/OULAD.zip --workdir _raw
python reproduce.py rerun-uci --workdir _uci --workers 3
```

These are substantial, explicit training workflows. The OULAD command requires the
matching original ZIP and reconstructs the previously supported key failure pipeline
(105 search trials); the UCI command repeats the original E5A information-horizon
workflow, including search and scoring. Neither reconstructs all E1–E14.
See `docs/DATA_SOURCES.txt`, `docs/PROTOCOLS.txt` and the scope document.

## Results and scientific boundaries

- `results/S26*`–`S31*`: full-precision final added result panels, including corrected E13 v2.
- `results/S23_historical_ablation_loss.csv`: historical intervals with full-minus-reduced sign.
- `results/S32_verified_posthoc_contrasts.csv`: four explicitly post-hoc saved-draw contrasts.
- `docs/MANUSCRIPT_CROSSWALK.csv`: current main tables/figures and supplement source mapping.
- Existing `results/Results_Table*.csv` names refer to the **original frozen closeout**, not
  the current manuscript's table numbering.
- `code/revision_*.py`: byte-identical added-experiment source mirrors. Several expect the
  author's full archive layout; use `revision.py` for the portable fixed-fit interface.
- `evidence/*.part*`: small binary transport pieces, automatically joined and extracted.
  Do not upload the outer ZIP as a repository file, rename chunks, or omit them.

E11 is internal only. E12 uses calendar proxies with unverified actual label-release
dates. E13 removes macroeconomic fields and uses corrected training order, not new model
selection. E14 equal-module and pooled weighting are different estimands. Conditional
failure excludes eventual withdrawers; common cohorts remain retrospective. Added
intervals are conditional, pointwise descriptive results without familywise adjustment.
S32 is post-hoc; no intervention effect or prospective deployment test is claimed.

## Provenance and upload

All frozen payload files have exact SHA256 values in `evidence/*SOURCE_INDEX.json`.
`FILE_MANIFEST.json` checks repository contents; declared UTF-8 text accepts CRLF/LF
transport changes only. Frozen payload bytes always require exact hashes.
`package_verification.json` records release QA. Archived audit records do not imply a
new independent full-model replay in this release.

Unzip and upload the **contents of `revisiting-academic-risk`** to the repository root.
See `UPLOAD_GUIDE_CN.txt`. Keep the existing repository name. Licensing and dataset
attribution are described in `LICENSING.md`; no new software license is assigned here.
