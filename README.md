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
