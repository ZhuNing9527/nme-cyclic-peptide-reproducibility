# Frozen-model validation for manuscript v3.0

This directory extends the original core with a new independent 144-geometry
NMe-Ala test and retrospective transfer to 72 existing NMe-Val geometries.
The 217-coefficient pairwise and 729-coefficient joint models remain frozen.

From the repository root, with Python 3.10+ and NumPy installed:

```sh
python -B validation/reproduce_validation_scores.py --out validation_reproduced.json
```

The command reconstructs predictions from the coefficients and observed
labels, checks the reported metrics, and reproduces the Ala overall paired
bootstrap interval. It does not launch QM, refit coefficients, or run MD.

| Comparison | Non-reference points | Pairwise RMSE | Joint RMSE |
|---|---:|---:|---:|
| Ala independent test | 144 | 2.299531 | 1.999995 |
| Val primary, local trans increment | 45 | 3.116316 | 1.827676 |
| Val alphaL, local trans increment | 15 | 3.719710 | 1.377983 |
| Val primary, common-reference full correction | 53 | 4.442145 | 2.503872 |
| Val alphaL, common-reference full correction | 18 | 7.203586 | 5.083867 |

Units are kcal/mol. The 54 primary and 18 alphaL Val geometries are scored
under two different physical reference contrasts; the rows are not additional
QM jobs. The trans references are excluded from each relevant RMSE. In the
Ala trans subset the joint error is higher; all subset results are retained
in data/Ala/holdout_metrics.tsv. The older 144 representation-selection points
remain separate in qm/ and data/Ala/prior_scoring_144.tsv.

## Native inputs

Unpack the small native-input archive in this directory if coordinates and
ORCA inputs are needed:

```sh
python -m zipfile -e validation/native_inputs_v3.0.zip validation
```

It contains 144 Ala main inputs plus 24 unused reserves, 72 Val inputs, their
available start/optimized geometries and frozen topology/charge files.
NATIVE_INPUT_PROVENANCE.json records the original source-member names and
hashes. No automatic job launcher is supplied here.

## Raw-result audits

The following commands require the separately distributed original archives
listed in ../docs/raw_data_archives.tsv:

```sh
python -B validation/audit_independent_holdout.py --archive /path/to/NME_CLEAN_HOLDOUT_RESULTS_v1.2.tar.gz --reference-data validation/data/Ala --out ala_raw_audit.json
python -B validation/rescore_nmeval_transfer_v1.py --archive /path/to/NMEVAL_EXISTING_FORMAL_INPUTS_AND_RESULTS_v1.tar.gz --coefficients validation/data/Val/frozen_coefficients.tsv --protocol validation/data/Val/RESCORING_PROTOCOL_v1.json --out val_rescored
```

These scripts read original outputs and verify their frozen hashes, energy
labels and geometry checks. They do not invoke ORCA, PMEMD, OpenMM or a GPU.
Original GP/evaluator implementations remain in ../acquisition/; upstream
fragment label/selection recipes are in ../qm/original_method_source_v1.zip.
Full production trajectories are separate data; see ../docs/reproduction_scope.md.
