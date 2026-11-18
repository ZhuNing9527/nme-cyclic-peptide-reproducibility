# NMe cyclic peptide reproducibility

**v1.5.0**, companion to manuscript v3.0, *A Backbone-Dependent Torsion Correction
for N-Methylated Cyclic Pentaalanines*.

This release contains the fragment fitting and validation data, original GP
implementation, native peptide models and starting arrays, structural-analysis
source, and numerical figure data. Scientific inputs, coefficients, predictions
and repeat-level data are unchanged from v1.4.1. Server collection tools,
development/repair history and duplicate analysis packages have been removed.

## Reproduce the reported numerical checks

Use Python 3.10+ and NumPy. From the repository root, choose fresh output paths:

```sh
python -m pip install -r requirements.txt
python -B qm/code/compare_nme_fourier_pairwise.py --out qm_reproduced
python -B qm/code/verify_reproduction.py qm_reproduced
python -B validation/reproduce_validation_scores.py --out validation_reproduced.json
python -B acquisition/reproduce_acquisition_benchmark.py --out acquisition_reproduced.json
python -B md/code/audit_tri_nme.py --out md_reproduced
python -B md/code/audit_native_inputs.py --out native_inputs_reproduced.json
```

These commands use existing labels and coordinates; they do not launch QM or MD.
The optional GP-prediction replay uses SciPy:

```sh
python -m pip install -r acquisition/requirements.txt
python -B acquisition/reproduce_acquisition_benchmark.py --verify-predictions --out acquisition_predictions_reproduced.json
```

| Check | Reference result |
|---|---|
| Earlier Ala selection set, 144 geometries | Pairwise / joint RMSE 2.291551 / 1.947528 kcal/mol |
| Independent Ala test, 144 geometries | Pairwise / joint RMSE 2.299531 / 1.999995 kcal/mol |
| Tri-NMe reference-basin occupancy, M0 / A only / Full | 7.643% / 86.857% / 91.596% |
| Native-input audit | Three systems, six paired starts and 71 ROE atom bindings |

The earlier 144-point set was excluded from coefficient fitting and used for
representation selection. The independent 144-point test is separate. Val
transfer uses 72 existing geometries retrospectively; both local-increment and
common-reference full-correction results are retained in validation/.

## Contents

| Directory | Contents |
|---|---|
| qm/ | Frozen 246-label fit, earlier 144-point comparison and compact original label-generation source |
| validation/ | Independent Ala and retrospective Val inputs, labels, scores, bootstrap and raw-output audit code |
| acquisition/ | Original GP/Fourier modules, 12 training manifests, 24 predictions and benchmark replay |
| md/ | M0/Full native systems and starts for mono/di/tri-NMe, tri-NMe A-only reconstruction, REST2 source, derived data and representative structures |
| source_data/ | Numerical sources for the main and supporting figures |
| provenance/ | Original scientific-source identities, public aliases and verification summary |
| docs/ | Data-archive identities and reproduction coverage |

Within md/, the original production implementation is consolidated in
production_source_v1.zip. Original RMSD, projected-free-energy, intramolecular
hydrogen-bond, REST2-diagnostic and M1–M3/ROE analysis sources are retained in
original_analysis_source_v1.zip. The full frozen di-NMe analysis package is in
analysis_tools/; earlier overlapping packages were removed. Formal run records
remain in original_run_records_v1.zip.

The original source bundles preserve their source bytes and lab-path guards.
Their manifests map descriptive public aliases to original file identities.
Portable reproduction uses the commands above. The OpenMM construction check
in md/runtime/openmm_static_check.py is optional and requires OpenMM 8.4.0 and
openmmtools 0.26.0; it constructs force definitions without a Context or dynamics.

## Data and provenance

The six derived-run NPZs contain torsion/RMSD metrics, not complete production
coordinates. Starting NPZs are exact boxed position/velocity inputs. Full raw
ORCA outputs and production coordinates require separately supplied archives;
the known QM archive hashes are in docs/raw_data_archives.tsv. Recomputing
whole-ensemble ROE/contact averages requires those original production frames.
Coverage is summarized in docs/reproduction_scope.md.

Original state identifiers and frozen source names remain unchanged inside
scientific records. provenance/sources.json records their identities and public
locations. SHA256SUMS covers distributed files except itself and .gitignore.
Use CITATION.cff for the software citation. No software license or archival DOI
has been assigned in this prepared release.
