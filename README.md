# HighTors

HighTors is a coupled backbone–amide torsion correction for N-methylated peptides.

Reproducibility materials for
*HighTors: Separating Backbone and Amide Torsion Effects in N-Methylated Fragments and Cyclic Pentaalanines*.

This repository contains the code, compact numerical data, original input
records and final figure-reproduction materials for the submission. The current numerical table/figure map is inside data/Supporting_Data.zip.

## Submission data

| File | Contents |
|---|---|
| [data/Supporting_Data.zip](data/Supporting_Data.zip) | Current numerical figure data, all 17 table exports, frozen coefficients and predictions, complete Table S5 evidence, QM inputs and available optimized coordinates, representative structures, and six derived tri-NMe time series |
| [data/Simulation_Inputs.zip](data/Simulation_Inputs.zip) | Physical systems, six original paired starting arrays, protocol records, seeds and original run contracts |

The archives are ready-to-download distributions. Existing qm/, validation/,
acquisition/ and md/ directories retain the analysis and simulation source and
their original numerical inputs. Duplicate data in the distributions are
intentional; SHA256SUMS covers both archives and the working source files.

## Reproduce the numerical results

Use Python 3.10+ and NumPy, and choose fresh output paths:

```sh
python -m pip install -r requirements.txt
python -B qm/code/compare_nme_fourier_pairwise.py --out qm_reproduced
python -B qm/code/verify_reproduction.py qm_reproduced
python -B validation/reproduce_validation_scores.py --out validation_reproduced.json
python -B acquisition/reproduce_acquisition_benchmark.py --out acquisition_reproduced.json
python -B md/code/audit_tri_nme.py --out md_reproduced
python -B md/code/audit_native_inputs.py --out native_inputs_reproduced.json
```

These commands use existing labels, inputs and derived arrays; they do not launch
QM or MD. Optional frozen GP/Fourier prediction replay also requires SciPy:

```sh
python -m pip install -r acquisition/requirements.txt
python -B acquisition/reproduce_acquisition_benchmark.py --verify-predictions --out acquisition_predictions_reproduced.json
```

| Check | Reference result |
|---|---|
| Earlier Ala selection set, 144 geometries | Pairwise / joint RMSE 2.291551 / 1.947528 kcal/mol |
| Independent Ala test, 144 geometries | Pairwise / joint RMSE 2.299531 / 1.999995 kcal/mol |
| Tri-NMe reference-basin occupancy, M0 / A only / Full | 7.643% / 86.857% / 91.596% |
| Native-input check | Three systems, six paired starts and 71 ROE atom bindings |

The earlier 144-point set was excluded from coefficient fitting and used for
representation selection. The independent test is separate. Val transfer uses
72 pre-existing geometries retrospectively; both local-increment and
common-reference full-correction contrasts are retained.

## Reproduce the final figures

The drawing scripts and required frozen vector panels, numerical inputs and
authors' manual ChimeraX images are in figures/. Follow [figures/README.md](figures/README.md):

```sh
python -m pip install -r figures/requirements.txt
python -B figures/build_submission_figures.py --out figure_outputs
```

The scripts assemble the approved vector panels and manually drawn molecular
images and plot the frozen numerical values. They do not generate new molecular
coordinates, perform new sampling or alter population estimates. The TOC uses
the frozen fitted J slice, reduced title lettering and an open coordinate frame.

## Contents and scope

| Directory | Contents |
|---|---|
| data/ | Current submission data distributions |
| qm/ | Frozen 246-label fit, earlier selection comparison and original label-generation source |
| validation/ | Independent Ala and retrospective Val inputs, labels, scores and raw-output audit source |
| acquisition/ | Original GP/Fourier modules, 12 training tables, 24 predictions and benchmark replay |
| md/ | Native models and starting arrays, production/analysis source, derived observables and representative structures |
| figures/ | Final figure scripts, frozen vector inputs and original manual molecular images |
| source_data/ | Historical numerical bundle retained for source traceability; use data/Supporting_Data.zip for current manuscript numbering |
| provenance/ | Original scientific-source identities and source-bundle mappings |
| docs/ | Reproduction scope and raw-output archive identities |

Full production coordinate trajectories are not distributed through GitHub
because of their size. Derived-run NPZs contain torsion/RMSD observables, not full
coordinate trajectories. Complete original QM output archives are also outside
this compact repository; docs/raw_data_archives.tsv identifies the known QM
archives and records their distribution status.

The public package supports the numerical checks described above. Recomputing
whole-ensemble contact/ROE values directly from coordinates requires the original
production trajectories. See [docs/reproduction_scope.md](docs/reproduction_scope.md).

Original state identifiers and source names are preserved inside scientific
records. CONTROL/OLD denote M0; TEST/NEW denote Full. SHA256SUMS excludes itself
and .gitignore. Use CITATION.cff for the software citation.

## Authors and funding

Ning Zhu, Jian Min Liu, Chongyang Li, Sen Cao, Chengyun Zhang, Jingjing Guo, Hongliang Duan.

Jian Min Liu: Institut für Chemie, Technische Universität Berlin, Strasse des 17. Juni 124, 10623 Berlin, Germany. All other authors: Macao Polytechnic University, Rua de Luís Gonzaga Gomes, Macao, China.

Corresponding author: Hongliang Duan.

This project was supported by the Macao Science and Technology Development Fund (Grant No. 0029/2025/RIB1) and the internal grant from Macao Polytechnic University (RP/FCA-11/2025).
