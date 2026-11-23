# Reproduction scope

| Calculation or output | Distributed material | Scope |
|---|---|---|
| Matched Fourier comparison | 246 frozen labels, fitting code, earlier 144-point scores and coefficients | Numerical fit and score reproduction; full raw electronic-structure outputs are separate |
| Independent Ala test | 144 labels, native input/coordinate bundle, predictions and bootstrap source | Frozen-model scores and intervals; raw output audit additionally needs the original result archive |
| Retrospective Val transfer | 72 geometries, topology/charges, both contrasts, predictions and score source | Both published contrasts; raw output audit additionally needs the original result archive |
| Acquisition benchmark | GP/Fourier source, 12 training tables, 24 predictions, frozen fit parameters and protocol records | Scores, learning curves and frozen-parameter prediction replay |
| Native MD models | Physical systems, six paired boxed starts, run contracts, original REST2 source and 71 ROE bindings | Static inputs and initial-coordinate checks; no claim to recalculate complete production ensembles from starting coordinates |
| Tri-NMe populations and RMSD | Six per-frame derived arrays and representative coordinates | Published M2 masks, 30 block counts, RMSD summaries and representative-frame geometry |
| Whole-ensemble structural analysis | Original contact/ROE/Ramachandran/exchange analysis source and published summary tables | Recalculation from coordinates requires original production trajectories |
| Final figures and TOC | figures/ scripts, frozen vector panels, numerical inputs and supplied manual molecular images | Approved figure assembly, distance-label refinement, compact panel proportions and TOC rendering; no new QM or MD |

Current table and figure data are in data/Supporting_Data.zip. Native systems,
starting arrays and protocol records are in data/Simulation_Inputs.zip.
source_data/Supporting_Data_v3.0.zip is a historical source bundle and does not
define current manuscript numbering.

Full production trajectories are excluded from GitHub distribution because of
their size. Full original QM output archives are not included in the compact
repository. These distribution limits do not make the derived time-series
arrays equivalent to raw coordinates or raw electronic-structure output files.
