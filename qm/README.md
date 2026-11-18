# Matched Fourier fit and fragment-method source

The portable fit uses 246 frozen training labels and evaluates the earlier
144 representation-selection geometries. Those geometries were excluded from
coefficient fitting but used for representation selection. The new independent
144-point test is in ../validation/.

From the repository root:

```sh
python -B qm/code/compare_nme_fourier_pairwise.py --out qm_reproduced
python -B qm/code/verify_reproduction.py qm_reproduced
```

Expected pairwise/joint RMSE: 2.291551 / 1.947528 kcal/mol. The original basis,
regularization, centering and coefficient ordering are preserved.

original_method_source_v1.zip contains the original fragment QM-input/job,
same-geometry M0 label, output-validation and final-batch GP-selection source,
plus its model protocol. Source aliases and hashes are recorded inside its
MANIFEST.json. These lab-specific recipes retain their original path/guard
dependencies. Portable score checks use the supplied labels; native fragment
inputs and coordinates are in ../validation/native_inputs_v3.0.zip. The original
GP kernel, fitting and fantasy-batch selection functions are in ../acquisition/code/.
