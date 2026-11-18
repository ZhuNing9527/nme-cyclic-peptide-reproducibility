# Native MD inputs and structural analysis

inputs/ contains the exact physical models and paired starts for mono-, di-
and tri-NMe. ALL_SYSTEM_INPUTS_v1.json indexes the three systems, 12 original
formal-run input contracts and 71 reviewed ROE bindings. Existing tri-NMe files
are at inputs/' root; mono/di inputs are in their named subdirectories.

From the repository root:

```sh
python -B md/code/audit_native_inputs.py --out native_inputs_reproduced.json
python -B md/code/audit_tri_nme.py --out md_reproduced
```

The first command audits native models, coefficient/sign conventions, six
starting arrays and all 71 ROE atom bindings. The second replays the tri-NMe
derived-run statistics and representative-coordinate calculations. Neither
imports OpenMM or runs dynamics. runtime/ retains the original REST2 transform,
state adapter and optional OpenMM force-definition check.

production_source_v1.zip consolidates the original production worker, provided
dependencies and protocol/start authorities. original_analysis_source_v1.zip
retains the original exchange-diagnostic, RMSD/PMF, hydrogen-bond and M1–M3/ROE
analysis implementations. The bundles preserve source bytes and internal lab
paths; their manifests identify every retained source. original_run_records_v1.zip
contains the formal run metadata, including recovery boundaries.

The complete di-NMe frozen source package is retained in analysis_tools/.
The original mono-NMe standalone launcher was not recovered; its exact native
inputs, reviewed binding and shared numerical/graph implementations are supplied.
Production coordinates remain separately archived; the supplied derived arrays
and representative structures have the coverage described in the root README.
