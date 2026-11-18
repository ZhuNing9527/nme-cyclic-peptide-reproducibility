# Acquisition benchmark

Reproduces manuscript Table S5: three acquisition strategies, four added-label
budgets (24, 48, 96, 192) and two fixed evaluators. Each fit uses 54 seeds plus
the acquired labels. This benchmark uses the earlier 144-point selection set,
separate from the independent Ala test in ../validation/.

```sh
python -B acquisition/reproduce_acquisition_benchmark.py --out acquisition_reproduced.json
```

The standard-library replay verifies the curated source files and reconstructs
3,456 pointwise errors, overall and regional metrics, six learning-curve areas
and the frozen comparison decision. It uses all 12 training manifests and all
24 prediction tables.

```sh
python -m pip install -r acquisition/requirements.txt
python -B acquisition/reproduce_acquisition_benchmark.py --verify-predictions --out acquisition_predictions_reproduced.json
```

The optional mode uses the exact original GP and Fourier modules and their
archived parameters. It does not optimize hyperparameters again. The original
periodic_gp_v1.py includes kernel, fitting, posterior and fantasy-batch selection
implementations. Original evaluator/scoring drivers remain in archived_drivers/;
the compact upstream label/selection source is in ../qm/original_method_source_v1.zip.
Scientific input authorities and strategy sequences remain in authorities/.

| Strategy | Fourier AULC | Fourier final RMSE | GP AULC | GP final RMSE |
|---|---:|---:|---:|---:|
| Maximin | 3.215183 | 3.181236 | 2.304398 | 1.814072 |
| GP acquisition | 3.319277 | 3.221794 | 2.665802 | 1.975667 |
| Random | 3.235179 | 3.199054 | 2.547639 | 2.083284 |

Units are kcal/mol. AULC is the trapezoidal RMSE integral over log2(budget/24),
divided by 3. Maximin has the lowest final error and area under both evaluators.
The frozen cross-model acquisition-advantage gate is not met. All three
strategies and every budget remain in the data.
