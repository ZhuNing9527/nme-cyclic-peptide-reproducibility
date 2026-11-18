#!/usr/bin/env python3
"""Replay the archived acquisition benchmark without new labels or optimization.

The default mode uses the Python standard library. --verify-predictions also
checks the original Fourier basis and GP posterior at frozen fitted parameters;
that mode requires NumPy and SciPy. No QM, dynamics or hyperparameter search.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def close(actual, expected, tolerance=2e-9):
    require(abs(actual - float(expected)) <= tolerance,
            'Numerical value differs: %r versus %r' % (actual, expected))


def rmse(values):
    return math.sqrt(math.fsum(x*x for x in values) / len(values))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(folder, output, verify_predictions):
    data = folder / 'data'
    manifest = json.loads((folder / 'PROVENANCE.json').read_text())
    for item in manifest['files']:
        require(sha(folder / item['file']) == item['sha256'], 'SHA mismatch: ' + item['file'])
    truth = rows(data / 'validation_labels.tsv')
    require(len(truth) == 144, 'Expected 144 validation labels')
    truth_by_id = {r['state_id']: r for r in truth}
    require(len(truth_by_id) == 144, 'Duplicate validation states')
    index = rows(data / 'prediction_index.tsv')
    fits = {r['fit_index']: r for r in rows(data / 'fit_metadata.tsv')}
    archived_metrics = {r['fit_index']: r for r in rows(data / 'cell_metrics.tsv')}
    archived_aulc = rows(data / 'aulc_summary.tsv')
    expected_cells = {(arm, budget, ev) for arm in ('GP', 'MAXIMIN', 'RANDOM')
                      for budget in (24, 48, 96, 192) for ev in ('FOURIER_K2', 'PERIODIC_GP')}
    require(len(index) == len(fits) == len(archived_metrics) == 24, 'Expected 24 fits')
    require({(r['arm'], int(r['budget_new_labels']), r['evaluator_class']) for r in index}
            == expected_cells, 'Arm/budget/evaluator cells differ')
    energy_min = min(float(r['QM_energy_Eh']) for r in truth)
    low = {r['state_id'] for r in truth if (float(r['QM_energy_Eh'])-energy_min)*627.509474 <= 5}
    require(len(low) == 15, 'Low-energy subset differs')
    original_gp = original_fourier = np = None
    if verify_predictions:
        import numpy as np
        original_gp = load_module(folder / 'code/periodic_gp_v1.py', 'nme_original_periodic_gp')
        original_fourier = load_module(folder / 'code/fourier_k2_v1.py', 'nme_original_fourier')
    metrics = []
    train_digests = {}
    posterior_max = 0.0
    for r in index:
        fit = fits[r['fit_index']]
        pred_path = data / 'predictions' / Path(r['prediction_file']).name
        train_path = data / 'training' / Path(fit['training_manifest']).name
        require(sha(pred_path) == r['prediction_file_sha256'], 'Original prediction SHA differs')
        require(sha(train_path) == fit['training_manifest_sha256'], 'Original training SHA differs')
        training = rows(train_path)
        require(len(training) == int(fit['training_rows']) == 54+int(r['budget_new_labels']),
                'Training count differs')
        train_digests[train_path.name] = sha(train_path)
        require(not ({v['state_id'] for v in training} & set(truth_by_id)),
                'Training/validation state overlap')
        pred = rows(pred_path)
        require(len(pred) == 144 and {v['state_id'] for v in pred} == set(truth_by_id),
                'Prediction states differ')
        errors = []
        subsets = {'cis': [], 'trans': [], 'intermediate_omega': [], 'low_energy': []}
        for v in pred:
            observed = truth_by_id[v['state_id']]
            for angle in ('phi', 'psi', 'omega'):
                diff = (float(v[angle+'_target_deg']) - float(observed[angle+'_deg']) + 180) % 360 - 180
                require(abs(diff) < 1e-9, 'Query angle differs')
            error = float(v['predicted_R_kcal_mol']) - float(observed['absolute_QM_minus_M0_residual_kcal_mol'])
            errors.append(error)
            w = float(observed['omega_deg'])
            region = 'cis' if abs(w) < 1e-9 else 'trans' if abs(abs(w)-180) < 1e-9 else 'intermediate_omega'
            subsets[region].append(error)
            if v['state_id'] in low:
                subsets['low_energy'].append(error)
        calc = {'RMSE_kcal_mol': rmse(errors), 'MAE_kcal_mol': math.fsum(abs(x) for x in errors)/144,
                'MaxAbs_kcal_mol': max(abs(x) for x in errors),
                **{k+'_RMSE_kcal_mol': rmse(v) for k, v in subsets.items()}}
        archived = archived_metrics[r['fit_index']]
        for key, value in calc.items():
            close(value, archived[key])
        rec = {'arm': r['arm'], 'budget_new_labels': int(r['budget_new_labels']),
               'evaluator_class': r['evaluator_class'], **calc}
        if verify_predictions:
            q = np.array([[float(v[k+'_target_deg']) for k in ('phi', 'psi', 'omega')] for v in pred])
            if r['evaluator_class'] == 'FOURIER_K2':
                beta = np.array(json.loads(fit['fourier_coefficients_json']))
                computed = original_fourier.predict(original_fourier.design_k2(*q.T), beta)
            else:
                tq = np.array([[float(v[k+'_deg']) for k in ('phi', 'psi', 'omega')] for v in training])
                ty = np.array([float(v['absolute_QM_minus_M0_residual_kcal_mol']) for v in training])
                close(float(ty.mean()), fit['target_mean_kcal_mol'])
                computed = original_gp.posterior_mean(tq, ty, q, float(fit['sigma_f']),
                           np.array([float(fit['ell_'+k]) for k in ('phi', 'psi', 'omega')]))
            difference = float(np.max(np.abs(computed - [float(v['predicted_R_kcal_mol']) for v in pred])))
            require(difference < 2e-6, 'Frozen-parameter prediction differs')
            posterior_max = max(posterior_max, difference)
            rec['frozen_parameter_prediction_max_difference_kcal_mol'] = difference
        metrics.append(rec)
    lookup = {(r['arm'], r['evaluator_class'], r['budget_new_labels']): r['RMSE_kcal_mol'] for r in metrics}
    curves = []
    for ref in archived_aulc:
        arm, ev = ref['arm'], ref['evaluator_class']
        vals = [lookup[arm, ev, b] for b in (24, 48, 96, 192)]
        area = sum(.5*(vals[i]+vals[i+1]) for i in range(3))/3
        close(area, ref['normalized_log_budget_AULC_kcal_mol'])
        for budget, value in zip((24,48,96,192), vals):
            close(value, ref['RMSE_B%d_kcal_mol' % budget])
        curves.append({'arm': arm, 'evaluator_class': ev, 'normalized_AULC_kcal_mol': area,
                       'B192_RMSE_kcal_mol': vals[-1]})
    require(len(curves) == 6, 'Expected six learning curves')
    gates = []
    for ev in ('FOURIER_K2', 'PERIODIC_GP'):
        cc = {v['arm']: v for v in curves if v['evaluator_class'] == ev}
        beats = all(cc['GP'][key] < cc[arm][key]-1e-6
                    for arm in ('MAXIMIN', 'RANDOM')
                    for key in ('normalized_AULC_kcal_mol', 'B192_RMSE_kcal_mol'))
        gates.append(beats)
    decision = ('CROSS_MODEL_ACQUISITION_ADVANTAGE_SUPPORTED' if all(gates)
                else 'ORTHOGONAL_FOURIER_ACQUISITION_ADVANTAGE_ONLY' if gates[0]
                else 'GP_MODEL_MATCHED_ADVANTAGE_ONLY' if gates[1]
                else 'NO_ROBUST_ACQUISITION_ADVANTAGE_SUPPORTED')
    archived_decisions = {r['final_precommitted_decision'] for r in rows(data / 'claim_gate.tsv')}
    require(archived_decisions == {decision}, 'Acquisition decision differs')
    report = {'status': 'PASS', 'verified_source_files': len(manifest['files']),
              'validation_geometries': 144, 'training_manifests': len(train_digests),
              'prediction_tables': 24, 'pointwise_predictions': 3456,
              'metrics': metrics, 'learning_curves': curves, 'frozen_claim_gate': decision,
              'original_basis_and_posterior_replayed': verify_predictions,
              'max_prediction_difference_kcal_mol': posterior_max if verify_predictions else None,
              'hyperparameters_reoptimized': False, 'QM_or_MD_launched': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'metrics'}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, default=HERE)
    ap.add_argument('--out', type=Path, default=Path('acquisition_reproduced.json'))
    ap.add_argument('--verify-predictions', action='store_true')
    a = ap.parse_args()
    run(a.data_root, a.out, a.verify_predictions)
