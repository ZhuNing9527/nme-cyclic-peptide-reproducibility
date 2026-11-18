#!/usr/bin/env python3
"""Matched nested Fourier comparison on the frozen NMe-Ala training and scoring data."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np


K = 4
LAMBDA = 100.0
RCOND = 1e-12
TRAIN_SHA = '48eec6767bd2258d3378e0c255e0d7b9445bb4f77782e2809332ce2b4956d39e'
SCORING_SHA = '191b9005727a179f11fc8433d7b77c2d47610ee81bdb37867105242d295f4307'
FROZEN_JOINT = 'FOURIER_K4_L100'
FROZEN_SPLINE = 'SPLINE_M4_L1e-1'
ANGLES = ('phi_deg', 'psi_deg', 'omega_deg')
ROOT = Path(__file__).resolve().parent


def check(test, description):
    if not test:
        raise ValueError(description)


def digest(data):
    return sha256(data).hexdigest()


def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig')), delimiter='\t'))


def angles(records):
    return np.array([[float(r[a]) for a in ANGLES] for r in records], dtype=float)


def key(q):
    return tuple((q + 180.0) % 360.0 - 180.0)


def basis(theta):
    phase = theta[:, None] * np.arange(1, K + 1)
    v = np.stack((np.cos(phase), np.sin(phase)), axis=2).reshape(len(theta), 2 * K)
    return np.column_stack((np.ones(len(theta)), v))


def design(angle_deg):
    phi, psi, omega = np.deg2rad(angle_deg).T
    p, s = basis(phi), basis(psi)
    g = basis(omega)[:, 1:] - basis(np.array([np.pi]))[:, 1:]
    A = np.einsum('ni,nj->nij', p, s).reshape(len(phi), -1)
    J = np.einsum('ni,nj,nk->nijk', p, s, g).reshape(len(phi), -1)
    return np.column_stack((A, J))


def penalty_and_masks():
    f = np.array([0] + [i for i in range(1, K + 1) for _ in range(2)])
    a = np.square(f[:, None]**2 + f[None, :]**2).ravel()
    j = np.square(f[:, None, None]**2 + f[None, :, None]**2 + f[None, None, 1:]**2).ravel()
    q = np.concatenate((a, j)).astype(float)
    scale = float(np.median(q[q > 0]))
    keep_j = np.broadcast_to((f[:, None, None] == 0) | (f[None, :, None] == 0), (len(f), len(f), len(f)-1)).ravel()
    pair = np.concatenate((np.ones(len(a), dtype=bool), keep_j))
    check(len(q)==729 and np.count_nonzero(pair)==217 and q[0]==0.0, 'unexpected model dimensions or intercept')
    # Normalize once in the full basis. The restricted fit uses the corresponding
    # diagonal entries unchanged, so shared coefficients retain identical penalties.
    return np.sqrt(q/scale), pair, scale


def solve(X, y, reg, chosen):
    xp = X[:, chosen]
    d = reg[chosen]
    augmented = np.vstack((xp, np.diag(np.sqrt(LAMBDA)*d)))
    coeff, _, rank, _ = np.linalg.lstsq(augmented, np.concatenate((y, np.zeros(len(d)))), rcond=RCOND)
    check(rank==len(coeff) and np.isfinite(coeff).all(), 'invalid augmented solution')
    return coeff


def measure(errors):
    return tuple(float(x) for x in (np.sqrt(np.mean(errors**2)), np.mean(np.abs(errors)), np.max(np.abs(errors))))


def write_tsv(path, fields, items):
    with path.open('w', encoding='utf-8', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(items)


def run(training, scoring, out):
    raw_train = training.read_bytes()
    check(digest(raw_train)==TRAIN_SHA, 'training table SHA256 differs')
    tr = rows(raw_train)
    qt = angles(tr)
    check(len(tr)==246 and len({r['state_id'] for r in tr})==246 and len({key(q) for q in qt})==246, 'invalid train identities')
    check(Counter(r['training_origin'] for r in tr)['IDENTIFIABILITY_SEED54_HARMONIZED']==54, 'invalid common seed count')
    check({r['arm'] for r in tr}=={'MAXIMIN'}, 'training corpus is not the frozen maximin corpus')
    y = np.array([float(r['absolute_QM_minus_M0_residual_kcal_mol']) for r in tr])
    center = float(y.mean())
    X = design(qt)
    reg, pair, scale = penalty_and_masks()
    cfull = solve(X, y-center, reg, np.ones(len(reg), dtype=bool))
    cpair = solve(X, y-center, reg, pair)
    check((design(np.column_stack((qt[:, :2], np.full(len(tr),180.0))))[:, 81:]==0).all(), 'trans gauge failed')

    # Load the frozen scoring labels only after both fits have been determined.
    scoring_bytes = scoring.read_bytes()
    check(digest(scoring_bytes)==SCORING_SHA, 'scoring table SHA256 differs')
    comparison = rows(scoring_bytes)
    full_reference = sorted((r for r in comparison if r['candidate_id']==FROZEN_JOINT), key=lambda r:int(r['validation_rank']))
    spline_reference = {r['state_id']:r for r in comparison if r['candidate_id']==FROZEN_SPLINE}
    check(len(comparison)==288 and len(full_reference)==len(spline_reference)==144, 'invalid comparison table size')
    check([int(r['validation_rank']) for r in full_reference]==list(range(1,145)), 'invalid scoring order')
    check({r['state_id'] for r in full_reference}==set(spline_reference), 'candidate scoring sets differ')
    qv = angles(full_reference)
    check(len({key(q) for q in qv})==144, 'duplicate periodic coordinates in scoring set')
    check(not ({key(q) for q in qt} & {key(q) for q in qv}), 'training/scoring angle overlap')
    check(not ({r['state_id'] for r in tr} & {r['state_id'] for r in full_reference}), 'training/scoring state overlap')
    for r in full_reference:
        s = spline_reference[r['state_id']]
        check(all(s[k]==r[k] for k in (*ANGLES,'observed_R_kcal_mol','low_energy_le_5kcal','omega_region')), 'frozen labels differ')

    Xv = design(qv)
    yf = Xv @ cfull + center
    yp = Xv[:, pair] @ cpair + center
    yo = np.array([float(r['observed_R_kcal_mol']) for r in full_reference])
    yref = np.array([float(r['predicted_R_kcal_mol']) for r in full_reference])
    ys = np.array([float(spline_reference[r['state_id']]['predicted_R_kcal_mol']) for r in full_reference])
    frozen_delta = float(np.max(np.abs(yf-yref)))
    check(frozen_delta < 1e-8, f'original full predictions not reproduced: max difference {frozen_delta}')
    check(abs(measure(yf-yo)[0]-1.9475278317702668)<1e-8, 'original full RMSE not reproduced')
    sub = {
        'All geometries': np.ones(len(yo), dtype=bool),
        'Low energy': np.array([r['low_energy_le_5kcal']=='True' for r in full_reference]),
        'cis': np.array([r['omega_region']=='CIS' for r in full_reference]),
        'trans': np.array([r['omega_region']=='TRANS' for r in full_reference]),
        'Intermediate ω': np.array([r['omega_region']=='INTERMEDIATE_OMEGA' for r in full_reference]),
    }
    check([int(k.sum()) for k in sub.values()]==[144,15,12,12,120], 'frozen stratum sizes differ')
    scores = []
    for name, mask in sub.items():
        record = {'subset': name, 'n': int(mask.sum())}
        for m, pred in (('constant', np.full(len(yo), center)), ('spline',ys), ('pairwise',yp), ('joint',yf)):
            for field, value in zip(('RMSE','MAE','MaxAbs'), measure((pred-yo)[mask])):
                record[f'{m}_{field}_kcal_mol'] = value
        full, restricted = record['joint_RMSE_kcal_mol'], record['pairwise_RMSE_kcal_mol']
        record['RMSE_reduction_pct'] = 100*(1-full/restricted)
        scores.append(record)

    out.mkdir(parents=True, exist_ok=False)
    write_tsv(out/'QM144_matched_fourier_metrics.tsv', list(scores[0]), scores)
    pointwise = []
    for i,r in enumerate(full_reference):
        pointwise.append({**{k:r[k] for k in ('validation_rank','state_id',*ANGLES,'low_energy_le_5kcal','omega_region')},
                          'observed_R_kcal_mol':yo[i], 'joint_prediction_kcal_mol':yf[i],
                          'pairwise_prediction_kcal_mol':yp[i], 'archived_joint_prediction_kcal_mol':yref[i],
                          'joint_error_kcal_mol':yf[i]-yo[i], 'pairwise_error_kcal_mol':yp[i]-yo[i]})
    write_tsv(out/'QM144_matched_fourier_predictions.tsv',list(pointwise[0]),pointwise)
    coefficients = [{'column_full_basis_0based':i,'normalized_penalty_q':float(reg[i]**2),
                     'joint_coeff_kcal_mol':float(cfull[i]),'retained_by_pairwise':bool(pair[i]),
                     'pairwise_coeff_kcal_mol':0.0} for i in range(len(reg))]
    for j, idx in enumerate(np.flatnonzero(pair)):
        coefficients[idx]['pairwise_coeff_kcal_mol'] = float(cpair[j])
    write_tsv(out/'QM246_matched_fourier_coefficients.tsv',list(coefficients[0]),coefficients)
    audit = {'training_sha256':TRAIN_SHA,'scoring_table_sha256':SCORING_SHA,
             'n_train':len(tr),'n_score':len(yo),'training_scoring_state_overlap':0,'training_scoring_periodic_angle_overlap':0,
             'K':K,'lambda':LAMBDA,'rcond':RCOND,'full_basis_penalty_median':scale,'training_R_mean_kcal_mol':center,
             'joint_coefficients':len(cfull),'pairwise_coefficients':len(cpair),'max_reproduced_joint_difference_kcal_mol':frozen_delta,
             'pairwise_restriction':'Keep full A; keep J when phi index or psi index is the constant; inherit full-basis penalty scale.',
             'scoring_designation':'Held out from coefficient fitting, but previously used for representation selection; retrospective comparison.',
             'training_RMSE_kcal_mol': {'joint':measure(X @ cfull + center - y)[0],
                                        'pairwise':measure(X[:, pair] @ cpair + center - y)[0]}}
    (out/'verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit, indent=2))
    for r in scores:
        print(f"{r['subset']:>15}  n={r['n']:3d}   pairwise={r['pairwise_RMSE_kcal_mol']:.6f}   joint={r['joint_RMSE_kcal_mol']:.6f}   reduction={r['RMSE_reduction_pct']:.2f}%")


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training', type=Path, default=ROOT.parent/'inputs/training_246.tsv')
    ap.add_argument('--scoring', type=Path, default=ROOT.parent/'inputs/scoring_144.tsv')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    run(a.training, a.scoring, a.out)
