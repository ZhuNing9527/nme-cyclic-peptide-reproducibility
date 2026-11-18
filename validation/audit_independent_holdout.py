#!/usr/bin/env python3
"""Verify archived independent QM results without fitting models or running QM/MD.

Usage: python audit_independent_holdout.py --archive NME_CLEAN_HOLDOUT_RESULTS_v1.2.tar.gz
Requires Python 3.10+ and NumPy. All archive access is read-only.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np

ARCHIVE_SHA = 'ebda5b6e86fe7dbbf86e9aa549825ce67e0e878aa2325cd77fc6bff41a667352'
PROTOCOL_SHA = '07de7124541d328c9a41b90007a3bac714404cfd432c3032f2ffe22b9882cb77'
TOPOLOGY_SHA = '9b2c68fad482b03562a4b9c956dc57586bfc73384bb3a68b019e866c1275f5e6'
READER_SHA = 'b11a8cdca1c51180b3cc80e8823867531a3cf3ab9667e49275fac9cb00be4cc5'
EH_TO_KCAL = 627.509474
SEED = 20261117
NBOOT = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def table(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t'))


def numbers(rows, key):
    return np.array([float(r[key]) for r in rows])


def angles(rows):
    return np.column_stack([numbers(rows, k) for k in ('phi_deg', 'psi_deg', 'omega_deg')])


def basis(theta):
    phase = theta[:, None] * np.arange(1, 5)
    v = np.stack((np.cos(phase), np.sin(phase)), axis=2).reshape(len(theta), 8)
    return np.column_stack((np.ones(len(theta)), v))


def design(q):
    phi, psi, omega = np.deg2rad(q).T
    p, s = basis(phi), basis(psi)
    g = basis(omega)[:, 1:] - basis(np.array([np.pi]))[:, 1:]
    a = np.einsum('ni,nj->nij', p, s).reshape(len(phi), -1)
    j = np.einsum('ni,nj,nk->nijk', p, s, g).reshape(len(phi), -1)
    return np.column_stack((a, j))


def xyz(data):
    ls = data.decode().splitlines()
    n = int(ls[0])
    rows = [s.split() for s in ls[2:2+n]]
    require(n == 25 and len(rows) == n, 'Fragment atom count differs')
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def dihedral(x, ids):
    p0, p1, p2, p3 = x[list(ids)]
    b0, b1, b2 = -(p1-p0), p2-p1, p3-p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1
    return np.rad2deg(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def bootstrap(a, b):
    idx = np.random.default_rng(SEED).integers(len(a), size=(NBOOT, len(a)))
    delta = np.sqrt(np.mean(a[idx]**2, axis=1)) - np.sqrt(np.mean(b[idx]**2, axis=1))
    return np.quantile(delta, [.025, .975])


def stratified_bootstrap(a, b, strata):
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(strata == k) for k in sorted(set(strata))]
    values = []
    for _ in range(NBOOT):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        values.append(np.sqrt(np.mean(a[ix]**2)) - np.sqrt(np.mean(b[ix]**2)))
    return np.quantile(values, [.025, .975])


def run(archive, refs, out):
    raw_archive = archive.read_bytes()
    require(sha(raw_archive) == ARCHIVE_SHA, 'Result archive SHA256 differs')
    references = json.loads((refs/'frozen_model_verification.json').read_text())
    coefficients = table((refs/'frozen_coefficients.tsv').read_bytes())
    training = table((refs/'training_246.tsv').read_bytes())
    old_scoring = table((refs/'prior_scoring_144.tsv').read_bytes())
    require(len(coefficients) == 729 and len(training) == 246 and len(old_scoring) == 144, 'Reference dimensions differ')
    require(sha((refs/'training_246.tsv').read_bytes()) == references['training_sha256'], 'Training bytes changed')
    center = references['training_R_mean_kcal_mol']
    require(abs(numbers(training, 'absolute_QM_minus_M0_residual_kcal_mol').mean()-center) < 1e-9, 'Training offset differs')
    cfull = numbers(coefficients, 'joint_coeff_kcal_mol')
    cpair = numbers(coefficients, 'pairwise_coeff_kcal_mol')

    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode='r:gz') as t:
        members = t.getmembers()
        require(len({m.name for m in members}) == len(members), 'Duplicate archive members')
        for m in members:
            p = PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and not m.issym() and not m.islnk(), 'Unsafe archive member')
        # One sequential read avoids repeatedly decompressing a large gzip stream.
        content = {m.name:t.extractfile(m).read() for m in members if m.isfile()}
        def read(name):
            return content[name]
        contract = json.loads(read('RUN_CONTRACT.json'))
        summary = json.loads(read('scored/holdout_summary.json'))
        status = json.loads(read('STATUS.json'))
        require(contract['design_protocol_sha256'] == summary['protocol_sha256'] == PROTOCOL_SHA, 'Frozen protocol differs')
        require(contract['code_sha256']['qm_geometry.py'] == READER_SHA, 'Parser hotfix SHA differs')
        require(sha(read('formal_resp.prmtop')) == contract['formal_resp_prmtop_sha256'] == TOPOLOGY_SHA, 'Topology bytes differ')
        require(status['status'] == 'COMPLETE' and status['completed_main_slots'] == 144 and status['failed_geometries'] == 0, 'Incomplete panel')
        require(contract['MD'] is False and contract['GPU'] is False, 'Unexpected compute mode')
        attempts = table(read('attempt_log.tsv'))
        rows = table(read('scored/holdout_predictions.tsv'))
        qm = table(read('qm_results.tsv'))
        metrics = table(read('scored/holdout_metrics.tsv'))
        require(len(attempts) == len(rows) == len(qm) == 144, 'Panel counts differ')
        ids = [f'HO{i:03d}' for i in range(1, 145)]
        require([r['slot_id'] for r in rows] == ids, 'Unexpected main slots')
        require(all(r['slot_id'] == r['geometry_id'] and r['attempt_number'] == '1' and r['status'] == 'SUCCESS' for r in attempts), 'Unexpected replacement or failed attempt')
        require(sha(read('qm_results.tsv')) == summary['input_hashes']['results'], 'QM score input differs')
        require(sha(read('attempt_log.tsv')) == summary['input_hashes']['attempt_log'], 'Attempt score input differs')
        qm_by_id = {r['holdout_id']:r for r in qm}
        output_hash_checks = 0
        max_raw_energy_error = max_m0_error = max_residual_error = max_restart_error = max_angle_error = 0.0
        wall_seconds = []
        min_chirality_product = float('inf')
        for row in rows:
            sid = row['geometry_id']
            prefix = f'jobs/{sid}/'
            result = json.loads(read(prefix+'RESULT.json'))
            require(result['status'] == 'SUCCESS' and not result['failure_code'], 'Unexpected failed result')
            for name, expected in result['output_sha256'].items():
                require(sha(read(prefix+name)) == expected, f'Output checksum differs: {sid}/{name}')
                output_hash_checks += 1
            inp = read(prefix+'job.inp').decode()
            require('r2SCAN-3c TightSCF TightOpt CPCM(Water)' in inp and 'nprocs 8' in inp and 'MaxIter 300' in inp, 'QM settings differ')
            constraint = re.findall(r'\{\s*D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+\d.]+)\s+C\s*\}', inp)
            require(len(constraint) == 3, 'Constraint count differs')
            by_atoms = {tuple(map(int,c[:4])):c for c in constraint}
            atom_quads = [(1,3,5,7),(3,5,7,9),(0,1,3,5)]
            require(set(by_atoms) == set(atom_quads), 'Constrained atom indices differ')
            constraint = [by_atoms[k] for k in atom_quads]
            expected_q = np.array([float(row[k]) for k in ('phi_deg','psi_deg','omega_deg')])
            target_delta = (np.array([float(x[4]) for x in constraint])-expected_q+180)%360-180
            require(np.max(np.abs(target_delta)) < 1e-8, 'Native angles differ from scored angles')
            output = read(prefix+'job.out').decode(errors='replace')
            require('THE OPTIMIZATION HAS CONVERGED' in output and 'ORCA TERMINATED NORMALLY' in output, 'Unconverged raw ORCA output')
            energies = re.findall(r'FINAL SINGLE POINT ENERGY\s+([-+\d.]+)', output)
            energy = float(energies[-1])
            max_raw_energy_error = max(max_raw_energy_error, abs(energy-result['qm_energy_Eh']))
            m0_text = read(prefix+'m0.out').decode()
            component_sum = 0.0
            for name in ('BOND','ANGLE','DIHED','VDWAALS','EEL','HBOND','1-4 VDW','1-4 EEL','RESTRAINT'):
                vals = re.findall(r'(?<![\w-])(?<!1-4 )'+re.escape(name)+r'\s*=\s*([-+\d.Ee]+)',m0_text)
                require(bool(vals), f'Missing M0 component {name}')
                component_sum += float(vals[-1])
            max_m0_error = max(max_m0_error, abs(component_sum-result['M0_component_sum_kcal_mol']))
            residual = energy*EH_TO_KCAL-component_sum
            max_residual_error = max(max_residual_error, abs(residual-float(row['observed_R_kcal_mol'])), abs(residual-float(qm_by_id[sid]['absolute_QM_minus_M0_residual_kcal_mol'])))
            elements, x = xyz(read(prefix+'job.xyz'))
            start_elements, start = xyz(read(prefix+'start.xyz'))
            require(elements == start_elements, 'Atom order changed')
            actual = np.array([dihedral(x,tuple(map(int,c[:4]))) for c in constraint])
            angle_error = float(np.max(np.abs((actual-expected_q+180)%360-180)))
            max_angle_error = max(max_angle_error, angle_error)
            require(angle_error <= .1, 'Angle constraint failed')
            def chirality(x):
                return float(np.dot(x[3]-x[5],np.cross(x[7]-x[5], x[6]-x[5])))
            product = chirality(start)*chirality(x)
            min_chirality_product = min(min_chirality_product,product)
            require(product > 0, 'Alpha chirality inverted')
            rst = read(prefix+'exact_qm.rst7').decode().splitlines()
            rx = np.array([float(v) for line in rst[2:] for v in line.split()]).reshape(25,3)
            max_restart_error = max(max_restart_error,float(np.max(np.abs(rx-x))))
            wall_seconds.append(result['wall_seconds'])
        require(max_raw_energy_error < 1e-10 and max_m0_error < 1e-8 and max_residual_error < 1e-8, 'Raw energies do not reproduce score labels')
        require(max_restart_error < 5.1e-8, 'M0 coordinates differ from QM coordinates beyond saved precision')
        q = angles(rows)
        old_q = np.vstack((angles(training), angles(old_scoring)))
        distances = np.linalg.norm((q[:,None,:]-old_q[None,:,:]+180)%360-180,axis=2)
        require(float(distances.min()) > 1e-6, 'Training or old scoring angle overlap')
        require(len({tuple((v+180)%360-180) for v in q}) == 144, 'Duplicate periodic holdout angles')
        x = design(q)
        pred_full = x@cfull+center
        pred_pair = x@cpair+center
        prediction_error = max(float(np.max(np.abs(pred_full-numbers(rows,'joint_prediction_kcal_mol')))),float(np.max(np.abs(pred_pair-numbers(rows,'pairwise_prediction_kcal_mol')))))
        require(prediction_error < 1e-8, 'Frozen coefficient predictions differ')
        y = numbers(rows,'observed_R_kcal_mol')
        a, b = pred_pair-y, pred_full-y
        qm_energy = np.array([float(qm_by_id[r['geometry_id']]['qm_energy_kcal_mol']) for r in rows])
        low = qm_energy-qm_energy.min() <= 5
        strata = np.array([r['stratum'] for r in rows])
        masks = {'All geometries':np.ones(len(rows),bool),'cis':strata=='CIS','trans':strata=='TRANS','Intermediate omega':strata=='INTERMEDIATE','Low energy (<=5 kcal/mol)':low}
        max_metric_error = max_ci_error = 0.0
        for r in metrics:
            mask = masks[r['subset']]
            require(int(mask.sum()) == int(r['n']), 'Subset size differs')
            aa, bb = a[mask], b[mask]
            rp, rj = np.sqrt(np.mean(aa**2)),np.sqrt(np.mean(bb**2))
            values = {'pairwise_RMSE_kcal_mol':rp,'joint_RMSE_kcal_mol':rj,'delta_RMSE_kcal_mol':rp-rj,'RMSE_reduction_pct':100*(rp-rj)/rp,'pairwise_MaxAbs_kcal_mol':np.max(np.abs(aa)),'joint_MaxAbs_kcal_mol':np.max(np.abs(bb))}
            for key,val in values.items():
                max_metric_error = max(max_metric_error,abs(float(r[key])-val))
            require(int(r['n_joint_closer']) == int(np.sum(np.abs(bb)<np.abs(aa))), 'Closer-prediction count differs')
            ci = bootstrap(aa,bb)
            archived_ci = [float(r['paired_CI95_low_kcal_mol']),float(r['paired_CI95_high_kcal_mol'])]
            max_ci_error = max(max_ci_error,float(np.max(np.abs(ci-archived_ci))))
        require(max_metric_error < 1e-8 and max_ci_error < 1e-8, 'Metrics or paired confidence intervals differ')
        primary_stratified_replay = stratified_bootstrap(a,b,strata)
        archived_stratified = np.array([summary['primary']['omega_stratified_CI95_low_kcal_mol'],summary['primary']['omega_stratified_CI95_high_kcal_mol']])
        require(np.max(np.abs(primary_stratified_replay-archived_stratified)) < .01, 'Independent stratified bootstrap is inconsistent')
        report = {
            'archive_sha256':ARCHIVE_SHA,'protocol_sha256':PROTOCOL_SHA,'status':'PASS',
            'completed_main_slots':144,'first_attempt_successes':144,'failed_geometries':0,'reserves_used':0,
            'raw_output_sha256_checks':output_hash_checks,'raw_ORCA_convergence_checks':144,
            'max_raw_QM_energy_difference_Eh':max_raw_energy_error,'max_raw_M0_difference_kcal_mol':max_m0_error,
            'max_raw_residual_difference_kcal_mol':max_residual_error,'max_angle_constraint_error_deg':max_angle_error,
            'alpha_chirality_preserved_all':True,'max_QM_to_M0_coordinate_difference_A':max_restart_error,
            'frozen_coefficient_prediction_max_difference_kcal_mol':prediction_error,'models_refit':False,'test_labels_recentered':False,
            'training_246_and_prior_scoring_144_exact_periodic_overlaps':0,
            'minimum_circular_3D_angular_distance_to_these_390_points_deg':float(distances.min()),
            'max_recomputed_metric_difference':max_metric_error,'max_recomputed_paired_CI_difference_kcal_mol':max_ci_error,
            'bootstrap_seed':SEED,'bootstrap_resamples':NBOOT,
            'primary_omega_stratified_CI_archived_kcal_mol':archived_stratified.tolist(),
            'primary_omega_stratified_CI_independent_stream_kcal_mol':primary_stratified_replay.tolist(),
            'stratified_replay_note':'Same strata, paired resampling and seed; loop-based random draw allocation differs from the archived scorer. Both intervals exclude zero.',
            'job_wall_hours_sum':sum(wall_seconds)/3600,'job_mean_wall_minutes':float(np.mean(wall_seconds)/60),
            'low_energy_n':int(low.sum()),'main_metrics':summary['primary'],
            'review_scope':'Raw outputs, coordinates, frozen coefficients and available training/prior-scoring tables. The historical 685-point exclusion manifest is not included in this result archive and is not independently re-audited here.',
            'new_QM_executed_by_review':False,'MD_executed_by_review':False
        }
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--reference-data',type=Path,default=Path(__file__).resolve().parent/'source_data')
    parser.add_argument('--out',type=Path,default=Path('Independent_Holdout_Verification.json'))
    args = parser.parse_args()
    run(args.archive,args.reference_data,args.out)
