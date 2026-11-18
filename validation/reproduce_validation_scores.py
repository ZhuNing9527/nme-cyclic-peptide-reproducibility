#!/usr/bin/env python3
"""Reproduce the frozen-model Ala and Val validation scores using bundled tables.

No electronic-structure jobs, coefficient fitting, MD, or GPU calls are made.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter='\t'))

def design(q):
    def b(t):
        phase=np.deg2rad(t)[:,None]*np.arange(1,5)
        return np.column_stack((np.ones(len(t)),np.stack((np.cos(phase),np.sin(phase)),axis=2).reshape(len(t),8)))
    p,s,w=(b(q[:,k]) for k in range(3))
    g=w[:,1:]-b(np.array([180.]))[:,1:]
    return np.column_stack((np.einsum('ni,nj->nij',p,s).reshape(len(q),81),
        np.einsum('ni,nj,nk->nijk',p,s,g).reshape(len(q),648)))

def rmse(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))

def main(data,out):
    a=data/'Ala';v=data/'Val'
    tr=read(a/'training_246.tsv');co=read(a/'frozen_coefficients.tsv')
    coeff={k:np.array([float(r[k+'_coeff_kcal_mol']) for r in co]) for k in ('pairwise','joint')}
    offset=np.mean([float(r['absolute_QM_minus_M0_residual_kcal_mol']) for r in tr])
    pred=read(a/'holdout_predictions.tsv')
    q=np.array([[float(r[k+'_deg']) for k in ('phi','psi','omega')] for r in pred])
    y=np.array([float(r['observed_R_kcal_mol']) for r in pred])
    X=design(q);errors={k:X@c+offset-y for k,c in coeff.items()}
    for k in errors:
        np.testing.assert_allclose(errors[k],[float(r[k+'_error_kcal_mol']) for r in pred],rtol=0,atol=1e-8)
    delta=rmse(errors['pairwise'])-rmse(errors['joint'])
    ix=np.random.default_rng(20261117).integers(len(y),size=(10000,len(y)))
    draws=np.sqrt(np.mean(errors['pairwise'][ix]**2,axis=1))-np.sqrt(np.mean(errors['joint'][ix]**2,axis=1))
    ci=np.quantile(draws,[.025,.975])
    expected=read(a/'holdout_metrics.tsv')[0]
    np.testing.assert_allclose([rmse(errors['pairwise']),rmse(errors['joint']),delta,*ci],
        [float(expected[k]) for k in ('pairwise_RMSE_kcal_mol','joint_RMSE_kcal_mol','delta_RMSE_kcal_mol','paired_CI95_low_kcal_mol','paired_CI95_high_kcal_mol')],rtol=0,atol=1e-8)
    result={'Ala_independent':{'n':144,'pairwise_RMSE':rmse(errors['pairwise']),'joint_RMSE':rmse(errors['joint']),
        'paired_RMSE_difference':delta,'paired_CI95':ci.tolist()},'Val':[]}
    allrows=read(v/'NMEVAL_pointwise_frozen_model_predictions.tsv')
    metrics=read(v/'NMEVAL_frozen_transfer_metrics.tsv')
    for contrast in ('context_trans_increment','common_AR_t_p180_full_correction'):
        for panel in ('primary54','alphaL18'):
            rr=[r for r in allrows if r['contrast']==contrast and r['panel']==panel and r['reference_point']=='False']
            q=np.array([[float(r[k+'_deg']) for k in ('phi','psi','omega')] for r in rr])
            ref=q.copy()
            if contrast=='context_trans_increment':ref[:,2]=180.
            else:ref[:]=[-60.,-40.,180.]
            xx=design(q)-design(ref)
            obs=np.array([float(r['observed_relative_residual_kcal_mol']) for r in rr])
            values={k:rmse(xx@c-obs) for k,c in coeff.items()}
            expected=next(r for r in metrics if (r['contrast'],r['panel'],r['subset'])==(contrast,panel,'ALL'))
            np.testing.assert_allclose([values['pairwise'],values['joint']],
                [float(expected[k+'_RMSE_kcal_mol']) for k in ('pairwise','joint')],rtol=0,atol=1e-8)
            result['Val'].append({'contrast':contrast,'panel':panel,'n':len(rr),**values})
    result.update({'status':'PASS','coefficients_refitted':False,'new_QM_or_MD':False})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=HERE/'data')
    p.add_argument('--out',type=Path,default=Path('QM_Validation_Reproduction.json'))
    a=p.parse_args();main(a.data,a.out)
