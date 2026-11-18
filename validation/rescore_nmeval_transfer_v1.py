#!/usr/bin/env python3
"""Audit and rescore existing NMe-Val QM data with frozen NMe-Ala models.

No parameter fitting, energy-offset fitting, QM, MD or GPU work is performed.
Requires NumPy. Raw data remain in the separately supplied original archive.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np

ARCHIVE_SHA='2444070f1762639ad84fcdee853b3527e4dc4cacce854dea3d3aa377c5fda3c4'
COEFF_SHA='45ea3fae8a60a01f490cb95baeeb42fc9510e1d1d31004d266f0b415b8d9e29e'
PROTOCOL_SHA='8f7b500f98b13b27ddac4fb536b0c3883c298747820465c432ea4e310e5de17a'
TOPO_SHA='d05ec43286f24efdc52f003595d47a355f59080537971a0aed6c819dc491851e'
PREFIX='source/07_nmeala_separability_pilot/11_nmeval_transfer/'
SERVER_ROOT='/mnt/lab/zn/mem-perm/P0_starter_v1/'
EH=627.509474
SEED=20261118
NBOOT=10000


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(b):return hashlib.sha256(b).hexdigest()


def tsv(b):return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig')),delimiter='\t'))


def write_table(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)


def basis(t):
    ph=t[:,None]*np.arange(1,5)
    return np.column_stack((np.ones(len(t)),np.stack((np.cos(ph),np.sin(ph)),axis=2).reshape(len(t),8)))


def design(q):
    phi,psi,omega=np.deg2rad(q).T;p,s=basis(phi),basis(psi)
    g=basis(omega)[:,1:]-basis(np.array([np.pi]))[:,1:]
    return np.column_stack((np.einsum('ni,nj->nij',p,s).reshape(len(q),81),np.einsum('ni,nj,nk->nijk',p,s,g).reshape(len(q),648)))


def parse_topology(raw):
    blocks={}
    for part in raw.decode().split('%FLAG ')[1:]:
        ls=part.splitlines();name=ls[0].strip();fmt=ls[1]
        m=re.search(r'\(\d+([a-zA-Z])(\d+)(?:\.\d+)?\)',fmt)
        require(m is not None,'Unknown topology field format')
        typ,width=m.group(1).upper(),int(m.group(2))
        vs=[line[i:i+width].strip() for line in ls[2:] for i in range(0,len(line),width)]
        vs=[v for v in vs if v]
        blocks[name]=vs if typ=='A' else [int(v) if typ=='I' else float(v.replace('D','E')) for v in vs]
    n=blocks['POINTERS'][0];require(n==31,'Wrong capped fragment')
    elements=[{1:'H',6:'C',7:'N',8:'O'}[int(v)] for v in blocks['ATOMIC_NUMBER']]
    require(dict(Counter(elements))=={'C':9,'H':18,'N':2,'O':2},'Wrong molecular formula')
    bonds=[]
    for k in ('BONDS_INC_HYDROGEN','BONDS_WITHOUT_HYDROGEN'):
        v=blocks[k];bonds.extend((int(v[i])//3,int(v[i+1])//3) for i in range(0,len(v),3))
    require(len(bonds)==30,'Wrong bond count')
    graph={i:set() for i in range(n)}
    for a,b in bonds:graph[a].add(b);graph[b].add(a)
    candidates=[]
    for a in graph:
        if elements[a]!='C':continue
        ns=[i for i in graph[a] if elements[i]=='N']
        cs=[i for i in graph[a] if elements[i]=='C' and any(elements[j]=='O' for j in graph[i])]
        bs=[i for i in graph[a] if elements[i]=='C' and i not in cs]
        if len(ns)==len(cs)==len(bs)==1 and len([i for i in graph[bs[0]] if elements[i]=='C' and i!=a])==2:
            candidates.append((a,ns[0],cs[0],bs[0]))
    require(len(candidates)==1,'Nonunique Val backbone binding')
    a,n,c,b=candidates[0]
    cp=[i for i in graph[n] if elements[i]=='C' and any(elements[j]=='O' for j in graph[i])]
    require(len(cp)==1,'Nonunique preceding carbonyl')
    prev=cp[0]
    cap=[i for i in graph[prev] if elements[i]=='C']
    nxt=[i for i in graph[c] if elements[i]=='N']
    gamma=sorted([i for i in graph[b] if elements[i]=='C' and i!=a],key=lambda i:blocks['ATOM_NAME'][i])
    require(len(cap)==len(nxt)==1,'Wrong capping graph')
    quads={'phi':(prev,n,a,c),'psi':(n,a,c,nxt[0]),'omega':(cap[0],prev,n,a),'chi1':(n,a,b,gamma[0])}
    return blocks,elements,bonds,quads,(a,n,c,b)


def xyz(raw):
    ls=raw.decode().splitlines();require(int(ls[0])==31,'QM atom count differs')
    r=[s.split() for s in ls[2:33]]
    return [s[0] for s in r],np.array([[float(v) for v in s[1:4]] for s in r])


def dihedral(x,ids):
    p0,p1,p2,p3=x[list(ids)];b0=-(p1-p0);b1=p2-p1;b2=p3-p2;b1/=np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1;w=b2-np.dot(b2,b1)*b1
    return float(np.rad2deg(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))


def cluster_ci(rows):
    groups=sorted({r['context'] for r in rows})
    counts=np.array([sum(r['context']==g for r in rows) for g in groups])
    sums={k:np.array([sum(r[k+'_error_kcal_mol']**2 for r in rows if r['context']==g) for g in groups]) for k in ('pairwise','joint')}
    idx=np.random.default_rng(SEED).integers(len(groups),size=(NBOOT,len(groups)))
    denom=counts[idx].sum(axis=1)
    d=np.sqrt(sums['pairwise'][idx].sum(axis=1)/denom)-np.sqrt(sums['joint'][idx].sum(axis=1)/denom)
    return np.quantile(d,[.025,.975]),len(groups)


def metrics(rows,contrast,panel,subset):
    rms={k:float(np.sqrt(np.mean([r[k+'_error_kcal_mol']**2 for r in rows]))) for k in ('M0','pairwise','joint')}
    ci,nc=cluster_ci(rows)
    return {'contrast':contrast,'panel':panel,'subset':subset,'n_nonreference':len(rows),'n_context_clusters':nc,
        **{k+'_RMSE_kcal_mol':v for k,v in rms.items()},'pairwise_minus_joint_RMSE_kcal_mol':rms['pairwise']-rms['joint'],
        'relative_RMSE_reduction_pct':100*(rms['pairwise']-rms['joint'])/rms['pairwise'],
        'paired_cluster_CI95_low_kcal_mol':float(ci[0]),'paired_cluster_CI95_high_kcal_mol':float(ci[1]),
        'joint_closer_count':sum(abs(r['joint_error_kcal_mol'])<abs(r['pairwise_error_kcal_mol']) for r in rows)}


def rescore(archive,coefficients,protocol,out):
    raw=archive.read_bytes();require(sha(raw)==ARCHIVE_SHA,'Source archive differs')
    require(sha(coefficients.read_bytes())==COEFF_SHA,'Frozen coefficients differ')
    require(sha(protocol.read_bytes())==PROTOCOL_SHA,'Review protocol differs')
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        ms=t.getmembers();require(len({m.name for m in ms})==len(ms),'Duplicate archive names')
        for m in ms:require(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts,'Unsafe archive')
        data={m.name:t.extractfile(m).read() for m in ms}
    manifest=json.loads(data['SOURCE_MANIFEST.json'])
    for r in manifest:require(sha(data['source/'+r['path']])==r['sha256'],'Archived source hash differs: '+r['path'])
    get=lambda name:data[PREFIX+name]
    top=get('05_formal_m0/03_tleap_topology/nmeval_resp_v1.prmtop');require(sha(top)==TOPO_SHA,'Formal topology changed')
    blocks,elements,bonds,quads,chiral_ids=parse_topology(top)
    charge=tsv(get('04_formal_resp_fit/NMEVAL_FORMAL_RESP_CHARGE_SET_v1.tsv'))
    require([r['atom_name'] for r in charge]==blocks['ATOM_NAME'],'Charge atom order differs')
    dq=float(np.max(np.abs(np.array(blocks['CHARGE'])/18.2223-np.array([float(r['resp_charge_e']) for r in charge]))))
    require(dq<1e-8,'Formal RESP/topology charge mismatch')
    freeze_files=['05_formal_m0/NMEVAL_FORMAL_M0_v1.sha256','07_primary_transfer_54state/03_primary_transfer_analysis/NMEVAL_PRIMARY_TRANSFER_DECISION_FREEZE_v1.sha256','08_alphal_prospective_holdout_18state/04_prospective_analysis/NMEVAL_ALPHAL_PROSPECTIVE_DECISION_FREEZE_v1.sha256']
    nfreeze=0;missing_freeze=[]
    for name in freeze_files:
        for line in get(name).decode().splitlines():
            h,p=line.split(None,1);key='source/'+p.strip().removeprefix(SERVER_ROOT)
            if key not in data:missing_freeze.append(p);continue
            require(sha(data[key])==h,'Historical final freeze mismatch: '+p);nfreeze+=1
    cf=tsv(coefficients.read_bytes());require(len(cf)==729,'Wrong coefficient count')
    cs={k:np.array([float(r[k+'_coeff_kcal_mol']) for r in cf]) for k in ('pairwise','joint')}
    config=[('primary54','07_primary_transfer_54state/02_primary_m0_same_geometry/NMEVAL_PRIMARY_54STATE_SAME_GEOMETRY_M0_RESULTS_v1.tsv','07_primary_transfer_54state/01_primary_qm/jobs/','07_primary_transfer_54state/02_primary_m0_same_geometry/runs/'),
        ('alphaL18','08_alphal_prospective_holdout_18state/03_same_geometry_m0/NMEVAL_ALPHAL_18STATE_SAME_GEOMETRY_M0_RESULTS_v1.tsv','08_alphal_prospective_holdout_18state/02_prospective_qm/jobs/','08_alphal_prospective_holdout_18state/03_same_geometry_m0/runs/')]
    carrier_raw=get('05_formal_m0/03_tleap_topology/nmeval_resp_v1.inpcrd').decode().splitlines()
    carrier=np.array([float(v) for line in carrier_raw[2:] for v in line.split()]).reshape(31,3)
    def chirality(x):
        a,n,c,b=chiral_ids;return float(np.dot(x[n]-x[a],np.cross(x[c]-x[a],x[b]-x[a])))
    raw_error=coord_error=constraint_error=bond_max=component_error=0.0
    rows=[];original_rows=[]
    for panel,tab,qroot,mroot in config:
        rr=tsv(get(tab));require(len(rr)==(54 if panel=='primary54' else 18),'Wrong original panel size')
        for r in rr:
            require(r['status']=='PASS' and int(r['pmemd_exitcode'])==0,'Historical execution failed')
            sid=r['state_id'];original_rows.append(r)
            raw_xyz=get(qroot+sid+'/job.xyz');require(sha(raw_xyz)==r['qm_xyz_sha256'],'QM geometry source hash differs')
            els,x=xyz(raw_xyz);require(els==elements,'QM atom order differs')
            require(chirality(x)*chirality(carrier)>0,'Alpha chirality differs from carrier')
            observed={k:dihedral(x,q) for k,q in quads.items()}
            target={k:float(r[k+'_target_deg']) for k in quads}
            ce=max(abs((observed[k]-target[k]+180)%360-180) for k in quads)
            constraint_error=max(constraint_error,ce);require(ce<.1,'QM dihedral constraint differs')
            bond_max=max(bond_max,max(float(np.linalg.norm(x[a]-x[b])) for a,b in bonds));require(bond_max<1.9,'Formal bond broken')
            inp=get(qroot+sid+'/job.inp').decode()
            require('r2SCAN-3c TightSCF TightOpt CPCM(Water)' in inp,'QM level differs')
            native=re.findall(r'\{\s*D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+\d.]+)\s+C\s*\}',inp)
            nd={tuple(map(int,r[:4])):float(r[4]) for r in native}
            require(set(nd)==set(quads.values()),'Native constrained atom indices differ')
            require(all(abs((nd[quads[k]]-target[k]+180)%360-180)<1e-7 for k in quads),'Native targets differ')
            output=get(qroot+sid+'/job.out').decode(errors='replace')
            require('THE OPTIMIZATION HAS CONVERGED' in output and 'ORCA TERMINATED NORMALLY' in output,'Raw QM convergence failed')
            energy=float(re.findall(r'FINAL SINGLE POINT ENERGY\s+([-+\d.]+)',output)[-1])
            require(abs(energy-float(r['qm_energy_Eh']))<1e-10,'Raw QM energy differs')
            m0=get(mroot+sid+'/m0.out').decode();final=m0.split('FINAL RESULTS')[-1]
            total=float(re.search(r'^\s*\d+\s+([-+\d.Ee]+)\s+[-+\d.Ee]+\s+[-+\d.Ee]+',final,re.M)[1])
            raw_error=max(raw_error,abs(total-float(r['m0_total_kcal_mol'])));require(raw_error<1e-8,'Raw M0 total differs')
            components=0.0
            for k in ('BOND','ANGLE','DIHED','VDWAALS','EEL','HBOND','1-4 VDW','1-4 EEL','RESTRAINT'):
                components+=float(re.search(r'(?<![\w-])(?<!1-4 )'+re.escape(k)+r'\s*=\s*([-+\d.Ee]+)',final)[1])
            component_error=max(component_error,abs(components-float(r['m0_component_sum_kcal_mol'])));require(component_error<1e-8,'M0 component sum differs')
            rin=get(mroot+sid+'/single_point.in').decode();require(re.search(r'maxcyc\s*=\s*0\b',rin) is not None,'M0 input includes minimization')
            rst=get(mroot+sid+'/qm_geometry.inpcrd').decode().splitlines();rx=np.array([float(v) for line in rst[2:] for v in line.split()]).reshape(31,3)
            coord_error=max(coord_error,float(np.max(np.abs(rx-x))));require(coord_error<1e-6,'M0 coordinates differ from QM')
            rows.append({'state_id':sid,'panel':panel,'backbone':r.get('backbone_basin',r.get('backbone')),'chi1_stratum':r['chi1_stratum'],
                **{k+'_deg':target[k] for k in quads},**{k+'_achieved_deg':observed[k] for k in quads},
                'qm_energy_Eh':energy,'M0_total_kcal_mol':total,'M0_component_sum_kcal_mol':components,
                'absolute_QM_minus_M0_kcal_mol':energy*EH-total,'absolute_QM_minus_M0_components_kcal_mol':energy*EH-components})
    require(len(rows)==72 and len({r['state_id'] for r in rows})==72,'Repeated or missing original states')
    nominal=np.array([[r[k+'_deg'] for k in ('phi','psi','omega')] for r in rows])
    achieved=np.array([[r[k+'_achieved_deg'] for k in ('phi','psi','omega')] for r in rows])
    for k,coef in cs.items():
        v=design(nominal)@coef;va=design(achieved)@coef
        for r,y,ya in zip(rows,v,va):r[k+'_V_kcal_mol']=float(y);r[k+'_V_achieved_kcal_mol']=float(ya)
    refs={(r['backbone'],r['chi1_stratum']):r for r in rows if r['omega_deg']==180.}
    require(len(refs)==12,'Missing original trans references')
    common=next(r for r in rows if r['state_id']=='AR_t_p180')
    pointwise=[]
    for contrast in ('context_trans_increment','common_AR_t_p180_full_correction'):
        for r in rows:
            ref=refs[r['backbone'],r['chi1_stratum']] if contrast=='context_trans_increment' else common
            y=r['absolute_QM_minus_M0_kcal_mol']-ref['absolute_QM_minus_M0_kcal_mol']
            yc=r['absolute_QM_minus_M0_components_kcal_mol']-ref['absolute_QM_minus_M0_components_kcal_mol']
            rec={'contrast':contrast,**r,'reference_state_id':ref['state_id'],'reference_point':r['state_id']==ref['state_id'],
                'context':r['backbone']+'|'+r['chi1_stratum'],'observed_relative_residual_kcal_mol':y,
                'observed_relative_residual_component_sum_kcal_mol':yc,'M0_error_kcal_mol':-y}
            for k in cs:
                pred=r[k+'_V_kcal_mol']-ref[k+'_V_kcal_mol'];pred_a=r[k+'_V_achieved_kcal_mol']-ref[k+'_V_achieved_kcal_mol']
                rec[k+'_relative_prediction_kcal_mol']=pred;rec[k+'_error_kcal_mol']=pred-y
                rec[k+'_achieved_angle_error_kcal_mol']=pred_a-y
                rec[k+'_component_sum_error_kcal_mol']=pred-yc
            pointwise.append(rec)
    scores=[]
    for contrast in ('context_trans_increment','common_AR_t_p180_full_correction'):
        for panel in ('primary54','alphaL18'):
            selected=[r for r in pointwise if r['contrast']==contrast and r['panel']==panel and not r['reference_point']]
            scores.append(metrics(selected,contrast,panel,'ALL'))
            for chi in ('g-','g+','trans'):scores.append(metrics([r for r in selected if r['chi1_stratum']==chi],contrast,panel,chi))
            for basin in sorted({r['backbone'] for r in selected}):scores.append(metrics([r for r in selected if r['backbone']==basin],contrast,panel,'backbone:'+basin))
    sensitivities=[]
    for s in [s for s in scores if s['subset']=='ALL']:
        selected=[r for r in pointwise if r['contrast']==s['contrast'] and r['panel']==s['panel'] and not r['reference_point']]
        for alternative in ('achieved_angle','component_sum'):
            val={k:float(np.sqrt(np.mean([r[k+'_'+alternative+'_error_kcal_mol']**2 for r in selected]))) for k in cs}
            sensitivities.append({'contrast':s['contrast'],'panel':s['panel'],'alternative':alternative,**val,
                'max_RMSE_shift_kcal_mol':max(abs(val[k]-s[k+'_RMSE_kcal_mol']) for k in cs)})
    require(max(s['max_RMSE_shift_kcal_mol'] for s in sensitivities)<.001,'Nominal-angle/energy-precision sensitivity materially differs')
    report={'status':'PASS','source_archive_sha256':ARCHIVE_SHA,'coefficient_sha256':COEFF_SHA,'protocol_sha256':PROTOCOL_SHA,
        'verified_archived_source_files':len(manifest),'historical_final_freeze_entries_verified':nfreeze,'missing_historical_freeze_targets':missing_freeze,
        'topology_sha256':TOPO_SHA,'formal_RESP_charge_max_difference_e':dq,'QM_and_M0_jobs_verified':72,
        'maximum_four_dihedral_constraint_error_deg':constraint_error,'maximum_bond_length_A':bond_max,
        'max_M0_restart_QM_coordinate_difference_A':coord_error,'raw_M0_total_table_max_difference_kcal_mol':raw_error,
        'raw_M0_component_table_max_difference_kcal_mol':component_error,'alpha_chirality_preserved':True,
        'bound_torsion_indices_0based':{k:list(v) for k,v in quads.items()},'new_QM':False,'new_MD':False,'refit':False,
        'new_test_energy_offset_fitted':False,'assessment_designation':'Retrospective transfer of frozen coefficients to historical NMe-Val data',
        'context_trans_contrast_scope':'Amide increment J; nominal-angle A term cancels within each context',
        'common_reference_scope':'Full correction A+J, referenced to one predetermined physical Val geometry; reference uncertainty held fixed',
        'bootstrap_resamples':NBOOT,'bootstrap_seed':SEED,'bootstrap_unit':'Backbone x chi1 context; original omega values remain together; conditional on this fixed small design',
        'main_metrics':[s for s in scores if s['subset']=='ALL'],'precision_sensitivities':sensitivities}
    out.mkdir(parents=True,exist_ok=True)
    write_table(out/'NMEVAL_pointwise_frozen_model_predictions.tsv',pointwise)
    write_table(out/'NMEVAL_frozen_transfer_metrics.tsv',scores)
    write_table(out/'NMEVAL_precision_sensitivities.tsv',sensitivities)
    (out/'NMEVAL_TRANSFER_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'SOURCE_MANIFEST.json').write_bytes(data['SOURCE_MANIFEST.json'])
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--coefficients',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();rescore(a.archive,a.coefficients,a.protocol,a.out)
