#!/usr/bin/env python3
"""Audit exact native models and paired starts for the three peptide systems.

Uses XML and NumPy only; no OpenMM import, Context, GPU or dynamics. Reuses
the existing frozen graph, torsion and serialized-force arithmetic routines.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import audit_tri_nme as core
import graph_binding_v1 as graph
import frozen_math_v1 as obs

ROOT=Path(__file__).resolve().parents[1]

def require(ok,message):
    if not bool(ok):raise ValueError(message)

def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

def independent_energy(angles,signs,coef):
    def basis(x):
        z=np.asarray(x)[...,None]*np.arange(1,5)
        v=np.stack((np.cos(z),np.sin(z)),axis=-1).reshape(*z.shape[:-1],8)
        return np.concatenate((np.ones((*z.shape[:-1],1)),v),axis=-1)
    signed=np.deg2rad(angles*np.asarray(signs)[None,:,None])
    p,s,w=signed.transpose(2,0,1)
    p,s=basis(p),basis(s)
    w=basis(w)[...,1:]-basis(np.pi)[1:]
    return 4.184*(np.einsum('...i,...j,ij->...',p,s,coef[:81].reshape(9,9))+
        np.einsum('...i,...j,...k,ijk->...',p,s,w,coef[81:].reshape(9,9,8)))

def main(out):
    index=json.loads((ROOT/'inputs/ALL_SYSTEM_INPUTS_v1.json').read_text())
    coeff=np.array([float(r['joint_coeff_kcal_mol']) for r in table(ROOT/'expected/QM246_matched_fourier_coefficients.tsv')])
    require(len(coeff)==729,'Coefficient count')
    reports=[];contracts_checked=0
    for spec in index['systems']:
        paths={}
        for key,item in spec['files'].items():
            path=ROOT/item['path']
            require(hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],'Input hash: '+item['path'])
            paths[key]=path
        geometry=json.loads(paths['geometry'].read_text())
        build=json.loads(paths['build_report'].read_text())
        sites=spec['methylated_residues'];nsite=len(sites)
        nsolute=spec['solute_particles'];ntotal=spec['total_particles']
        require(build['solute_atoms']==nsolute and build['total_atoms']==ntotal,'Build counts')
        m0,full=(ET.parse(paths[k]).getroot() for k in ('M0','Full'))
        require(len(m0.find('Forces'))==5 and len(full.find('Forces'))==6,'Physical arm force definitions')
        require(core.noncorrected_forces(m0)==core.noncorrected_forces(full),'Baseline forces changed')
        for tag in ('Particles','Constraints','PeriodicBoxVectors'):
            require(core.serialize(m0.find(tag))==core.serialize(full.find(tag)),'Base system setting changed: '+tag)
        grouped={}
        for bond in core.correction_element(full).findall('Bonds/Bond'):
            key=tuple(int(bond.get('p%d'%j)) for j in range(1,7))
            grouped.setdefault(key,[]).append([float(bond.get('param%d'%j)) for j in range(1,15)])
        terms=[]
        for residue in sites:
            g=geometry[residue-1];key=tuple(g['omega_atoms']+g['psi_atoms'][2:])
            require(key[1:5]==tuple(g['phi_atoms']) and key[2:6]==tuple(g['psi_atoms']),'Torsion atom order')
            terms.append(grouped.pop(key))
        require(not grouped,'Unexpected correction sites')
        terms=np.asarray(terms)
        require(terms.shape==(nsite,729,14),'Correction term count')
        signs=np.array([-1 if i==1 else 1 for i in sites])
        require(np.array_equal(terms[:,:,:3],np.broadcast_to(signs[:,None,None],(nsite,729,3))),
                'D/L angle signs')
        basis=core._basis_spec()
        a=[tuple(p)+tuple(s)+(0,0,0,1) for p in basis for s in basis]
        j=[tuple(p)+tuple(s)+(k,c,sn,-c*(-1)**k) for p in basis for s in basis for k,c,sn in basis[1:]]
        require(np.array_equal(terms[:,:,4:],np.broadcast_to(np.array(a+j),(nsite,729,10))),
                'Fourier basis or trans gauge differs')
        coeff_error=float(np.max(np.abs(terms[:,:,3]/4.184-coeff)))
        require(coeff_error<1e-9,'Frozen coefficients differ')
        binding=graph.bind_both_systems(paths['M0'].read_bytes(),paths['Full'].read_bytes(),geometry,
                                      sites,nsolute,ntotal)
        expected=json.loads(paths['binding'].read_text())
        for key in ('semantic_to_atom_index_0based','bonds','unwrap_tree_parent_child'):
            require(binding[key]==expected[key],'Archived chemical graph differs: '+key)
        bounds=table(paths['roe_bounds'])
        pairs=np.array([[obs.resolve_endpoint(r[k],binding) for k in ('source_endpoint_1','source_endpoint_2')]
                        for r in bounds],dtype=np.int64)
        require(len(pairs)==spec['roe_count'] and pairs.tolist()==expected['restraint_atom_pairs_0based'],
                'Archived ROE atom binding differs')
        require([r['restraint_id'] for r in bounds]==expected['bound_restraint_ids'],'ROE identifiers differ')
        binding['restraint_count']=len(pairs)
        allq=[];array_reports=[]
        for rep in ('R1','R2'):
            with np.load(paths[rep],allow_pickle=False) as z:
                p=z['positions_nm'];v=z['velocities_nm_ps'];box=z['box_nm']
                require(p.shape==v.shape==(12,ntotal,3) and box.shape==(3,3),'Start array dimensions')
                require(all(np.isfinite(x).all() for x in (p,v,box)),'Nonfinite start arrays')
                require(np.linalg.det(box)>0,'Box volume')
                boxes=np.broadcast_to(box,(12,3,3))
                q=core.angles(p[:,:nsolute],boxes,geometry)[:,np.array(sites)-1,:]
                distance,_=graph.intramolecular_distances_A(p[:,:nsolute],boxes,pairs,binding)
                require(distance.shape==(12,len(pairs)) and np.all(distance>0),'Coordinate-to-ROE distance replay')
                allq.append(q)
                array_reports.append({'repeat':rep,'replicas':12,'particles':ntotal,
                    'positions_and_velocities_finite':True,'box_volume_nm3':float(np.linalg.det(box))})
        q=np.concatenate(allq)
        energies=core.local_xml_terms(q,terms).sum(axis=-1)
        delta=float(np.max(np.abs(energies-independent_energy(q,signs,coeff))))
        require(delta<1e-8,'Independent Fourier/XML energy mismatch')
        qt=q.copy();qt[:,:,2]=180
        trans_error=float(np.max(np.abs(core.local_xml_terms(qt,terms)[:,:,81:].sum(axis=-1))))
        require(trans_error<1e-10,'J does not vanish at trans')
        for contract in spec['formal_contracts']:
            require(contract['physical_xml_sha256']==spec['files'][contract['arm']]['sha256'],
                    'Formal physical system pin differs')
            require(contract['formal_start_sha256']==spec['files'][contract['repeat']]['sha256'],
                    'Formal paired start pin differs')
            contracts_checked+=1
        reports.append({'system':spec['system'],'methylated_residues':sites,
            'particles':ntotal,'solute_particles':nsolute,'ROE_bindings_replayed':len(pairs),
            'correction_terms':nsite*729,'D_residue_correction_signs_checked':True,
            'baseline_forces_constraints_and_box_equal':True,'max_coefficient_difference_kcal_mol':coeff_error,
            'max_XML_vs_independent_energy_difference_kJ_mol_per_site':delta,
            'max_J_at_trans_kJ_mol_per_site':trans_error,'starting_arrays':array_reports})
    report={'status':'PASS','systems':reports,'formal_contracts_checked':contracts_checked,
        'unique_starting_arrays':6,'starting_configurations':72,'ROE_bindings_checked':71,
        'OpenMM_imported':False,'Context_created':False,'QM_or_MD_launched':False,
        'production_trajectories_read':False,
        'scope':'Static native-input and initial-coordinate calculations; not production-ensemble estimates.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('native_inputs_reproduced.json'))
    main(p.parse_args().out)
