import hashlib,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEV=Path('07_nmeala_separability_pilot/13_joint_correction_development')
SHORT=DEV/'20r_post_m4_dmso_solvation_and_short_md_v1'
REPAIR=DEV/'20s_post_m4_python_dependency_isolation_v1'
PERIODIC=DEV/'20u_post_m4_periodic_rest2_validation_v1'
CLOSURE=DEV/'20v_post_m4_periodic_rest2_independent_closure_v1'
FAILED=DEV/'20w_post_m4_state_cache_and_ladder_pilot_v1'
DIAGNOSTIC=DEV/'20x_post_m4_cuda_cache_force_diagnostic_v1'
PRIOR=DEV/'20y_post_m4_double_cache_and_ladder_pilot_v1'
CHECK=DEV/'20z_post_m4_numpy24_swap_compatibility_check_v1'
OUT=DEV/'21a_post_m4_repaired_ladder_pilot_v1'
SYSTEMS=('TIERA_JACS1','TIERA_JACS2','TIERA_JACS3','TIERA_JACS4','TIERA_JACS7')
ARMS=('CONTROL','TEST');SNAPSHOTS=('PARENT','CONTROL','TEST')
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text())
def write_json(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')
def protocol():return read_json(HERE/'PROTOCOL_v1.json')
def pins():return read_json(HERE/'INPUT_PINS_v1.json')
def verify_freeze(base,name):
    base=Path(base);rows=(base/name).read_text().splitlines()
    for row in rows:
        h,p=row.split(None,1);require(not Path(p).is_absolute() and '..' not in Path(p).parts,'UNSAFE_PIN')
        require(sha(base/p)==h,'HASH_CHANGED: '+str(base/p))
    return len(rows)
def freeze(base,name):
    base=Path(base);require(not (base/name).exists(),'FREEZE_EXISTS')
    files=sorted(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    (base/name).write_text(''.join(sha(p)+'  '+p.relative_to(base).as_posix()+'\n' for p in files))
_rest2=None
def load_rest2():
    global _rest2
    if _rest2 is None:
        path=HERE/'rest2_scaling_openmm_v2.py';require(sha(path)==pins()['rest2_source_sha256'],'FROZEN_TRANSFORM_CHANGED')
        spec=importlib.util.spec_from_file_location('frozen_rest2_v2',path);_rest2=importlib.util.module_from_spec(spec);spec.loader.exec_module(_rest2)
    return _rest2
