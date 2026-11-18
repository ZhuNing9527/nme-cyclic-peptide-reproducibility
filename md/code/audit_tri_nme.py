#!/usr/bin/env python3
"""Audit the frozen tri-NMe models and endpoints on existing data, using NumPy."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import frozen_math_v1 as obs
import graph_binding_v1 as graph


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "inputs"
DATA = ROOT / "data"
PUBLISHED = ROOT / "expected"
ARMS = {"M0": "M0", "A_ONLY": "A only", "FULL": "Full"}
REP = ("R1", "R2")
SITES = (1, 2, 5)
PARAMETERS = ("sphi", "spsi", "somega", "coef", "kp", "cp", "sp", "ks", "cs",
              "ss", "kw", "cw", "sw", "wzero")
ENERGY = (
    "coef*(cp*cos(kp*ph)+sp*sin(kp*ph))*(cs*cos(ks*ps)+ss*sin(ks*ps))*"
    "(wzero+cw*cos(kw*om)+sw*sin(kw*om));ph=sphi*dihedral(p2,p3,p4,p5);"
    "ps=spsi*dihedral(p3,p4,p5,p6);om=somega*dihedral(p1,p2,p3,p4)"
)


def check(have, text):
    if not have:
        raise ValueError(text)


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def xml(path):
    return ET.parse(path).getroot()


def serialize(x):
    # XML indentation and the tail of a force can change with its list position.
    return (x.tag, sorted(x.attrib.items()), (x.text or "").strip(), tuple(map(serialize, x)))


def correction_element(root):
    elements = root.findall("./Forces/Force[@type='CustomCompoundBondForce']")
    check(len(elements) == 1, "Expected the single frozen correction force")
    force, = elements
    check(force.get("forceGroup") == "31" and force.get("particles") == "6"
          and force.get("usesPeriodic") == "1" and force.get("energy") == ENERGY,
          "Correction force metadata do not match")
    check(tuple(x.get("name") for x in force.find("PerBondParameters")) == PARAMETERS,
          "Unexpected parameter order")
    return force


def noncorrected_forces(root):
    elements = [f for f in root.find("Forces") if f.get("type") != "CustomCompoundBondForce"]
    return {f.get("type"): serialize(f) for f in elements}


def a_only_xml(full_path):
    """Keep the frozen A terms and set only the omega-dependent coefficients to zero."""
    full = xml(full_path)
    force = correction_element(full)
    omega_frequency = "param" + str(PARAMETERS.index("kw") + 1)
    coefficient = "param" + str(PARAMETERS.index("coef") + 1)
    bonds = force.findall("Bonds/Bond")
    ntotal = len(bonds)
    check(ntotal == 3 * 729, "Term count")
    jcount = 0
    for bond in bonds:
        if int(bond.get(omega_frequency)) != 0:
            jcount += 1
            bond.set(coefficient, "0")
    check(jcount == 3 * 648, "Wrong number of J terms")
    return ET.tostring(full, encoding="utf-8", xml_declaration=True)


def force_table(root, geometry):
    rows = defaultdict(list)
    for bond in correction_element(root).findall("Bonds/Bond"):
        key = tuple(int(bond.get(f"p{i}")) for i in range(1, 7))
        rows[key].append([float(bond.get(f"param{i}")) for i in range(1, 15)])
    ordered = []
    for i in SITES:
        g = geometry[i - 1]
        atoms = tuple(g["omega_atoms"] + g["psi_atoms"][2:])
        check(tuple(g["phi_atoms"]) == atoms[1:5] and tuple(g["psi_atoms"]) == atoms[2:6],
              "Geometry conventions do not match XML")
        ordered.append(rows.pop(atoms))
    check(not rows, "Unexpected correction site")
    arr = np.array(ordered)
    check(arr.shape == (3, 729, 14) and np.isfinite(arr).all(), "Invalid term table")
    check(np.array_equal(arr[:, 0, 0], [-1, 1, 1]), "Unexpected D-Ala / L-Ala signs")
    check(np.array_equal(arr[:, :, :3], np.broadcast_to(
        np.array([-1, 1, 1])[:, None, None], (3, 729, 3))), "Inconsistent signs")
    return arr


def _basis_spec(n=4):
    a = [(0, 1, 0)]
    for k in range(1, n + 1):
        a += [(k, 1, 0), (k, 0, 1)]
    return np.array(a)


def verify_basis(table):
    basis = _basis_spec()
    trans = (0, 0, 0, 1)
    a = [tuple(p) + tuple(s) + trans for p in basis for s in basis]
    j = [tuple(p) + tuple(s) + (k, c, sn, -c * (-1) ** k)
         for p in basis for s in basis for k, c, sn in basis[1:]]
    check(np.array_equal(table[:, :, 4:], np.broadcast_to(np.array(a + j), (3, 729, 10))),
          "Basis order or trans shift differs from the Fourier fit")


def local_xml_terms(angles, matrix):
    """Evaluate the serialized force expression at specified torsions, in kJ/mol."""
    p, s, o = (np.deg2rad(angles)[:, :, None, :] * matrix[None, :, :, :3]).transpose(3, 0, 1, 2)
    k_p, cp, sp, k_s, cs, ss, k_w, cw, sw, zero = matrix[:, :, 4:].transpose(2, 0, 1)
    return (matrix[:, :, 3] *
            (cp * np.cos(k_p * p) + sp * np.sin(k_p * p)) *
            (cs * np.cos(k_s * s) + ss * np.sin(k_s * s)) *
            (zero + cw * np.cos(k_w * o) + sw * np.sin(k_w * o)))


def frozen_fit_energy(angles, coeff):
    def basis(x):
        x = np.asarray(x)[..., None] * np.arange(1, 5)
        b = np.stack((np.cos(x), np.sin(x)), axis=-1).reshape(*x.shape[:-1], 8)
        return np.concatenate((np.ones((*x.shape[:-1], 1)), b), axis=-1)

    signed = np.deg2rad(angles * np.array([-1, 1, 1])[None, :, None])
    p, s, o = signed.transpose(2, 0, 1)
    p, s = (basis(x) for x in (p, s))
    o = basis(o)[..., 1:] - basis(np.pi)[1:]
    a = np.einsum("...i,...j,ij->...", p, s, coeff[:81].reshape(9, 9))
    j = np.einsum("...i,...j,...k,ijk->...", p, s, o, coeff[81:].reshape(9, 9, 8))
    return 4.184 * (a + j)


def angles(positions, boxes, geometry):
    return np.stack([np.column_stack([obs.omega_degrees(positions, boxes, r[f"{k}_atoms"])
                                       for r in geometry]) for k in ("phi", "psi", "omega")], axis=-1)


def verify_models(geometry, out):
    m0, full, a_only = [xml(INP / x) for x in ("M0.xml", "Full.xml", "A_only.xml")]
    check(len(m0.find("Forces")) == 5 and len(full.find("Forces")) == len(a_only.find("Forces")) == 6
          and not m0.findall("./Forces/Force[@type='CustomCompoundBondForce']"), "Wrong arm definitions")
    check(noncorrected_forces(m0) == noncorrected_forces(full) == noncorrected_forces(a_only),
          "Noncorrection forces are not identical")
    for root in (full, a_only):
        check(all(serialize(m0.find(x)) == serialize(root.find(x))
                  for x in ("Constraints", "PeriodicBoxVectors", "Particles")), "A base System setting changed")
    check((INP / "A_only.xml").read_bytes() == a_only_xml(INP / "Full.xml"), "A-only reconstruction mismatch")

    full_terms, a_terms = force_table(full, geometry), force_table(a_only, geometry)
    verify_basis(full_terms)
    check(np.array_equal(full_terms[:, :, 4:], a_terms[:, :, 4:])
          and np.array_equal(full_terms[:, :81], a_terms[:, :81])
          and np.count_nonzero(a_terms[:, 81:, 3]) == 0, "Non-A terms remained in A only")
    keys = read_tsv(PUBLISHED / "QM246_matched_fourier_coefficients.tsv")
    check([int(x["column_full_basis_0based"]) for x in keys] == list(range(729)), "Coefficient order")
    coef = np.array([float(x["joint_coeff_kcal_mol"]) for x in keys])
    err = np.abs(full_terms[:, :, 3] / 4.184 - coef)
    check(np.max(err) < 1e-9, "Serialized coefficients do not match the archived fit")
    check(np.max(err[:, 1:]) < 1e-11, "Non-constant coefficients differ")

    confs = []
    for k in REP:
        with np.load(INP / f"{k}_initial_replicas.npz", allow_pickle=False) as a:
            p, b, v = a["positions_nm"], a["box_nm"], a["velocities_nm_ps"]
            check(p.shape == v.shape == (12, 4939, 3) and b.shape == (3, 3)
                  and all(np.isfinite(x).all() for x in (p, v, b)), "Unexpected archived start arrays")
            confs.append(angles(p[:, :59], np.broadcast_to(b, (12, 3, 3)), geometry)[:, [0, 1, 4], :])
    q = np.concatenate(confs)
    t = local_xml_terms(q, full_terms)
    ao = local_xml_terms(q, a_terms).sum(axis=-1)
    check(np.allclose(ao, t[:, :, :81].sum(axis=-1), atol=1e-10, rtol=0),
          "A-only energy does not agree with the A components of Full")
    energy_error = np.max(np.abs(t.sum(axis=-1) - frozen_fit_energy(q, coef)))
    check(energy_error < 1e-8, "Independent Fourier energy does not agree with XML")
    q_trans = q.copy()
    q_trans[:, :, 2] = 180
    qt = local_xml_terms(q_trans, full_terms)
    trans_error = np.max(np.abs(qt[:, :, 81:].sum(axis=-1)))
    check(trans_error < 1e-10, "J must vanish at trans")

    r = {
        "sites": list(SITES), "particles": 4939, "solute_particles": 59,
        "archived_start_configurations": len(q),
        "full_terms_per_site": 729, "a_terms_per_site": 81, "j_terms_per_site": 648,
        "baseline_forces_and_constraints_unchanged": True,
        "max_coefficient_delta_kcal_mol": float(np.max(err)),
        "max_nonconstant_coefficient_delta_kcal_mol": float(np.max(err[:, 1:])),
        "max_xml_vs_independent_fit_energy_delta_kJ_mol_per_site": float(energy_error),
        "max_J_at_trans_kJ_mol_per_site": float(trans_error),
        "a_only_origin": "Reconstructed from the archived Full XML by zeroing all omega-dependent coefficients.",
        "ran_dynamics": False,
        "used_OpenMM_Context": False,
    }
    (out / "model_check.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print("PASS 3 OpenMM arm definitions, 729 fit coefficients per site, 24 frozen start coordinates")


def pdb_coordinates(path, elements):
    q = [s for s in path.read_text().splitlines() if s.startswith(("ATOM  ", "HETATM"))]
    check([int(s[6:11]) for s in q] == list(range(1, len(elements) + 1)), "Representative atom order")
    check([s[76:78].strip() for s in q] == elements, "Representative atom elements")
    return np.array([[float(s[k:k + 8]) for k in (30, 38, 46)] for s in q]) / 10.0


def verify_coordinates(geometry, out):
    b0, b1 = ((INP / f"{m}.xml").read_bytes() for m in ("M0", "Full"))
    _, elements, _ = graph.read_system_graph(b0, 59, 4939, 3)
    binding = graph.bind_both_systems(b0, b1, geometry, SITES, 59, 4939)
    targets = read_tsv(INP / "roe_bounds.tsv")
    pairs = np.array([[obs.resolve_endpoint(t[k], binding) for k in ("source_endpoint_1", "source_endpoint_2")]
                      for t in targets], dtype=np.int64)
    expected = json.loads((INP / "REVIEWED_TOPOLOGY_BINDING_v1.json").read_text())
    check(len(pairs) == 20 and pairs.tolist() == expected["restraint_atom_pairs_0based"], "ROE binding mismatch")
    check(binding["semantic_to_atom_index_0based"] == expected["semantic_to_atom_index_0based"],
          "Original chemical labels differ")
    check([x["restraint_id"] for x in targets] == expected["bound_restraint_ids"], "ROE identifiers differ")
    binding["restraint_count"] = 20
    lower = np.array([float(t["lower_bound_A"]) for t in targets])
    upper = np.array([float(t["upper_bound_A"]) for t in targets])

    refs = read_tsv(INP / "backbone_reference.tsv")
    ref = np.array([[float(t["phi_deg"]), float(t["psi_deg"])] for t in refs])
    check([int(r["residue_index"]) for r in refs] == [1, 2, 3, 4, 5] and
          all(r["omega_state"] == "t" for r in refs), "Non-frozen reference")
    frames = read_tsv(DATA / "representative_frames.tsv")
    check(len(frames) == 12 and len({x["file"] for x in frames}) == 12, "Representative-frame list")
    for x in frames:
        check(Path(x["file"]).name == x["file"], "Unsafe representative filename")
    p = np.array([pdb_coordinates(DATA / "representatives" / r["file"], elements) for r in frames])
    with np.load(INP / "R1_initial_replicas.npz", allow_pickle=False) as d:
        box = d["box_nm"]
    boxes = np.broadcast_to(box, (len(p), 3, 3))
    q = angles(p, boxes, geometry)
    s = obs.calculate_observables(*q.transpose(2, 0, 1), ref, [2] * 5)
    max_error_crms = np.max(np.abs(s["crms_deg"] - np.array([float(x["backbone_crms_deg"]) for x in frames])))
    max_omega = np.max(np.abs(obs.periodic_delta_deg(q[:, :, 2], 180)), axis=1)
    max_error_omega = np.max(np.abs(max_omega - np.array([float(x["omega_max_deviation_deg"]) for x in frames])))
    check(max_error_crms < .15 and max_error_omega < .15, "Rounded coordinates do not reproduce original frame angles")
    check(s["M2"].tolist() == [r["in_M2"] == "True" for r in frames], "Wrong representative M2 classifications")

    d, _ = graph.intramolecular_distances_A(p, boxes, pairs, binding)
    d2 = 10 * np.linalg.norm(p[:, pairs[:, 1]] - p[:, pairs[:, 0]], axis=-1)
    check(np.allclose(d, d2, atol=1e-10, rtol=0), "Unwrapped distances disagree with direct coordinate distances")
    pwrapped = np.mod(p, np.diag(box))
    d_w, _ = graph.intramolecular_distances_A(pwrapped, boxes, pairs, binding)
    check(np.allclose(d, d_w,
                      atol=1e-10, rtol=0), "PBC changed ROE distances")
    mm = d ** -3
    record = []
    for i, x in enumerate(frames):
        _, distance, *_ = obs.evaluate_moments(mm[i], lower, upper)
        check(np.allclose(distance, d[i], rtol=0, atol=1e-10), "Single-frame inverse-cube reconstruction")
        ctx = {"file": x["file"], "arm": x["arm"], "repeat": x["repeat"], "scope": "representative_frame"}
        record.extend(obs.restraint_records(targets, mm[i], lower, upper, ctx))
    obs.write_table(out / "representative_ROE_distances.tsv", record)
    obs.write_table(out / "representative_M2_checks.tsv", [
        {"file": x["file"], "arm": x["arm"], "repeat": x["repeat"],
         "source_M2": x["in_M2"], "replayed_M2": bool(v), "backbone_crms_deg": float(r),
         "omega_max_deviation_deg": float(w)} for x, v, r, w in zip(frames, s["M2"], s["crms_deg"], max_omega)
    ])
    print(f"PASS 12 archived representative frames, all 20 original ROE atom pairs; "
          f"max PDB rounding deviation (cRMS/omega)={max_error_crms:.4f}/{max_error_omega:.4f} deg")


def verify_runs(out):
    arm_rows = read_tsv(PUBLISHED / "tri_NMe_ablation.tsv")
    runs = {(x["model"], x["repeat"]): x for x in arm_rows}
    blocks = {(x["model"], x["repeat"], int(x["block"])): x
              for x in read_tsv(PUBLISHED / "tri_NMe_M2_blocks.tsv")}
    repeats = {(x["model"], x["repeat"]): x for x in read_tsv(PUBLISHED / "peptide_repeat_endpoints.tsv")
               if x["system"] == "tri-NMe"}
    row_out, block_out = [], []
    for arm, model in ARMS.items():
        for rep in REP:
            with np.load(DATA / "derived_runs" / f"{arm}_{rep}_DERIVED_v1.npz", allow_pickle=False) as a:
                j = a["iteration"]
                check(np.array_equal(j, np.arange(100010, 600001, 10)) and a["M2"].shape == (50_000,),
                      "Wrong 50 ns physical-state frame indices")
                w, c, rmsd = [a[s] for s in ("omega_max_deviation_deg", "backbone_crms_deg", "rmsd_A")]
                m1 = w <= obs.WINDOW_DEG
                m2 = m1 & (c <= obs.CRMS_MAX_DEG)
                check(np.array_equal(m2, a["M2"]), "Published per-frame M2 mask differs")
                result = obs.summarize_series({"M1": m1, "M2": m2, "crms_deg": c})
                rmsd_med = float(np.median(rmsd))
                ct = result["M2_major_frames"]
                check(abs(100 * ct / len(j) - float(runs[model, rep]["M2_pct"])) < 1e-9,
                      "Published per-repeat M2 differs")
                check(abs(rmsd_med - float(runs[model, rep]["median_RMSD_A"])) <= .000051,
                      "Published median RMSD differs")
                if (model, rep) in repeats:
                    q = repeats[model, rep]
                    check(abs(100 * result["M1_exact_pattern_occupancy"] - float(q["M1_pct"])) < 1e-8
                          and abs(result["M3_crms_mean_deg"] - float(q["M3_mean_deg"])) <= .00000051,
                          "Published M1/M3 differs")
                row_out.append({"model": model, "repeat": rep, **result, "median_RMSD_A": rmsd_med})
                for b in obs.block_summaries({"M1": m1, "M2": m2, "crms_deg": c}, f"{arm}_{rep}"):
                    expected = blocks[model, rep, b["block_index"]]
                    check(int(expected["in_M2_frames"]) == b["M2_major_frames"], "Published 10 ns M2 block differs")
                    block_out.append({"model": model, "repeat": rep, **b})
    obs.write_table(out / "replayed_ensemble_endpoints.tsv", row_out)
    obs.write_table(out / "replayed_10ns_blocks.tsv", block_out)
    by = {a: np.mean([r["M2_joint_major_basin_occupancy"] for r in row_out if r["model"] == a])
          for a in ARMS.values()}
    check(np.allclose([by[x] for x in ARMS.values()], [0.07643, 0.86857, 0.91596], rtol=0, atol=1e-12),
          "Arm mean M2 differs")
    print("PASS archived M2 masks and 30 block counts; equal-repeat averages:",
          ", ".join(f"{a}={100*p:.3f}%" for a, p in by.items()))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    d = p.parse_args().out
    d.mkdir(parents=True, exist_ok=False)
    geometry = json.loads((INP / "SOLUTE_GEOMETRY_v1.json").read_text())
    obs.validate_geometry(geometry, 59)
    verify_models(geometry, d)
    verify_coordinates(geometry, d)
    verify_runs(d)


if __name__ == "__main__":
    main()
