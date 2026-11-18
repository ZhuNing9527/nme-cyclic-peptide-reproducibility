"""Numerical functions copied without edits from the completed frozen JACS3 analysis."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
NREP=12
WINDOW_DEG=30.0
CRMS_MAX_DEG=45.0
M4="ROE_M4_restraint_satisfaction_fraction"
M5="ROE_M5_mean_absolute_bound_violation_A"

def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def validate_mapping(mapping, rows, nrep=NREP):
    require(mapping.shape == (rows, nrep), "MAPPING_SHAPE")
    require(np.issubdtype(mapping.dtype, np.integer), "NONINTEGER_MAPPING")
    require(np.array_equal(np.sort(mapping, axis=1),
                          np.broadcast_to(np.arange(nrep), mapping.shape)), "MAPPING_NOT_PERMUTATION")

def validate_audit(audit, start, end, nrep=NREP):
    n = end - start
    require(np.array_equal(audit["iteration"], np.arange(start + 1, end + 1)), "AUDIT_GLOBAL_ITERATIONS")
    before, after = audit["before"], audit["after"]
    validate_mapping(before, n, nrep)
    validate_mapping(after, n, nrep)
    require(np.array_equal(before[1:], after[:-1]), "AUDIT_WITHIN_CHUNK_MAPPING_HANDOFF")
    energy, proposed, accepted = (audit[k] for k in ("energy", "proposed", "accepted"))
    for label, array in (("energy", energy), ("proposed", proposed), ("accepted", accepted)):
        require(array.shape == (n, nrep, nrep), label + "_SHAPE")
        require(np.isfinite(array).all(), label + "_NONFINITE")
    adjacent = np.abs(np.arange(nrep)[:, None] - np.arange(nrep)[None, :]) == 1
    require(np.all(proposed[:, ~adjacent] == 0), "NONNEIGHBOR_PROPOSAL")
    require(np.all((proposed == 0) | (proposed == 1)), "NONBINARY_PROPOSAL")
    require(np.all((accepted == 0) | (accepted == 1)), "NONBINARY_ACCEPTANCE")
    require(np.all(accepted <= proposed), "ACCEPTED_WITHOUT_PROPOSAL")
    require(np.array_equal(proposed, proposed.swapaxes(1, 2)), "ASYMMETRIC_PROPOSAL")
    require(np.array_equal(accepted, accepted.swapaxes(1, 2)), "ASYMMETRIC_ACCEPTANCE")
    require(np.all(proposed.sum(axis=2) <= 1), "OVERLAPPING_NEIGHBOR_PROPOSALS")
    # Each event must propose precisely one of the two swap-neighbors matchings.
    even = np.zeros((nrep, nrep), dtype=np.int8)
    odd = even.copy()
    for parity, matrix in ((0, even), (1, odd)):
        for i in range(parity, nrep - 1, 2):
            matrix[i, i + 1] = matrix[i + 1, i] = 1
    valid_proposals = np.all(proposed == even, axis=(1, 2)) | np.all(proposed == odd, axis=(1, 2))
    require(valid_proposals.all(), "PROPOSAL_MATCHING_NOT_SWAP_NEIGHBORS")
    inverse = np.argsort(before, axis=1)
    expected_after = before.copy()
    probability = np.empty((n, nrep - 1), dtype=np.float64)
    ti = np.arange(n)
    for i in range(nrep - 1):
        j = i + 1
        ri, rj = inverse[:, i], inverse[:, j]
        logp = -energy[ti, ri, j] - energy[ti, rj, i] + energy[ti, ri, i] + energy[ti, rj, j]
        require(np.isfinite(logp).all(), "NONFINITE_EXCHANGE_LOG_RATIO")
        probability[:, i] = np.exp(np.minimum(0.0, logp))
        mask = accepted[:, i, j] == 1
        expected_after[ti[mask], ri[mask]] = j
        expected_after[ti[mask], rj[mask]] = i
    require(np.array_equal(after, expected_after), "MAPPING_ACCEPTANCE_INCONSISTENT")
    prop_edge = proposed[:, np.arange(nrep - 1), np.arange(1, nrep)]
    acc_edge = accepted[:, np.arange(nrep - 1), np.arange(1, nrep)]
    return {
        "proposed": prop_edge.sum(axis=0), "accepted": acc_edge.sum(axis=0),
        "expected_proposed_sum": (probability * prop_edge).sum(axis=0),
        "expected_all_sum": probability.sum(axis=0),
    }

def validate_capture(cap, audit, start, end, nsolute, nrep=NREP):
    n1, n10 = (end - start) // 10, (end - start) // 100
    for cadence, count in ((1, n1), (10, n10)):
        stride = cadence * 10
        require(np.array_equal(cap[f"capture_{cadence}ps_iteration"],
                               np.arange(start + stride, end + 1, stride)), "CAPTURE_GLOBAL_ITERATIONS")
        validate_mapping(cap[f"capture_{cadence}ps_mapping"], count, nrep)
        extra = () if cadence == 1 else (nrep,)
        require(cap[f"capture_{cadence}ps_positions"].shape == (count,) + extra + (nsolute, 3),
                "POSITIONS_SHAPE")
        boxes = cap[f"capture_{cadence}ps_box"]
        require(boxes.shape == (count,) + extra + (3, 3), "BOX_SHAPE")
        for key in (f"capture_{cadence}ps_positions", f"capture_{cadence}ps_box"):
            require(np.isfinite(cap[key]).all(), "NONFINITE_COORDINATES_OR_BOX")
        require(np.all(np.linalg.det(boxes) > 0), "NONPOSITIVE_BOX_VOLUME")
        audited = audit["after"][stride - 1::stride]
        require(np.array_equal(cap[f"capture_{cadence}ps_mapping"], audited), "CAPTURE_AUDIT_MAPPING")
    replica = cap["capture_1ps_lambda1_replica"]
    require(replica.shape == (n1,) and np.issubdtype(replica.dtype, np.integer), "LAMBDA1_WALKER_SHAPE")
    require(np.all((replica >= 0) & (replica < nrep)), "LAMBDA1_WALKER_RANGE")
    require(np.all(cap["capture_1ps_mapping"][np.arange(n1), replica] == 0), "NOT_PHYSICAL_LAMBDA1")
    require(cap["capture_1ps_energy"].shape == (n1, nrep, nrep), "CAPTURE_ENERGY_SHAPE")
    require(np.isfinite(cap["capture_1ps_energy"]).all(), "NONFINITE_CAPTURE_ENERGY")
    # Audit i+1 uses energies from the post-propagation report at iteration i.
    require(np.array_equal(cap["capture_1ps_energy"][:-1], audit["energy"][10::10]),
            "CAPTURE_ENERGY_AUDIT_HANDOFF")
    at1 = np.arange(9, n1, 10)
    at10 = np.arange(n10)
    require(np.array_equal(cap["capture_10ps_positions"][at10, replica[at1]],
                           cap["capture_1ps_positions"][at1]), "CROSS_STREAM_POSITIONS_DIFFER")
    require(np.array_equal(cap["capture_10ps_box"][at10, replica[at1]],
                           cap["capture_1ps_box"][at1]), "CROSS_STREAM_BOX_DIFFER")

def omega_degrees(positions, boxes, atoms):
    """Minimum-image bonded dihedral, in degrees. Used for sampling only."""
    p = np.asarray(positions, dtype=np.float64)[..., atoms, :]
    bonds = np.diff(p, axis=-2)
    boxes = np.asarray(boxes, dtype=np.float64)
    # Frozen DMSO boxes are expected to be orthorhombic. Refuse a different cell
    # rather than silently applying a nearest-image approximation to skew boxes.
    diagonal = np.zeros_like(boxes)
    diagonal[..., np.arange(3), np.arange(3)] = boxes[..., np.arange(3), np.arange(3)]
    require(np.allclose(boxes, diagonal, atol=1e-12, rtol=0), "SKEW_BOX_NEEDS_EXPLICIT_IMAGE_REVIEW")
    lengths = np.diagonal(boxes, axis1=-2, axis2=-1)[..., None, :]
    require(np.all(lengths > 0), "INVALID_BOX_LENGTH")
    bonds -= np.rint(bonds / lengths) * lengths
    b0, b1, b2 = -bonds[..., 0, :], bonds[..., 1, :], bonds[..., 2, :]
    norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    require(np.all(norm > 1e-12), "DEGENERATE_DIHEDRAL_BOND")
    b1 = b1 / norm
    v = b0 - (b0 * b1).sum(axis=-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(axis=-1, keepdims=True) * b1
    require(np.all(np.linalg.norm(v, axis=-1) > 1e-12) and
            np.all(np.linalg.norm(w, axis=-1) > 1e-12), "DEGENERATE_DIHEDRAL_PLANE")
    x = (v * w).sum(axis=-1)
    y = (np.cross(b1, v) * w).sum(axis=-1)
    return np.degrees(np.arctan2(y, x))

def basin_labels(angles):
    absolute = np.abs(angles)
    return np.where(absolute <= 60.0, 1, np.where(absolute >= 120.0, 2, 0)).astype(np.int8)

def norm_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum(axis=-1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1))).sum(axis=-1) / np.log(counts.shape[-1])

def count_roundtrips(mapping_segments, nrep=NREP):
    """Two alternating endpoint passages complete a trip; reset per segment."""
    result = np.zeros(nrep, dtype=np.int64)
    for segment in mapping_segments:
        for walker in range(nrep):
            sequence = segment[:, walker]
            sequence = sequence[(sequence == 0) | (sequence == nrep - 1)]
            result[walker] += int(np.count_nonzero(np.diff(sequence))) // 2
    return result

class Crossings:
    def __init__(self, nrep=NREP):
        self.nrep = nrep
        self.previous = np.zeros(nrep, dtype=np.int8)
        self.walker_counts = np.zeros(nrep, dtype=np.int64)
        self.state_counts = np.zeros(nrep, dtype=np.int64)
        self.state_population = np.zeros((nrep, 3), dtype=np.int64)

    def reset_boundary(self):
        self.previous[:] = 0

    def update(self, labels, mapping):
        require(labels.shape == mapping.shape, "CROSSING_SHAPE")
        for state in range(self.nrep):
            self.state_population[state] += np.bincount(labels[mapping == state], minlength=3)
        for w in range(self.nrep):
            indices = np.flatnonzero(labels[:, w] != 0)
            if not len(indices):
                continue
            seq = labels[indices, w]
            preceding = np.r_[self.previous[w], seq[:-1]]
            changed = (preceding != 0) & (seq != preceding)
            self.walker_counts[w] += int(changed.sum())
            self.state_counts += np.bincount(mapping[indices[changed], w], minlength=self.nrep)
            self.previous[w] = seq[-1]

    def result(self):
        total = int(self.walker_counts.sum())
        classes = ("NO_COMMITTED_INTERCONVERSION_OBSERVED", "SINGLE_COMMITTED_INTERCONVERSION_OBSERVED")
        return {
            "crossings": total,
            "class": classes[total] if total < 2 else "COMMITTED_RETURN_OBSERVED",
            "crossings_by_walker": self.walker_counts.tolist(),
            "crossings_by_state_at_destination": self.state_counts.tolist(),
            "state_counts_order": ["intermediate", "cis", "trans"],
            "state_population_counts": self.state_population.tolist(),
            "state_population_fractions": (self.state_population / self.state_population.sum(axis=1)[:, None]).tolist(),
            "physical_kinetic_rate_interpretation": False,
        }

def periodic_delta_deg(a, b):
    return (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 180.0) % 360.0 - 180.0

def frozen_omega_classes(omega):
    """0 intermediate, 1 cis, 2 trans. Scientific windows are +/-30, not 60."""
    omega = np.asarray(omega, dtype=np.float64)
    require(np.isfinite(omega).all(), "NONFINITE_OMEGA")
    cis = np.abs(periodic_delta_deg(omega, 0.0)) <= WINDOW_DEG
    trans = np.abs(periodic_delta_deg(omega, 180.0)) <= WINDOW_DEG
    require(not np.any(cis & trans), "OVERLAPPING_OMEGA_WINDOWS")
    return np.where(cis, 1, np.where(trans, 2, 0)).astype(np.uint8)

def calculate_observables(phi, psi, omega, reference, target_classes):
    phi, psi, omega = [np.asarray(x, dtype=np.float64) for x in (phi, psi, omega)]
    require(phi.ndim == 2 and phi.shape[1] == 5 and phi.shape == psi.shape == omega.shape, "ANGLE_ARRAY_SHAPE")
    require(all(np.isfinite(x).all() for x in (phi, psi, omega)), "NONFINITE_TORSION")
    reference = np.asarray(reference, dtype=np.float64)
    target_classes = np.asarray(target_classes, dtype=np.uint8)
    require(reference.shape == (5, 2) and np.isfinite(reference).all(), "REFERENCE_ANGLE_SHAPE")
    require(target_classes.shape == (5,) and np.isin(target_classes, [1, 2]).all(), "TARGET_PATTERN")
    angles = np.stack([phi, psi], axis=-1)
    differences = periodic_delta_deg(angles, reference)
    # The source D-Ala reference is already signed for the actual D residue.
    # Do not apply the force-field mirror rule again to any analysis angle.
    crms = np.sqrt(np.mean(differences**2, axis=(1, 2)))
    classes = frozen_omega_classes(omega)
    m1 = np.all(classes == target_classes, axis=1)
    m2 = m1 & (crms <= CRMS_MAX_DEG)
    require(np.all(~m2 | m1), "M2_NOT_SUBSET_OF_M1")
    return {"M1": m1, "M2": m2, "crms_deg": crms, "omega_class": classes}

def summarize_series(series):
    n = len(series["M1"])
    require(n > 0 and len(series["M2"]) == len(series["crms_deg"]) == n, "SERIES_LENGTH")
    return {
        "frames": n,
        "M1_matching_frames": int(np.count_nonzero(series["M1"])),
        "M2_major_frames": int(np.count_nonzero(series["M2"])),
        "M1_exact_pattern_occupancy": float(np.mean(series["M1"])),
        "M2_joint_major_basin_occupancy": float(np.mean(series["M2"])),
        "M3_crms_mean_deg": float(np.mean(series["crms_deg"])),
        "M3_crms_median_deg": float(np.median(series["crms_deg"])),
    }

def block_summaries(series, row, frames_per_block=10000):
    require(len(series["M1"]) == 5 * frames_per_block, "FIVE_FIXED_BLOCKS_REQUIRE_EQUAL_LENGTH")
    records = []
    for index in range(5):
        section = slice(index * frames_per_block, (index + 1) * frames_per_block)
        record = summarize_series({key: series[key][section] for key in ("M1", "M2", "crms_deg")})
        record.update(row=row, block_index=index + 1,
                      production_start_ns_exclusive=index * 10, production_end_ns_inclusive=(index + 1) * 10,
                      global_start_ns_exclusive=10 + index * 10, global_end_ns_inclusive=20 + index * 10,
                      independent_statistical_replicate=False)
        records.append(record)
    return records

def validate_geometry(geometry, nsolute):
    require(len(geometry) == 5 and [g["residue_index"] for g in geometry] == [1, 2, 3, 4, 5], "GEOMETRY_RESIDUE_ORDER")
    for g in geometry:
        for kind in ("phi_atoms", "psi_atoms", "omega_atoms"):
            atoms = g[kind]
            require(len(atoms) == 4 and len(set(atoms)) == 4
                    and all(type(i) is int and 0 <= i < nsolute for i in atoms), "GEOMETRY_ATOM_INDICES")
    ns = [g["phi_atoms"][1] for g in geometry]
    cas = [g["phi_atoms"][2] for g in geometry]
    cs = [g["phi_atoms"][3] for g in geometry]
    for i, g in enumerate(geometry):
        prev, nxt = (i - 1) % 5, (i + 1) % 5
        require(g["phi_atoms"] == [cs[prev], ns[i], cas[i], cs[i]], "PHI_CONVENTION")
        require(g["psi_atoms"] == [ns[i], cas[i], cs[i], ns[nxt]], "PSI_CONVENTION")
        require(g["omega_atoms"] == [cas[prev], cs[prev], ns[i], cas[i]], "OMEGA_CONVENTION")

def write_table(path, records, fields=None):
    require(records, "EMPTY_RESULT_TABLE")
    if fields is None:
        fields = list(records[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

def resolve_endpoint(endpoint, binding):
    match = re.fullmatch(r"1:(D-)?ALA([1-5]):(Hα|HN|Cβ|CN)", endpoint)
    require(match is not None, "UNSUPPORTED_SOURCE_ENDPOINT:" + endpoint)
    residue = int(match.group(2))
    require(bool(match.group(1)) == (residue == 1), "SOURCE_RESIDUE_CHIRALITY_LABEL")
    key = str(residue) + ":" + match.group(3)
    require(key in binding["semantic_to_atom_index_0based"], "NO_CHEMICALLY_VALID_ATOM:" + endpoint)
    return binding["semantic_to_atom_index_0based"][key]

def combine_moments(moments, frame_counts):
    moments = np.asarray(moments, dtype=np.float64)
    weights = np.asarray(frame_counts, dtype=np.float64)
    require(moments.ndim == 2 and weights.shape == (len(moments),) and np.isfinite(weights).all()
            and np.all(weights > 0), "MOMENT_WEIGHT_SHAPE")
    require(np.isfinite(moments).all() and np.all(moments > 0), "INVALID_RMINUS3_MOMENT")
    return np.average(moments, axis=0, weights=weights)

def evaluate_moments(moment, lower, upper):
    moment, lower, upper = [np.asarray(x, dtype=np.float64) for x in (moment, lower, upper)]
    require(moment.ndim == 1 and moment.shape == lower.shape == upper.shape and len(moment) > 0, "RESTRAINT_VECTOR_SHAPE")
    require(np.isfinite(moment).all() and np.all(moment > 0), "INVALID_EFFECTIVE_DISTANCE_MOMENT")
    require(np.isfinite(lower).all() and np.isfinite(upper).all() and np.all((lower > 0) & (upper >= lower)), "INVALID_ANALYSIS_BOUNDS")
    distance = 1.0 / np.cbrt(moment)
    violation = np.maximum(np.maximum(lower - distance, distance - upper), 0.0)
    satisfied = (distance >= lower) & (distance <= upper)
    summary = {"restraint_count": len(moment), "restraints_satisfied": int(np.count_nonzero(satisfied)),
               M4: float(np.mean(satisfied)), M5: float(np.mean(violation))}
    return summary, distance, violation, satisfied

def restraint_records(rows, moment, lower, upper, context):
    _, distance, violation, satisfied = evaluate_moments(moment, lower, upper)
    records = []
    for index, row in enumerate(rows):
        records.append({**context, "restraint_id": row["restraint_id"], "source_row_rank": int(row["source_row_rank"]),
            "source_endpoint_1": row["source_endpoint_1"], "source_endpoint_2": row["source_endpoint_2"],
            "lower_bound_A": row["lower_bound_A"], "upper_bound_A": row["upper_bound_A"],
            "mean_rminus3_Aminus3": float(moment[index]), "r_eff_A": float(distance[index]),
            "within_source_bounds": bool(satisfied[index]), "absolute_bound_violation_A": float(violation[index])})
    return records
