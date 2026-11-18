"""Chemical-graph binding and PBC distances with reviewed per-system counts.

The graph and PBC algorithms follow the completed JACS3 analysis. Fixed JACS3
particle and restraint counts are replaced by independently reviewed system
counts. The endpoint semantics and numerical distance algorithm are unchanged.
"""
from collections import Counter, deque
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from frozen_math_v1 import require


def read_system_graph(data, nsolute, ntotal, nmethyl):
    root = ET.fromstring(data)
    require(root.tag == "System", "NOT_OPENMM_SYSTEM_XML")
    masses = [float(p.attrib["mass"]) for p in root.findall("./Particles/Particle")]
    require(len(masses) == ntotal, "TOTAL_PARTICLE_COUNT")
    require(not root.findall(".//VirtualSite"), "VIRTUAL_SITE_NEEDS_SEPARATE_REVIEW")
    forces = root.findall('./Forces/Force[@type="HarmonicBondForce"]')
    require(len(forces) == 1, "EXACT_ONE_HARMONIC_BOND_FORCE")
    edges = set()
    for node in root.findall("./Constraints/Constraint") + forces[0].findall("./Bonds/Bond"):
        a, b = int(node.attrib["p1"]), int(node.attrib["p2"])
        require(0 <= a < ntotal and 0 <= b < ntotal and a != b, "INVALID_BOND_PARTICLE")
        if min(a, b) < nsolute:
            require(max(a, b) < nsolute, "SOLUTE_SOLVENT_COVALENT_BOND")
            edges.add(tuple(sorted((a, b))))
    require(len(edges) == nsolute, "SOLUTE_BOND_COUNT")
    elements = []
    for mass in masses[:nsolute]:
        matches = [element for value, element in ((1.008, "H"), (12.01, "C"), (14.01, "N"), (16.0, "O")) if abs(mass - value) < 1e-9]
        require(len(matches) == 1, "UNEXPECTED_SOLUTE_MASS_OR_HMR")
        elements.append(matches[0])
    require(Counter(elements) == Counter(C=15+nmethyl, H=25+2*nmethyl, N=5, O=5), "SOLUTE_FORMULA")
    return masses, elements, sorted(edges)


def bind_graph(elements, edges, geometry, methylated):
    nsolute = len(elements)
    require(len(geometry) == 5 and [g['residue_index'] for g in geometry] == list(range(1, 6)), 'RESIDUE_ORDER')
    adjacency = [set() for _ in range(nsolute)]
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, queue, tree = ({0}, deque([0]), [])
    while queue:
        parent = queue.popleft()
        for node in sorted(adjacency[parent]):
            if node not in seen:
                seen.add(node)
                queue.append(node)
                tree.append([parent, node])
    require(len(seen) == nsolute and len(tree) == nsolute - 1, 'DISCONNECTED_SOLUTE_GRAPH')
    for h, element in enumerate(elements):
        if element == 'H':
            require(len(adjacency[h]) == 1, 'HYDROGEN_NOT_TERMINAL')
    residues, semantic, expected_edges, covered = ([], {}, set(), set())

    def add_edge(a, b):
        expected_edges.add(tuple(sorted((a, b))))

    def element_is(atom, element):
        require(type(atom) is int and 0 <= atom < nsolute and (elements[atom] == element), 'ANCHOR_ELEMENT_MISMATCH')

    def unique_hydrogen(atom):
        values = [j for j in adjacency[atom] if elements[j] == 'H']
        require(len(values) == 1, 'NONUNIQUE_ALPHA_HYDROGEN')
        return values[0]

    def methyl_hydrogens(carbon, anchor):
        element_is(carbon, 'C')
        hs = sorted((j for j in adjacency[carbon] if elements[j] == 'H'))
        require(len(hs) == 3 and adjacency[carbon] == {anchor, *hs}, 'NOT_SINGLE_METHYL_CARBON')
        for h in hs:
            add_edge(carbon, h)
        covered.update(hs)
        return hs
    for index, g in enumerate(geometry):
        require(len(g['chirality_atoms']) == 4, 'CHIRALITY_ANCHOR_COUNT')
        ca, n, c, cb = g['chirality_atoms']
        previous_c = geometry[(index - 1) % 5]['phi_atoms'][3]
        next_n = geometry[(index + 1) % 5]['phi_atoms'][1]
        require(g['phi_atoms'] == [previous_c, n, ca, c] and g['psi_atoms'] == [n, ca, c, next_n], 'CYCLIC_BACKBONE_GEOMETRY')
        for atom, element in ((n, 'N'), (ca, 'C'), (c, 'C'), (cb, 'C')):
            element_is(atom, element)
        ha = unique_hydrogen(ca)
        require(adjacency[ca] == {n, c, cb, ha}, 'ALPHA_CARBON_NEIGHBORS')
        oxygens = [j for j in adjacency[c] if elements[j] == 'O']
        require(len(oxygens) == 1, 'CARBONYL_OXYGEN_NOT_UNIQUE')
        oxygen = oxygens[0]
        require(adjacency[oxygen] == {c} and adjacency[c] == {ca, oxygen, next_n}, 'CARBONYL_TOPOLOGY')
        cb_h = methyl_hydrogens(cb, ca)
        extra = adjacency[n] - {ca, previous_c}
        require(ca in adjacency[n] and previous_c in adjacency[n] and (len(extra) == 1), 'AMIDE_N_NEIGHBORS')
        extra_atom = next(iter(extra))
        residue = {'residue_index': index + 1, 'N': n, 'CA': ca, 'C': c, 'O': oxygen, 'Halpha': ha, 'Cbeta_proxy': cb, 'Cbeta_hydrogens_not_averaged': cb_h, 'HN': None, 'CN_proxy': None}
        if index + 1 in methylated:
            element_is(extra_atom, 'C')
            residue['CN_proxy'] = extra_atom
            residue['N_methyl_hydrogens_not_averaged'] = methyl_hydrogens(extra_atom, n)
            semantic[f'{index + 1}:CN'] = extra_atom
        else:
            element_is(extra_atom, 'H')
            require(adjacency[extra_atom] == {n}, 'AMIDE_H_NEIGHBOR')
            residue['HN'] = extra_atom
            semantic[f'{index + 1}:HN'] = extra_atom
        semantic[f'{index + 1}:Hα'] = ha
        semantic[f'{index + 1}:Cβ'] = cb
        for a, b in ((n, ca), (n, previous_c), (n, extra_atom), (ca, c), (ca, cb), (ca, ha), (c, oxygen)):
            add_edge(a, b)
        covered.update((n, ca, c, cb, oxygen, ha, extra_atom))
        residues.append(residue)
    require(covered == set(range(nsolute)), 'UNCLASSIFIED_OR_WRONG_SOLUTE_ATOM')
    require(expected_edges == {tuple(e) for e in edges}, 'INDEPENDENT_CHEMICAL_GRAPH_MISMATCH')
    require(len(set(semantic.values())) == len(semantic), 'REUSED_ROE_COORDINATE_ATOM')
    return {'residues': residues, 'semantic_to_atom_index_0based': semantic, 'bonds': [list(e) for e in edges], 'unwrap_tree_parent_child': tree}

def bind_both_systems(control_bytes, test_bytes, geometry, methylated, nsolute, ntotal):
    methylated = set(methylated)
    require(nsolute == 50 + 3 * len(methylated), "METHYLATION_ATOM_COUNT")
    control = read_system_graph(control_bytes, nsolute, ntotal, len(methylated))
    test = read_system_graph(test_bytes, nsolute, ntotal, len(methylated))
    require(control == test, "CONTROL_TEST_PARTICLE_ORDER_OR_SOLUTE_GRAPH_DIFFER")
    binding = bind_graph(control[1], control[2], geometry, methylated)
    binding.update(version="402EC_REVIEWED_CYCLIC_PENTAALANINE_ROE_BINDING_V1", total_particles=ntotal,
        solute_particles=nsolute, solute_bonds=nsolute,
        solute_formula=f"C{15+len(methylated)}H{25+2*len(methylated)}N5O5",
        N_methylated_residues=sorted(methylated),
        physical_system_xml_sha256={"CONTROL":hashlib.sha256(control_bytes).hexdigest(),"TEST":hashlib.sha256(test_bytes).hexdigest()},
        particle_masses_and_solute_graph_equal_between_arms=True,
        chemical_graph_checked_independently_of_ROE_distances=True,
        carbon_proxies_not_methyl_hydrogen_group_averages=True,
        OpenMM_imported=False, Context_created=False, trajectories_read=False)
    return binding


def intramolecular_distances_A(positions_nm, boxes_nm, pairs, binding):
    nsolute = binding['solute_particles']
    nrestraints = binding['restraint_count']
    positions = np.asarray(positions_nm, dtype=np.float64)
    boxes = np.asarray(boxes_nm, dtype=np.float64)
    pairs = np.asarray(pairs)
    n = len(positions)
    require(n > 0 and positions.shape == (n, nsolute, 3) and (boxes.shape == (n, 3, 3)), 'DISTANCE_INPUT_SHAPE')
    require(pairs.shape == (nrestraints, 2) and np.issubdtype(pairs.dtype, np.integer) and np.all((pairs >= 0) & (pairs < nsolute)) and np.all(pairs[:, 0] != pairs[:, 1]), 'DISTANCE_ATOM_PAIR_SHAPE')
    require(np.isfinite(positions).all() and np.isfinite(boxes).all(), 'NONFINITE_DISTANCE_INPUT')
    diagonal = np.zeros_like(boxes)
    diagonal[:, np.arange(3), np.arange(3)] = boxes[:, np.arange(3), np.arange(3)]
    require(np.allclose(boxes, diagonal, atol=1e-12, rtol=0), 'SKEW_BOX_NEEDS_EXPLICIT_IMAGE_REVIEW')
    lengths = np.diagonal(boxes, axis1=1, axis2=2)
    require(np.all(lengths > 0), 'NONPOSITIVE_BOX_LENGTH')

    def minimum_image(vectors):
        return vectors - np.rint(vectors / lengths[:, None, :]) * lengths[:, None, :]
    unwrapped = np.empty_like(positions)
    unwrapped[:, 0] = positions[:, 0]
    for parent, child in binding['unwrap_tree_parent_child']:
        vector = positions[:, child] - positions[:, parent]
        vector -= np.rint(vector / lengths) * lengths
        unwrapped[:, child] = unwrapped[:, parent] + vector
    bonds = np.asarray(binding['bonds'], dtype=np.int64)
    original_bonds = minimum_image(positions[:, bonds[:, 1]] - positions[:, bonds[:, 0]])
    restored_bonds = unwrapped[:, bonds[:, 1]] - unwrapped[:, bonds[:, 0]]
    closure_error = float(np.max(np.abs(original_bonds - restored_bonds)))
    require(closure_error <= 1e-08, 'MOLECULAR_UNWRAP_RING_CLOSURE')
    vectors = unwrapped[:, pairs[:, 1]] - unwrapped[:, pairs[:, 0]]
    distances = np.linalg.norm(vectors, axis=-1)
    direct = np.linalg.norm(minimum_image(positions[:, pairs[:, 1]] - positions[:, pairs[:, 0]]), axis=-1)
    direct_error = float(np.max(np.abs(distances - direct)))
    require(direct_error <= 1e-08, 'INTRAMOLECULAR_AND_PAIR_IMAGE_DISAGREE')
    require(np.isfinite(distances).all() and np.all(distances > 1e-08), 'ZERO_OR_NONFINITE_ROE_DISTANCE')
    return (10.0 * distances, {'unwrap_closure_max_residual_nm': closure_error, 'direct_minimum_image_distance_max_residual_nm': direct_error})
