# NMe-Val formal M0 build v1

## Status

PASS

## Model

Ac-N(CH3)-CH(CH(CH3)2)-C(O)-NHCH3

Formula:
C9H18N2O2

Formal charge:
0

Atoms:
31

Bonds:
30

## Formal baseline

GAFF2 + formal 18-conformer two-stage RESP charges.

Authoritative charge table:
/mnt/lab/zn/mem-perm/P0_starter_v1/07_nmeala_separability_pilot/11_nmeval_transfer/04_formal_resp_fit/NMEVAL_FORMAL_RESP_CHARGE_SET_v1.tsv

Charge-table SHA256:
bb15fd60f71433cfdde2736158b2c98f7c2e0c623850b23706fcf95d8280a8ff

Formal MOL2:
/mnt/lab/zn/mem-perm/P0_starter_v1/07_nmeala_separability_pilot/11_nmeval_transfer/05_formal_m0/01_gaff2_typed_mol2/nmeval_resp_v1_T_AR_chi180.mol2

Formal MOL2 SHA256:
75657d1b38137e9f5783c1b89c4176aa38a8ef3bbc97becb8ece078facae7dea

GAFF2 frcmod:
/mnt/lab/zn/mem-perm/P0_starter_v1/07_nmeala_separability_pilot/11_nmeval_transfer/05_formal_m0/02_gaff2_parameters/nmeval_resp_v1.frcmod

frcmod SHA256:
9dd4fef0648a4aa12745c3d3fd6a148cd62085f03ed79387cb945900186fcfe6

Formal prmtop:
/mnt/lab/zn/mem-perm/P0_starter_v1/07_nmeala_separability_pilot/11_nmeval_transfer/05_formal_m0/03_tleap_topology/nmeval_resp_v1.prmtop

prmtop SHA256:
d05ec43286f24efdc52f003595d47a355f59080537971a0aed6c819dc491851e

Formal inpcrd:
/mnt/lab/zn/mem-perm/P0_starter_v1/07_nmeala_separability_pilot/11_nmeval_transfer/05_formal_m0/03_tleap_topology/nmeval_resp_v1.inpcrd

inpcrd SHA256:
d5f64dfa1387c2c01079dc55bf71ef4f88873217ba8c89184837525ef51f4e56

Topology net charge:
-0.0000000042 e

Maximum topology-vs-authoritative RESP charge delta:
2.134746935e-09 e

## Carrier zero-step M0 sanity test

No MM relaxation:
maxcyc=0
ncyc=0

Total M0 energy:
+89.676000 kcal/mol

BOND:
+0.930000

ANGLE:
+3.071000

DIHED:
+42.670200

VDWAALS:
+87.277000

EEL:
-28.739400

HBOND:
+0.000000

1-4 VDW:
+6.418900

1-4 EEL:
-21.951700

RESTRAINT:
+0.000000

Component sum:
+89.676000 kcal/mol

Component minus printed total:
+0.00000000 kcal/mol

A setup-only diagnostic line contains:
Direct force subcell size = NaN NaN NaN

It occurs before the RESULTS section.

The actual RESULTS section contains no NaN, Inf, or
overflow field.

The earlier QC failure was caused first by treating the
setup diagnostic as an energy failure, and then by
substring-ambiguous parsing in which BOND matched HBOND
and EEL matched 1-4 EEL.

The corrected whole-line parser reproduces the printed
total energy exactly within the printed precision.

## Scope

This carrier energy is an execution/topology sanity check.

Its absolute value is not a transfer-test result.

The formal NMe-Val M0 is frozen for subsequent
same-geometry QM-minus-M0 residual calculations.

No M1/M2/M3 correction has been fitted.

No NMe-Val transfer-state residual has yet been evaluated.
