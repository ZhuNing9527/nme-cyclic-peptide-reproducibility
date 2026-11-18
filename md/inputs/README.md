# Native physical systems and starting arrays

mono_NMe/ and di_NMe/ contain the original M0/Full OpenMM XML systems,
R1/R2 twelve-replica boxed position/velocity arrays, solute geometry, build report,
reviewed chemical/ROE binding and restraint bounds. Existing tri-NMe files remain
at this directory's root; ALL_SYSTEM_INPUTS_v1.json maps all three systems to
their files and the 12 original formal-run contracts.

These are original starting arrays, not production trajectories. Their bytes
match the precommitted inputs. Coordinates and velocities use nm and nm/ps;
each box matrix uses nm. The six unique arrays contain 72 starting configurations.

Run the NumPy/XML-only audit from the repository root:

```sh
python -B md/code/audit_native_inputs.py --out native_inputs_reproduced.json
```

It verifies hashes, baseline force/constraint equality, all 729 coefficients
per corrected site, D/L signs, chemical graphs, all 71 ROE atom bindings and
the 12 original model/start pins. It evaluates the serialized correction and
the independent Fourier expression on the existing starting coordinates and
checks that J vanishes at trans. No OpenMM import, Context or dynamics is used.

The original constant coefficient is serialized at finite decimal precision;
the maximum observed coefficient difference is about 5.82e-11 kcal/mol and the
maximum per-site energy difference about 2.44e-10 kJ/mol. These are implementation
checks, not new production-ensemble estimates. A only remains a tri-NMe ablation.
