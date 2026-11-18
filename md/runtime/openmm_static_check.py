#!/usr/bin/env python3
"""Load a frozen physical System and build its REST2 force definitions without a Context."""

import argparse
import json
from pathlib import Path

import openmm
from rest2_scaling_openmm_v2 import apply_rest2_scaling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("M0", "A_only", "Full"), default="Full")
    a = p.parse_args()
    if openmm.__version__ not in ("8.4", "8.4.0"):
        raise RuntimeError("The original System and REST2 code were qualified with OpenMM 8.4.0")
    root = Path(__file__).resolve().parents[1] / "inputs"
    source = (root / f"{a.arm}.xml").read_text()
    system = openmm.XmlSerializer.deserialize(source)
    scaled, audit = apply_rest2_scaling(system, range(59))
    assert system.getNumParticles() == scaled.getNumParticles() == 4939
    correction = [f for f in scaled.getForces() if f.getForceGroup() == 31]
    if a.arm == "M0":
        assert not correction
    else:
        f, = correction
        assert f.getEnergyFunction().startswith("rest2_lambda*(")
        assert f.getNumBonds() == 2187
    print(json.dumps(audit, indent=2))
    print("PASS serialized System and REST2 transformation. No Context or dynamics.")


if __name__ == "__main__":
    main()
