#!/usr/bin/env python3
"""Check all output tables against the v2.3 reference values."""

from argparse import ArgumentParser
import csv
import math
from pathlib import Path


def compare(a, b):
    with a.open(newline="", encoding="utf-8-sig") as f, b.open(newline="", encoding="utf-8-sig") as g:
        r, s = (csv.reader(q, delimiter="\t") for q in (f, g))
        assert next(r) == next(s), f"{a.name}: columns differ"
        count = 0
        for i, (x, y) in enumerate(zip(r, s, strict=True), start=1):
            for j, (u, v) in enumerate(zip(x, y, strict=True), start=1):
                if u == v:
                    continue
                try:
                    uf, vf = float(u), float(v)
                except ValueError:
                    raise AssertionError(f"{a.name}, row {i}, column {j}: label or identifier differs") from None
                assert math.isclose(uf, vf, abs_tol=1e-8, rel_tol=0.0), (
                    f"{a.name}, row {i}, column {j}: {uf} differs from {vf}"
                )
            count += 1
    print(f"PASS {a.name}: {count} rows")


if __name__ == "__main__":
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("reproduced", type=Path)
    path = ap.parse_args().reproduced
    expected = Path(__file__).resolve().parent.parent / "expected"
    for p in sorted(expected.glob("*.tsv")):
        compare(path / p.name, p)
