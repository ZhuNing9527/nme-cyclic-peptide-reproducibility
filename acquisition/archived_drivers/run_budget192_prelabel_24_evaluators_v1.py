from pathlib import Path
from collections import Counter

import csv
import hashlib
import importlib.util
import json
import math
import os

import numpy as np


ROOT = Path(
    "/mnt/lab/zn/mem-perm/P0_starter_v1"
)

DEV = (
    ROOT
    / "07_nmeala_separability_pilot"
    / "13_joint_correction_development"
)

EXE = (
    DEV
    / "07_acquisition_benchmark_execution"
)


B96 = (
    EXE
    / "22_budget96_authoritative_training_registry"
)

B192 = (
    EXE
    / "42_budget192_authoritative_training_registry"
)


GP_B24 = (
    B96
    / "NMEALA_JOINT_GP_B24_TRAINING_78_v1.tsv"
)

GP_B48 = (
    B96
    / "NMEALA_JOINT_GP_B48_TRAINING_102_v1.tsv"
)

GP_B96 = (
    B96
    / "NMEALA_JOINT_GP_B96_TRAINING_150_v1.tsv"
)

GP_B192 = (
    B192
    / "NMEALA_JOINT_GP_B192_TRAINING_246_v1.tsv"
)


MAX_B24 = (
    B96
    / "NMEALA_JOINT_MAXIMIN_B24_TRAINING_78_v1.tsv"
)

MAX_B48 = (
    B96
    / "NMEALA_JOINT_MAXIMIN_B48_TRAINING_102_v1.tsv"
)

MAX_B96 = (
    B96
    / "NMEALA_JOINT_MAXIMIN_B96_TRAINING_150_v1.tsv"
)

MAX_B192 = (
    B192
    / "NMEALA_JOINT_MAXIMIN_B192_TRAINING_246_v1.tsv"
)


RND_B24 = (
    B96
    / "NMEALA_JOINT_RANDOM_B24_TRAINING_78_v1.tsv"
)

RND_B48 = (
    B96
    / "NMEALA_JOINT_RANDOM_B48_TRAINING_102_v1.tsv"
)

RND_B96 = (
    B96
    / "NMEALA_JOINT_RANDOM_B96_TRAINING_150_v1.tsv"
)

RND_B192 = (
    B192
    / "NMEALA_JOINT_RANDOM_B192_TRAINING_246_v1.tsv"
)


B96_FREEZE = (
    B96
    / "NMEALA_JOINT_BUDGET96_AUTHORITATIVE_TRAINING_REGISTRY_FREEZE_v1.sha256"
)


REG_PROTOCOL = (
    B192
    / "NMEALA_JOINT_BUDGET192_AUTHORITATIVE_TRAINING_REGISTRY_PROTOCOL_v1.md"
)

REG_SUMMARY = (
    B192
    / "NMEALA_JOINT_BUDGET192_AUTHORITATIVE_TRAINING_REGISTRY_SUMMARY_v1.md"
)

REG_GEN_FREEZE = (
    B192
    / "NMEALA_JOINT_BUDGET192_AUTHORITATIVE_TRAINING_REGISTRY_FREEZE_v1.sha256"
)

REG_CLOSE_REPORT = (
    B192
    / "NMEALA_JOINT_BUDGET192_AUTHORITATIVE_TRAINING_REGISTRY_CLOSURE_QC_v1.md"
)

REG_CLOSE_FREEZE = (
    B192
    / "NMEALA_JOINT_BUDGET192_AUTHORITATIVE_TRAINING_REGISTRY_CLOSURE_FREEZE_v1.sha256"
)

REG_CLOSE_QC = (
    EXE
    / "run_qc_budget192_authoritative_training_registry_closure_v1.py"
)


EVAL = (
    DEV
    / "08_acquisition_evaluation_precommit"
)

EVAL_PROTOCOL = (
    EVAL
    / "NMEALA_JOINT_ACQUISITION_EVALUATION_PRECOMMIT_v1.md"
)

EVAL_FREEZE = (
    EVAL
    / "NMEALA_JOINT_ACQUISITION_EVALUATION_PRECOMMIT_FREEZE_v1.sha256"
)

FOURIER = (
    EVAL
    / "fourier_k2_v1.py"
)


GP_IMPL = (
    DEV
    / "05_gp_implementation_precommit"
)

GP_PROTOCOL = (
    GP_IMPL
    / "NMEALA_JOINT_PERIODIC_GP_IMPLEMENTATION_PRECOMMIT_v1.md"
)

GP_STARTS = (
    GP_IMPL
    / "NMEALA_JOINT_GP_HYPERPARAMETER_STARTS_v1.tsv"
)

GP_MODULE = (
    GP_IMPL
    / "periodic_gp_v1.py"
)

GP_IMPL_FREEZE = (
    GP_IMPL
    / "NMEALA_JOINT_PERIODIC_GP_IMPLEMENTATION_FREEZE_v1.sha256"
)


VAL = (
    DEV
    / "03_internal_validation_split"
    / "NMEALA_JOINT_144_INTERNAL_VALIDATION_MANIFEST_v1.tsv"
)


OUT = (
    EXE
    / "44_budget192_prelabel_evaluator_predictions"
)

PRED_DIR = (
    OUT
    / "predictions"
)

PROTOCOL = (
    OUT
    / "NMEALA_JOINT_BUDGET192_PRELABEL_EVALUATOR_PROTOCOL_v1.md"
)

FIT_METADATA = (
    OUT
    / "NMEALA_JOINT_BUDGET192_PRELABEL_FIT_METADATA_v1.tsv"
)

PRED_INDEX = (
    OUT
    / "NMEALA_JOINT_BUDGET192_PRELABEL_PREDICTION_INDEX_v1.tsv"
)

SUMMARY = (
    OUT
    / "NMEALA_JOINT_BUDGET192_PRELABEL_EVALUATOR_SUMMARY_v1.md"
)

FREEZE = (
    OUT
    / "NMEALA_JOINT_BUDGET192_PRELABEL_PREDICTION_FREEZE_v1.sha256"
)


EXPECTED = {
    GP_B24:
        "32e61cc03d71f4ece16f15bbd09b9fe4f1b8cb4d6432a54a8ba0a1b43e18a611",

    GP_B48:
        "7ac383b4d4a99f75b6a450d87151ddd58aa969741a4e656a1189dee375955847",

    GP_B96:
        "2793ccffc7e1d81632f736d155e4c8225fe44865bbe1e8f08d92298d596d4812",

    GP_B192:
        "962d396b8f12d1f56eae1aee8c68ac5df33aa6c9623bb0676fd59937d5798ef5",

    MAX_B24:
        "47c2829667b07da5dceb2b7d656319592fecc9be478dd42b54edfdfe9026ccc0",

    MAX_B48:
        "4de9609efa5acee4bec9bb84ce16899a091fd8b6d2a8743290c1fe766a6a0467",

    MAX_B96:
        "1bcb6b97dc93a0b4de3ed4e141405cf01f401366c8f19fefb463e7a738ebf00e",

    MAX_B192:
        "083a15a2931590d8452be96f1f7886c24f00f1cce93f30c71817b15c6d576ad0",

    RND_B24:
        "f943b3c5c8d3a1d929f44285b7cd85637680dbbb5d8b4c3ec1cbdf26fe8b32af",

    RND_B48:
        "937d69293b1aa1ce55bc272377b4282b7ed6eb18c876b18d3b5754b42af0303b",

    RND_B96:
        "9bbbcc04f875e110ec23cc07bb8f6e29290e644128f06bfcb702deb1ae8dd89e",

    RND_B192:
        "7fe745bdfa6b3c79ea5b6869240cfc4932d8cfbbae9c6e10f3b2addaecc6cad6",

    B96_FREEZE:
        "049b72ce99f3e9b9d8d0114bbe773eee68dcc396d68ed724593cea6dffa8cade",

    REG_PROTOCOL:
        "6a945d030f9be6de714f327e21cc3307cac8728fccf00548ecc97b1820513480",

    REG_SUMMARY:
        "135cfa778d68abf5040681fafd23f9dbddff1c6ddf9ee0274f4ee2d0c685fb6b",

    REG_GEN_FREEZE:
        "a4898c3f9a9411d03d85df7d040227f8400eff6600d1265ac928bd255bb83eca",

    REG_CLOSE_REPORT:
        "4ff9b084c675e45c61999c5e96a28e285d97489526c67bf90acbd19d7d846af6",

    REG_CLOSE_FREEZE:
        "0f6557d63316f74205690f0566506ea317e6ab62fbe623fa478aef37e6bb40a7",

    REG_CLOSE_QC:
        "9d693ed4fe550c64b6f8b2e98aa2938d2c5f00d27dfd2ddff7ac97bd5880cac4",

    EVAL_PROTOCOL:
        "fadce5c50e36f0e5ad61fbdd2a556ad728cb35457a3bd29581a6fa1a25660761",

    EVAL_FREEZE:
        "cb33373cf47620333c369247db77fc3657dfa50d87ac0d88008fc7d65c69fb0c",

    FOURIER:
        "b29da26e24f749c38dcfb7849482115577fa634d16f2527fe04ffdf057c77bb8",

    GP_PROTOCOL:
        "ecadccb8150cfbb48b8d85677752b5345e6e90e0372a34a3fda580341362a669",

    GP_STARTS:
        "2c44f96c13879ff5bf9af880cdfb72161537716ad4a9dac89972c024f07b0d45",

    GP_MODULE:
        "9dac884ab37aa03e45dd9156a9fa4b47adf715d8fc58b6c77057f5db91e54e5c",

    GP_IMPL_FREEZE:
        "45d6950f7608163fdaa92842d4909fb19dafb9261c4a73c40eb862ef66b61d1e",

    VAL:
        "f93fc7c2b2348f1629ad41a9327d66618f90d46b31520062eed20bdf876f7f66",
}


TRAINING = {
    (
        "GP",
        24,
    ):
        (
            GP_B24,
            78,
        ),

    (
        "GP",
        48,
    ):
        (
            GP_B48,
            102,
        ),

    (
        "GP",
        96,
    ):
        (
            GP_B96,
            150,
        ),

    (
        "GP",
        192,
    ):
        (
            GP_B192,
            246,
        ),

    (
        "MAXIMIN",
        24,
    ):
        (
            MAX_B24,
            78,
        ),

    (
        "MAXIMIN",
        48,
    ):
        (
            MAX_B48,
            102,
        ),

    (
        "MAXIMIN",
        96,
    ):
        (
            MAX_B96,
            150,
        ),

    (
        "MAXIMIN",
        192,
    ):
        (
            MAX_B192,
            246,
        ),

    (
        "RANDOM",
        24,
    ):
        (
            RND_B24,
            78,
        ),

    (
        "RANDOM",
        48,
    ):
        (
            RND_B48,
            102,
        ),

    (
        "RANDOM",
        96,
    ):
        (
            RND_B96,
            150,
        ),

    (
        "RANDOM",
        192,
    ):
        (
            RND_B192,
            246,
        ),
}


ARMS = (
    "GP",
    "MAXIMIN",
    "RANDOM",
)

BUDGETS = (
    24,
    48,
    96,
    192,
)

EVALUATORS = (
    "FOURIER_K2",
    "PERIODIC_GP",
)

PRECISION = (
    "PMEMD_PRINTED_NINE_COMPONENT_SUM"
)

PREDICTION_SERIALIZATION = (
    ".17g"
)


FIT_FIELDS = [
    "fit_index",
    "arm",
    "budget_new_labels",
    "evaluator_class",
    "training_rows",
    "training_manifest",
    "training_manifest_sha256",
    "fit_status",
    "target_mean_kcal_mol",
    "selected_start_index",
    "selected_nll",
    "sigma_f",
    "ell_phi",
    "ell_psi",
    "ell_omega",
    "fourier_K",
    "fourier_lambda",
    "fourier_coefficients_json",
    "fit_diagnostics_json",
    "prediction_file",
    "prediction_file_sha256",
]


PREDICTION_FIELDS = [
    "validation_rank",
    "state_id",
    "phi_target_deg",
    "psi_target_deg",
    "omega_target_deg",
    "arm",
    "budget_new_labels",
    "evaluator_class",
    "prediction_status",
    "predicted_R_kcal_mol",
]


INDEX_FIELDS = [
    "fit_index",
    "arm",
    "budget_new_labels",
    "evaluator_class",
    "fit_status",
    "prediction_rows",
    "prediction_file",
    "prediction_file_sha256",
]


def sha(path):
    h = hashlib.sha256()

    with Path(path).open(
        "rb"
    ) as fh:

        for block in iter(
            lambda: fh.read(
                1048576
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def rel(path):
    path = Path(path)

    try:
        return str(
            path.relative_to(
                ROOT
            )
        )

    except ValueError:
        return str(path)


def resolve(raw):
    path = Path(
        str(raw).strip()
    )

    if path.is_absolute():
        return path

    return ROOT / path


def read_tsv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fh:

        reader = csv.DictReader(
            fh,
            delimiter="\t",
        )

        return (
            list(reader.fieldnames),
            list(reader),
        )


def write_tsv(
    path,
    fields,
    rows,
):
    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(rows)


def rehash_freeze(
    path,
    expected_count=None,
):
    rows = []

    for line in Path(path).read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        parts = line.split(
            maxsplit=1
        )

        if len(parts) != 2:
            raise RuntimeError(
                "malformed freeze line: "
                + str(path)
            )

        recorded, raw_path = parts

        member = resolve(
            raw_path
        ).resolve()

        if not member.is_file():
            raise RuntimeError(
                "freeze member missing: "
                + str(member)
            )

        if sha(member) != recorded:
            raise RuntimeError(
                "freeze member changed: "
                + str(member)
            )

        rows.append(
            (
                recorded,
                member,
            )
        )

    if expected_count is not None:
        if len(rows) != expected_count:
            raise RuntimeError(
                "freeze entry count differs for "
                + str(path)
            )

    if len({
        member
        for _, member in rows
    }) != len(rows):
        raise RuntimeError(
            "duplicate freeze member: "
            + str(path)
        )

    return rows


def load_module(
    name,
    path,
):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "cannot load module: "
            + str(path)
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def numeric_training(
    path,
    expected_rows,
    arm,
    budget,
    validation_ids,
):
    fields, rows = read_tsv(
        path
    )

    required = {
        "state_id",
        "training_origin",
        "arm",
        "phi_deg",
        "psi_deg",
        "omega_deg",
        "absolute_QM_minus_M0_residual_kcal_mol",
        "label_precision_convention",
        "label_source_file",
        "label_source_file_sha256",
    }

    missing = (
        required
        -
        set(fields)
    )

    if missing:
        raise RuntimeError(
            "training fields missing: "
            + repr(
                sorted(missing)
            )
        )

    if len(rows) != expected_rows:
        raise RuntimeError(
            arm
            + " B"
            + str(budget)
            + " rows differ"
        )

    ids = [
        row[
            "state_id"
        ]
        for row in rows
    ]

    if len(set(ids)) != expected_rows:
        raise RuntimeError(
            arm
            + " B"
            + str(budget)
            + " state IDs not unique"
        )

    if set(ids) & validation_ids:
        raise RuntimeError(
            arm
            + " B"
            + str(budget)
            + " validation identity leakage"
        )

    if {
        row[
            "arm"
        ]
        for row in rows
    } != {
        arm
    }:
        raise RuntimeError(
            arm
            + " B"
            + str(budget)
            + " arm field differs"
        )

    if any(
        row[
            "label_precision_convention"
        ]
        != PRECISION
        for row in rows
    ):
        raise RuntimeError(
            arm
            + " B"
            + str(budget)
            + " precision differs"
        )

    X = np.asarray(
        [
            [
                float(
                    row[
                        "phi_deg"
                    ]
                ),
                float(
                    row[
                        "psi_deg"
                    ]
                ),
                float(
                    row[
                        "omega_deg"
                    ]
                ),
            ]
            for row in rows
        ],
        dtype=float,
    )

    y = np.asarray(
        [
            float(
                row[
                    "absolute_QM_minus_M0_residual_kcal_mol"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )

    if not np.all(
        np.isfinite(X)
    ):
        raise RuntimeError(
            "nonfinite training coordinates"
        )

    if not np.all(
        np.isfinite(y)
    ):
        raise RuntimeError(
            "nonfinite training residual"
        )

    return (
        rows,
        X,
        y,
    )


def safe_float(
    value,
):
    if value == "":
        return ""

    value = float(value)

    if not math.isfinite(value):
        raise RuntimeError(
            "nonfinite fit metadata value"
        )

    return format(
        value,
        ".17g",
    )


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        allow_nan=False,
    )


print(
    "===== FORMAL B192 24-EVALUATOR PRE-LABEL FIT / PREDICTION RUNNER ====="
)

print(
    "CONDA_DEFAULT_ENV =",
    os.environ.get(
        "CONDA_DEFAULT_ENV"
    ),
)

print(
    "PYTHONPATH =",
    os.environ.get(
        "PYTHONPATH"
    ),
)

if os.environ.get(
    "CONDA_DEFAULT_ENV"
) != "nme-torsion-p0":
    raise SystemExit(
        "[FAIL] wrong environment"
    )

if os.environ.get(
    "PYTHONPATH"
) is not None:
    raise SystemExit(
        "[FAIL] PYTHONPATH must be unset"
    )

if OUT.exists():
    raise SystemExit(
        "[FAIL] pre-label evaluator output already exists"
    )


print(
    "===== 1. VERIFY EXACT FROZEN AUTHORITIES ====="
)

for path, expected in EXPECTED.items():

    if not path.is_file():
        raise SystemExit(
            "[FAIL] required authority missing: "
            + str(path)
        )

    actual = sha(path)

    print(path.name)
    print(" expected =", expected)
    print(" actual   =", actual)

    if actual != expected:
        raise SystemExit(
            "[FAIL] authority SHA mismatch: "
            + str(path)
        )

print(
    "[PASS] exact B24/B48/B96/B192/evaluator/validation authorities verified"
)


print(
    "===== 2. REHASH CLOSED TRAINING / EVALUATION / GP IMPLEMENTATION FREEZES ====="
)

rehash_freeze(
    B96_FREEZE,
    40,
)

rehash_freeze(
    REG_CLOSE_FREEZE,
    59,
)

rehash_freeze(
    EVAL_FREEZE,
    20,
)

gp_impl_freeze_rows = rehash_freeze(
    GP_IMPL_FREEZE
)

print(
    "GP implementation freeze entries =",
    len(
        gp_impl_freeze_rows
    ),
)

print(
    "[PASS] B96/B192/evaluation/GP implementation frozen lineages rehashed"
)


print(
    "===== 3. LOAD VALIDATION COORDINATES ONLY ====="
)

val_fields, val_rows = read_tsv(
    VAL
)

if len(val_rows) != 144:
    raise SystemExit(
        "[FAIL] validation rows !=144"
    )

if len({
    row[
        "state_id"
    ]
    for row in val_rows
}) != 144:
    raise SystemExit(
        "[FAIL] validation state IDs not unique144"
    )

for required in (
    "state_id",
    "phi_target_deg",
    "psi_target_deg",
    "omega_target_deg",
):
    if required not in val_fields:
        raise SystemExit(
            "[FAIL] validation coordinate field missing: "
            + required
        )

label_like = [
    field
    for field in val_fields
    if any(
        token
        in field.lower()
        for token in (
            "residual",
            "qm_energy",
            "m0_energy",
            "label_status",
        )
    )
]

print(
    "validation label-like fields =",
    label_like
)

if label_like:
    raise SystemExit(
        "[FAIL] validation manifest exposes label-like field"
    )

Xval = np.asarray(
    [
        [
            float(
                row[
                    "phi_target_deg"
                ]
            ),
            float(
                row[
                    "psi_target_deg"
                ]
            ),
            float(
                row[
                    "omega_target_deg"
                ]
            ),
        ]
        for row in val_rows
    ],
    dtype=float,
)

if not np.all(
    np.isfinite(Xval)
):
    raise SystemExit(
        "[FAIL] nonfinite validation coordinate"
    )

validation_ids = {
    row[
        "state_id"
    ]
    for row in val_rows
}

print(
    "[PASS] exact144 locked validation coordinates loaded with zero target labels"
)


print(
    "===== 4. VERIFY EXACT 12 TRAINING MANIFESTS ====="
)

training_cache = {}

for arm in ARMS:

    for budget in BUDGETS:

        path, expected_rows = TRAINING[
            (
                arm,
                budget,
            )
        ]

        if sha(
            path
        ) != EXPECTED[
            path
        ]:
            raise SystemExit(
                "[FAIL] training manifest SHA changed"
            )

        rows, X, y = numeric_training(
            path,
            expected_rows,
            arm,
            budget,
            validation_ids,
        )

        training_cache[
            (
                arm,
                budget,
            )
        ] = (
            path,
            rows,
            X,
            y,
        )

        print(
            arm,
            "B"
            + str(budget),
            "rows =",
            expected_rows,
            "residual range =",
            format(
                float(
                    np.min(y)
                ),
                ".6f",
            ),
            "to",
            format(
                float(
                    np.max(y)
                ),
                ".6f",
            ),
        )

print(
    "[PASS] exact12 frozen arm-by-budget training inputs loaded"
)


print(
    "===== 5. LOAD / VERIFY FROZEN EVALUATOR IMPLEMENTATIONS ====="
)

fourier = load_module(
    "nmeala_fourier_k2_evaluator",
    FOURIER,
)

gp = load_module(
    "nmeala_periodic_gp_evaluator",
    GP_MODULE,
)

if fourier.K != 2:
    raise SystemExit(
        "[FAIL] Fourier K !=2"
    )

if fourier.N_COEFFICIENTS != 25:
    raise SystemExit(
        "[FAIL] Fourier coefficient count !=25"
    )

if float(
    fourier.FIXED_LAMBDA
) != 0.1:
    raise SystemExit(
        "[FAIL] Fourier lambda !=0.1"
    )

for function_name in (
    "design_k2",
    "fit_ridge",
    "predict",
):
    if not callable(
        getattr(
            fourier,
            function_name,
            None,
        )
    ):
        raise SystemExit(
            "[FAIL] Fourier API missing "
            + function_name
        )

for function_name in (
    "fit_hyperparameters",
    "posterior_mean",
):
    if not callable(
        getattr(
            gp,
            function_name,
            None,
        )
    ):
        raise SystemExit(
            "[FAIL] periodic-GP API missing "
            + function_name
        )

if not math.isclose(
    float(
        gp.NUGGET
    ),
    1.0e-8,
    rel_tol=0.0,
    abs_tol=0.0,
):
    raise SystemExit(
        "[FAIL] periodic-GP nugget changed"
    )

if tuple(
    float(value)
    for value in gp.SIGMA_BOUNDS
) != (
    0.1,
    20.0,
):
    raise SystemExit(
        "[FAIL] periodic-GP sigma bounds changed"
    )

if tuple(
    float(value)
    for value in gp.ELL_BOUNDS
) != (
    0.15,
    3.0,
):
    raise SystemExit(
        "[FAIL] periodic-GP ell bounds changed"
    )

if len(
    gp.ELL_STARTS
) != 17:
    raise SystemExit(
        "[FAIL] periodic-GP deterministic starts !=17"
    )

starts_fields, starts_rows = read_tsv(
    GP_STARTS
)

print(
    "frozen GP starts table rows =",
    len(
        starts_rows
    ),
)

if len(
    starts_rows
) != 17:
    raise SystemExit(
        "[FAIL] frozen GP starts TSV rows !=17"
    )

print(
    "[PASS] Fourier K2/lambda0.1 and periodic-GP 17-start implementation exact"
)


print(
    "===== 6. MATERIALIZE FORMAL PRE-LABEL OUTPUT BOUNDARY ====="
)

OUT.mkdir(
    parents=True,
    exist_ok=False,
)

PRED_DIR.mkdir(
    parents=False,
    exist_ok=False,
)

protocol_lines = [
    "# NMe-Ala B192 pre-label 24-evaluator fit/prediction protocol v1",
    "",
    "STATUS: FORMAL PRE-LABEL EXECUTION.",
    "",
    "Purpose:",
    "- fit exactly24 evaluator models =3 acquisition arms x4 budgets x2 evaluator classes;",
    "- generate predictions at all144 locked validation coordinates;",
    "- freeze fit metadata and prediction tables before any validation target label is opened.",
    "",
    "Primary evaluator:",
    "- FOURIER_K2;",
    "- K=2;",
    "- 25 coefficients;",
    "- fixed lambda=0.1;",
    "- frozen frequency-weighted ridge implementation;",
    "- np.linalg.lstsq with rcond=None;",
    "- no arm-specific or budget-specific retuning.",
    "",
    "Secondary evaluator:",
    "- PERIODIC_GP;",
    "- anisotropic periodic squared-exponential kernel;",
    "- fixed nugget=1.0e-8 kcal^2/mol^2;",
    "- fitted sigma_f, ell_phi, ell_psi, ell_omega;",
    "- frozen sigma_f bounds0.10-20.0;",
    "- frozen ell bounds0.15-3.0;",
    "- exact17 deterministic starts;",
    "- deterministic L-BFGS-B negative-log-marginal-likelihood minimization;",
    "- frozen NLL tie rule and lexicographic tie break;",
    "- no random starts or repair.",
    "",
    "GP failure rule:",
    "- if all17 frozen starts fail for a required fit, record GP_EVALUATOR_FIT_FAILED;",
    "- do not increase nugget;",
    "- do not expand bounds;",
    "- do not add starts;",
    "- do not substitute kernel or optimizer;",
    "- do not impute a secondary prediction.",
    "",
    "Training inputs:",
    "- exact frozen B24/B48/B96/B192 logical manifest for each arm;",
    "- training counts78/102/150/246;",
    "- all residuals use PMEMD_PRINTED_NINE_COMPONENT_SUM precision.",
    "",
    "Validation boundary:",
    "- validation manifest supplies state identity and phi/psi/omega coordinates only;",
    "- no validation QM energy is read;",
    "- no validation M0 energy is read;",
    "- no validation residual is read;",
    "- no validation error metric is calculated;",
    "- prediction tables contain predicted_R only;",
    "- PPII remains sealed.",
    "",
    "Prediction serialization:",
    "- finite predicted R values are serialized using Python format .17g;",
    "- later evaluation must use these frozen serialized predictions;",
    "- no prediction may be regenerated after validation labels are opened.",
]

PROTOCOL.write_text(
    "\n".join(
        protocol_lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== 7. EXECUTE EXACT24 FITS / VALIDATION-COORDINATE PREDICTIONS ====="
)

fit_rows = []
index_rows = []
prediction_paths = []

fit_index = 0

gp_failure_count = 0

for arm in ARMS:

    for budget in BUDGETS:

        training_path, training_rows, Xtrain, ytrain = (
            training_cache[
                (
                    arm,
                    budget,
                )
            ]
        )

        for evaluator in EVALUATORS:

            fit_index += 1

            prediction_path = (
                PRED_DIR
                /
                (
                    "NMEALA_JOINT_"
                    + arm
                    + "_B"
                    + str(
                        budget
                    )
                    + "_"
                    + evaluator
                    + "_VALIDATION144_PREDICTIONS_v1.tsv"
                )
            )

            if prediction_path.exists():
                raise RuntimeError(
                    "duplicate prediction output path"
                )

            fit_status = "PASS"

            target_mean = ""
            selected_start_index = ""
            selected_nll = ""
            sigma_f = ""
            ell_phi = ""
            ell_psi = ""
            ell_omega = ""
            fourier_K = ""
            fourier_lambda = ""
            fourier_coefficients_json = ""
            diagnostics_json = ""

            predictions = None


            if evaluator == "FOURIER_K2":

                Xfit = fourier.design_k2(
                    Xtrain[
                        :,
                        0
                    ],
                    Xtrain[
                        :,
                        1
                    ],
                    Xtrain[
                        :,
                        2
                    ],
                )

                beta, diagnostics = fourier.fit_ridge(
                    Xfit,
                    ytrain,
                    fourier.FIXED_LAMBDA,
                )

                Xquery = fourier.design_k2(
                    Xval[
                        :,
                        0
                    ],
                    Xval[
                        :,
                        1
                    ],
                    Xval[
                        :,
                        2
                    ],
                )

                predictions = np.asarray(
                    fourier.predict(
                        Xquery,
                        beta,
                    ),
                    dtype=float,
                ).reshape(
                    -1
                )

                if predictions.size != 144:
                    raise RuntimeError(
                        "Fourier prediction count !=144"
                    )

                if not np.all(
                    np.isfinite(
                        predictions
                    )
                ):
                    raise RuntimeError(
                        "nonfinite Fourier prediction"
                    )

                fourier_K = str(
                    int(
                        fourier.K
                    )
                )

                fourier_lambda = safe_float(
                    fourier.FIXED_LAMBDA
                )

                fourier_coefficients_json = canonical_json(
                    [
                        float(value)
                        for value in beta
                    ]
                )

                diagnostics_json = canonical_json(
                    {
                        key:
                            (
                                int(value)
                                if key
                                ==
                                "augmented_rank"
                                else
                                float(value)
                            )
                        for key, value
                        in diagnostics.items()
                    }
                )


            elif evaluator == "PERIODIC_GP":

                try:

                    fit = gp.fit_hyperparameters(
                        Xtrain,
                        ytrain,
                    )

                except RuntimeError as exc:

                    if (
                        "all deterministic GP hyperparameter starts failed"
                        not in str(exc)
                    ):
                        raise

                    fit_status = (
                        "GP_EVALUATOR_FIT_FAILED"
                    )

                    gp_failure_count += 1

                    diagnostics_json = canonical_json(
                        {
                            "failure":
                                str(exc),
                        }
                    )


                if fit_status == "PASS":

                    best = fit[
                        "best"
                    ]

                    ell = np.asarray(
                        [
                            float(
                                best[
                                    "ell_phi"
                                ]
                            ),
                            float(
                                best[
                                    "ell_psi"
                                ]
                            ),
                            float(
                                best[
                                    "ell_omega"
                                ]
                            ),
                        ],
                        dtype=float,
                    )

                    predictions = np.asarray(
                        gp.posterior_mean(
                            Xtrain,
                            ytrain,
                            Xval,
                            float(
                                best[
                                    "sigma_f"
                                ]
                            ),
                            ell,
                        ),
                        dtype=float,
                    ).reshape(
                        -1
                    )

                    if predictions.size != 144:
                        raise RuntimeError(
                            "GP prediction count !=144"
                        )

                    if not np.all(
                        np.isfinite(
                            predictions
                        )
                    ):
                        raise RuntimeError(
                            "nonfinite GP prediction"
                        )

                    target_mean = safe_float(
                        fit[
                            "target_mean"
                        ]
                    )

                    selected_start_index = str(
                        int(
                            best[
                                "start_index"
                            ]
                        )
                    )

                    selected_nll = safe_float(
                        best[
                            "nll"
                        ]
                    )

                    sigma_f = safe_float(
                        best[
                            "sigma_f"
                        ]
                    )

                    ell_phi = safe_float(
                        best[
                            "ell_phi"
                        ]
                    )

                    ell_psi = safe_float(
                        best[
                            "ell_psi"
                        ]
                    )

                    ell_omega = safe_float(
                        best[
                            "ell_omega"
                        ]
                    )

                    diagnostics_json = canonical_json(
                        {
                            "centered_rms":
                                float(
                                    fit[
                                        "centered_rms"
                                    ]
                                ),

                            "sigma_start":
                                float(
                                    fit[
                                        "sigma_start"
                                    ]
                                ),

                            "successful_starts":
                                sum(
                                    1
                                    for attempt
                                    in fit[
                                        "attempts"
                                    ]
                                    if (
                                        attempt[
                                            "success"
                                        ]
                                        and
                                        math.isfinite(
                                            float(
                                                attempt[
                                                    "nll"
                                                ]
                                            )
                                        )
                                    )
                                ),

                            "attempts":
                                fit[
                                    "attempts"
                                ],
                        }
                    )


            else:
                raise RuntimeError(
                    "unknown evaluator class"
                )


            prediction_rows = []

            for rank, row in enumerate(
                val_rows,
                start=1,
            ):

                predicted_value = ""

                if fit_status == "PASS":

                    predicted_value = format(
                        float(
                            predictions[
                                rank - 1
                            ]
                        ),
                        PREDICTION_SERIALIZATION,
                    )

                prediction_rows.append(
                    {
                        "validation_rank":
                            str(
                                rank
                            ),

                        "state_id":
                            row[
                                "state_id"
                            ],

                        "phi_target_deg":
                            row[
                                "phi_target_deg"
                            ],

                        "psi_target_deg":
                            row[
                                "psi_target_deg"
                            ],

                        "omega_target_deg":
                            row[
                                "omega_target_deg"
                            ],

                        "arm":
                            arm,

                        "budget_new_labels":
                            str(
                                budget
                            ),

                        "evaluator_class":
                            evaluator,

                        "prediction_status":
                            fit_status,

                        "predicted_R_kcal_mol":
                            predicted_value,
                    }
                )


            write_tsv(
                prediction_path,
                PREDICTION_FIELDS,
                prediction_rows,
            )


            prediction_sha = sha(
                prediction_path
            )

            prediction_paths.append(
                prediction_path.resolve()
            )


            fit_rows.append(
                {
                    "fit_index":
                        str(
                            fit_index
                        ),

                    "arm":
                        arm,

                    "budget_new_labels":
                        str(
                            budget
                        ),

                    "evaluator_class":
                        evaluator,

                    "training_rows":
                        str(
                            len(
                                training_rows
                            )
                        ),

                    "training_manifest":
                        str(
                            training_path
                        ),

                    "training_manifest_sha256":
                        EXPECTED[
                            training_path
                        ],

                    "fit_status":
                        fit_status,

                    "target_mean_kcal_mol":
                        target_mean,

                    "selected_start_index":
                        selected_start_index,

                    "selected_nll":
                        selected_nll,

                    "sigma_f":
                        sigma_f,

                    "ell_phi":
                        ell_phi,

                    "ell_psi":
                        ell_psi,

                    "ell_omega":
                        ell_omega,

                    "fourier_K":
                        fourier_K,

                    "fourier_lambda":
                        fourier_lambda,

                    "fourier_coefficients_json":
                        fourier_coefficients_json,

                    "fit_diagnostics_json":
                        diagnostics_json,

                    "prediction_file":
                        str(
                            prediction_path
                        ),

                    "prediction_file_sha256":
                        prediction_sha,
                }
            )


            index_rows.append(
                {
                    "fit_index":
                        str(
                            fit_index
                        ),

                    "arm":
                        arm,

                    "budget_new_labels":
                        str(
                            budget
                        ),

                    "evaluator_class":
                        evaluator,

                    "fit_status":
                        fit_status,

                    "prediction_rows":
                        "144",

                    "prediction_file":
                        str(
                            prediction_path
                        ),

                    "prediction_file_sha256":
                        prediction_sha,
                }
            )


            print(
                "FIT",
                fit_index,
                arm,
                "B"
                + str(
                    budget
                ),
                evaluator,
                "status=",
                fit_status,
                "prediction_sha=",
                prediction_sha,
            )


if fit_index != 24:
    raise RuntimeError(
        "formal evaluator fit count !=24"
    )

if len(
    prediction_paths
) != 24:
    raise RuntimeError(
        "formal prediction-table count !=24"
    )


write_tsv(
    FIT_METADATA,
    FIT_FIELDS,
    fit_rows,
)

write_tsv(
    PRED_INDEX,
    INDEX_FIELDS,
    index_rows,
)


print(
    "[PASS] exact24 model cells executed"
)

print(
    "[PASS] exact24 validation-coordinate prediction tables materialized"
)


print(
    "===== 8. PRE-LABEL OUTPUT SEMANTIC CHECK ====="
)

fit_fields_check, fits_check = read_tsv(
    FIT_METADATA
)

index_fields_check, index_check = read_tsv(
    PRED_INDEX
)

if fit_fields_check != FIT_FIELDS:
    raise RuntimeError(
        "fit metadata schema changed"
    )

if index_fields_check != INDEX_FIELDS:
    raise RuntimeError(
        "prediction index schema changed"
    )

if len(
    fits_check
) != 24:
    raise RuntimeError(
        "fit metadata rows !=24"
    )

if len(
    index_check
) != 24:
    raise RuntimeError(
        "prediction index rows !=24"
    )


expected_cells = [
    (
        arm,
        str(
            budget
        ),
        evaluator,
    )
    for arm in ARMS
    for budget in BUDGETS
    for evaluator in EVALUATORS
]


actual_cells = [
    (
        row[
            "arm"
        ],
        row[
            "budget_new_labels"
        ],
        row[
            "evaluator_class"
        ],
    )
    for row in fits_check
]


if actual_cells != expected_cells:
    raise RuntimeError(
        "fit cell ordering differs"
    )


for path in prediction_paths:

    fields, rows = read_tsv(
        path
    )

    if fields != PREDICTION_FIELDS:
        raise RuntimeError(
            "prediction schema differs: "
            + path.name
        )

    if len(
        rows
    ) != 144:
        raise RuntimeError(
            "prediction rows !=144: "
            + path.name
        )

    if [
        row[
            "state_id"
        ]
        for row in rows
    ] != [
        row[
            "state_id"
        ]
        for row in val_rows
    ]:
        raise RuntimeError(
            "prediction validation order differs: "
            + path.name
        )

    for forbidden in (
        "true_R",
        "observed_R",
        "validation_residual",
        "error",
        "absolute_error",
        "squared_error",
        "rmse",
        "mae",
    ):

        if any(
            forbidden
            in field.lower()
            for field in fields
        ):
            raise RuntimeError(
                "prediction table contains validation metric/target field"
            )


print(
    "GP evaluator fit failures =",
    gp_failure_count,
)

print(
    "[PASS] prediction tables contain no validation target/error metric"
)

print(
    "[PASS] validation/PPII remain SEALED"
)


print(
    "===== 9. WRITE PRE-LABEL SUMMARY ====="
)

fit_status_counts = Counter(
    row[
        "fit_status"
    ]
    for row in fits_check
)

summary_lines = [
    "# NMe-Ala B192 pre-label 24-evaluator summary v1",
    "",
    "STATUS:",
    "PRE-LABEL FITS / VALIDATION-COORDINATE PREDICTIONS GENERATED;",
    "INDEPENDENT POSTEXECUTION CLOSURE REQUIRED BEFORE VALIDATION LABEL OPENING.",
    "",
    "Model cells:",
    "- arms =3;",
    "- budgets =4;",
    "- evaluator classes =2;",
    "- total model cells =24.",
    "",
    "Training sizes:",
    "- B24=78;",
    "- B48=102;",
    "- B96=150;",
    "- B192=246.",
    "",
    "Validation coordinates:",
    "- locked coordinates =144;",
    "- prediction tables =24;",
    "- each prediction table contains144 rows;",
    "- no validation target or error metric is present.",
    "",
    "Fit status counts:",
    repr(
        dict(
            sorted(
                fit_status_counts.items()
            )
        )
    ),
    "",
    "GP_EVALUATOR_FIT_FAILED count:",
    str(
        gp_failure_count
    ),
    "",
    "Boundary:",
    "- no validation label opened;",
    "- no RMSE/MAE/AULC calculated;",
    "- PPII remains sealed;",
    "- validation labels may not be opened until independent closure of this pre-label prediction freeze.",
]

SUMMARY.write_text(
    "\n".join(
        summary_lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    "fit metadata SHA256 =",
    sha(
        FIT_METADATA
    ),
)

print(
    "prediction index SHA256 =",
    sha(
        PRED_INDEX
    ),
)

print(
    "summary SHA256 =",
    sha(
        SUMMARY
    ),
)


print(
    "===== 10. WRITE FORMAL PRE-LABEL PREDICTION FREEZE ====="
)

freeze_members = [
    Path(
        __file__
    ).resolve(),

    B96_FREEZE.resolve(),

    REG_PROTOCOL.resolve(),
    REG_SUMMARY.resolve(),
    REG_GEN_FREEZE.resolve(),
    REG_CLOSE_REPORT.resolve(),
    REG_CLOSE_FREEZE.resolve(),
    REG_CLOSE_QC.resolve(),

    EVAL_PROTOCOL.resolve(),
    EVAL_FREEZE.resolve(),

    FOURIER.resolve(),

    GP_PROTOCOL.resolve(),
    GP_STARTS.resolve(),
    GP_MODULE.resolve(),
    GP_IMPL_FREEZE.resolve(),

    VAL.resolve(),

    GP_B24.resolve(),
    GP_B48.resolve(),
    GP_B96.resolve(),
    GP_B192.resolve(),

    MAX_B24.resolve(),
    MAX_B48.resolve(),
    MAX_B96.resolve(),
    MAX_B192.resolve(),

    RND_B24.resolve(),
    RND_B48.resolve(),
    RND_B96.resolve(),
    RND_B192.resolve(),

    PROTOCOL.resolve(),
    FIT_METADATA.resolve(),
    PRED_INDEX.resolve(),
    SUMMARY.resolve(),
]

freeze_members.extend(
    prediction_paths
)


if len(
    freeze_members
) != len(
    set(
        freeze_members
    )
):
    raise RuntimeError(
        "duplicate formal prediction-freeze member"
    )


print(
    "pre-label prediction freeze entries =",
    len(
        freeze_members
    ),
)


for path in freeze_members:

    if not path.is_file():
        raise RuntimeError(
            "prediction-freeze member missing: "
            + str(
                path
            )
        )


FREEZE.write_text(
    "\n".join(
        sha(
            path
        )
        + "  "
        + rel(
            path
        )
        for path in freeze_members
    )
    + "\n",
    encoding="utf-8",
)


print(
    "pre-label prediction freeze SHA256 =",
    sha(
        FREEZE
    ),
)


print(
    "===== 11. FINAL PRE-LABEL STAGE BOUNDARY ====="
)

for path, expected in EXPECTED.items():

    if sha(
        path
    ) != expected:
        raise RuntimeError(
            "frozen upstream authority changed during evaluator execution: "
            + str(
                path
            )
        )


prediction_files = sorted(
    [
        path
        for path in PRED_DIR.iterdir()
        if path.is_file()
    ],
    key=lambda path:
        path.name,
)


if len(
    prediction_files
) != 24:
    raise RuntimeError(
        "final prediction file count !=24"
    )


top_files = {
    path.name
    for path in OUT.iterdir()
    if path.is_file()
}


expected_top_files = {
    PROTOCOL.name,
    FIT_METADATA.name,
    PRED_INDEX.name,
    SUMMARY.name,
    FREEZE.name,
}


if top_files != expected_top_files:
    raise RuntimeError(
        "unexpected top-level evaluator output file set"
    )


if {
    path.name
    for path in OUT.iterdir()
    if path.is_dir()
} != {
    PRED_DIR.name
}:
    raise RuntimeError(
        "unexpected evaluator output subdirectory"
    )


print(
    "[PASS] FORMAL PRE-LABEL 24-EVALUATOR EXECUTION COMPLETE"
)

print(
    "[PASS] exact12 Fourier + exact12 periodic-GP model cells processed"
)

print(
    "[PASS] exact24 x144 validation-coordinate prediction tables frozen"
)

print(
    "[PASS] no validation target/error metric entered fit or prediction files"
)

print(
    "[PASS] no AULC/RMSE/MAE/regional evaluation calculated"
)

print(
    "[PASS] validation labels remain SEALED"
)

print(
    "[PASS] PPII remains SEALED"
)

print(
    "[HOLD] validation labels remain forbidden until independent prediction-freeze closure"
)

print(
    "[AUTHORIZED-NEXT] independent postexecution reconstruction/closure of pre-label fits and predictions"
)

print(
    "formal evaluator runner SHA256 =",
    sha(
        Path(
            __file__
        )
    ),
)

print(
    "===== FORMAL B192 24-EVALUATOR PRE-LABEL FIT / PREDICTION RUNNER END ====="
)
