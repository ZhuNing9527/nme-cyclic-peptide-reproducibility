from pathlib import Path
import csv
import hashlib
import math
import os


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


SCHEMA3 = (
    EXE
    / "run_recover_internal_validation144_evaluation_input_schema_v3.py"
)


PRED = (
    EXE
    / "44_budget192_prelabel_evaluator_predictions"
)

PRED_INDEX = (
    PRED
    / "NMEALA_JOINT_BUDGET192_PRELABEL_PREDICTION_INDEX_v1.tsv"
)

PRED_CLOSURE = (
    PRED
    / "NMEALA_JOINT_BUDGET192_PRELABEL_PREDICTION_CLOSURE_FREEZE_v1.sha256"
)


EVAL_PRECOMMIT_DIR = (
    DEV
    / "08_acquisition_evaluation_precommit"
)

EVAL_PRECOMMIT = (
    EVAL_PRECOMMIT_DIR
    / "NMEALA_JOINT_ACQUISITION_EVALUATION_PRECOMMIT_v1.md"
)

EVAL_PRECOMMIT_FREEZE = (
    EVAL_PRECOMMIT_DIR
    / "NMEALA_JOINT_ACQUISITION_EVALUATION_PRECOMMIT_FREEZE_v1.sha256"
)


TRUTH = (
    EXE
    / "47_internal_validation144_m0"
)

LABELS = (
    TRUTH
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_144_LABELS_v1.tsv"
)

TRUTH_CLOSURE = (
    TRUTH
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_LABEL_CLOSURE_FREEZE_v1.sha256"
)


QM = (
    EXE
    / "46_internal_validation144_qm"
)

QM_RESULTS = (
    QM
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_R2SCAN_RESULTS_v1.tsv"
)


OUT = (
    EXE
    / "49_internal_validation144_acquisition_evaluation"
)

PROTOCOL = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_ACQUISITION_EVALUATION_PROTOCOL_v1.md"
)

LOW_ENERGY = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_LOW_ENERGY_MEMBERSHIP_v1.tsv"
)

POINTWISE = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_POINTWISE_ERRORS_v1.tsv"
)

CELL_METRICS = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_CELL_METRICS_v1.tsv"
)

AULC = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_AULC_SUMMARY_v1.tsv"
)

CLAIM = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_ACQUISITION_CLAIM_GATE_v1.tsv"
)

SUMMARY = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_ACQUISITION_EVALUATION_SUMMARY_v1.md"
)

FREEZE = (
    OUT
    / "NMEALA_JOINT_INTERNAL_VALIDATION144_ACQUISITION_EVALUATION_FREEZE_v1.sha256"
)


EXPECTED = {
    SCHEMA3:
        "af0464c8d29ab9d4e8741ca17f8194d61c51f12d3326d94a9cd36a556df7c6af",

    PRED_INDEX:
        "6e69cca666783cd49a557ddd57cee7662e9bc6d76913f1fe9995c3d26119d738",

    PRED_CLOSURE:
        "24dcd9e6f4b18e3c6518c2526ae3af913e42db92d86ee56f701a165165081ba6",

    EVAL_PRECOMMIT:
        "fadce5c50e36f0e5ad61fbdd2a556ad728cb35457a3bd29581a6fa1a25660761",

    EVAL_PRECOMMIT_FREEZE:
        "cb33373cf47620333c369247db77fc3657dfa50d87ac0d88008fc7d65c69fb0c",

    LABELS:
        "363aa8d3161c2765b1d44cff81da05db338fdbf6c8a94e0c59c9f860cdc067dd",

    TRUTH_CLOSURE:
        "c48d1c37467e8efa8659af2b97c2dee9bab9d3fe56e0656da138d532d5cd824c",

    QM_RESULTS:
        "346fc8e6388d70b2e9ea7b6dd9102f482236c8e98a28785940be9c71319e2ed5",
}


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


LABEL_FIELDS = [
    "validation_rank",
    "state_id",
    "phi_deg",
    "psi_deg",
    "omega_deg",
    "QM_energy_Eh",
    "QM_xyz_sha256",
    "M0_energy_component_sum_kcal_mol",
    "absolute_QM_minus_M0_residual_kcal_mol",
    "M0_coordinate_sha256",
    "label_precision_convention",
    "label_status",
]


H2K = 627.509474
LOW_ENERGY_WINDOW = 5.0
TIE_TOL = 1.0e-6

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
    path = Path(
        path
    )

    try:
        return str(
            path.relative_to(
                ROOT
            )
        )

    except ValueError:
        return str(
            path
        )


def resolve(raw):
    path = Path(
        str(
            raw
        ).strip()
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
            list(
                reader.fieldnames
            ),
            list(
                reader
            ),
        )


def write_tsv(
    path,
    fieldnames,
    rows,
):
    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def rehash_freeze(
    path,
    expected_count,
):
    rows = []

    for line in Path(path).read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        recorded, raw = line.split(
            maxsplit=1
        )

        member = resolve(
            raw
        ).resolve()

        if not member.is_file():
            raise RuntimeError(
                "freeze member missing: "
                + str(
                    member
                )
            )

        if sha(
            member
        ) != recorded:
            raise RuntimeError(
                "freeze member changed: "
                + str(
                    member
                )
            )

        rows.append(
            (
                recorded,
                member,
            )
        )

    if len(
        rows
    ) != expected_count:
        raise RuntimeError(
            "freeze count differs: "
            + str(
                path
            )
        )

    return rows


def periodic_error(
    a,
    b,
):
    return abs(
        (
            float(
                a
            )
            -
            float(
                b
            )
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def rmse(values):
    if not values:
        raise RuntimeError(
            "RMSE subset is empty"
        )

    return math.sqrt(
        math.fsum(
            value
            *
            value
            for value in values
        )
        /
        len(
            values
        )
    )


def mae(values):
    if not values:
        raise RuntimeError(
            "MAE subset is empty"
        )

    return (
        math.fsum(
            abs(
                value
            )
            for value in values
        )
        /
        len(
            values
        )
    )


def better_beyond_tolerance(
    candidate,
    comparator,
):
    return (
        candidate
        <
        comparator
        -
        TIE_TOL
    )


print(
    "===== FORMAL FROZEN INTERNAL-VALIDATION144 ACQUISITION EVALUATION ====="
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
        "[FAIL] acquisition-evaluation output already exists"
    )


print(
    "===== 1. VERIFY EXACT CLOSED INPUT AUTHORITIES ====="
)


for path, expected in EXPECTED.items():

    if not path.is_file():
        raise SystemExit(
            "[FAIL] authority missing: "
            + str(
                path
            )
        )

    actual = sha(
        path
    )

    print(
        path.name,
        actual,
    )


    if actual != expected:
        raise SystemExit(
            "[FAIL] authority SHA mismatch: "
            + path.name
        )


prediction_closure_rows = rehash_freeze(
    PRED_CLOSURE,
    63,
)

truth_closure_rows = rehash_freeze(
    TRUTH_CLOSURE,
    883,
)

precommit_rows = rehash_freeze(
    EVAL_PRECOMMIT_FREEZE,
    20,
)


print(
    "[PASS] prediction63 / truth883 / precommit20 closed lineages rehashed"
)


print(
    "===== 2. LOAD EXACT144 CLOSED TRUTH ====="
)


label_fields, label_rows = read_tsv(
    LABELS
)

qm_fields, qm_rows = read_tsv(
    QM_RESULTS
)


if label_fields != LABEL_FIELDS:
    raise SystemExit(
        "[FAIL] label schema changed"
    )


if (
    len(
        label_rows
    )
    !=
    144
    or
    len(
        qm_rows
    )
    !=
    144
):
    raise SystemExit(
        "[FAIL] validation truth rows !=144"
    )


if [
    int(
        row[
            "validation_rank"
        ]
    )
    for row in label_rows
] != list(
    range(
        1,
        145,
    )
):
    raise SystemExit(
        "[FAIL] label ranks !=1..144"
    )


label_ids = [
    row[
        "state_id"
    ]
    for row in label_rows
]


if [
    row[
        "state_id"
    ]
    for row in qm_rows
] != label_ids:
    raise SystemExit(
        "[FAIL] QM/label identity order differs"
    )


truth = {}


for label_row, qm_row in zip(
    label_rows,
    qm_rows,
):

    sid = label_row[
        "state_id"
    ]

    if label_row[
        "label_status"
    ] != "ACCEPTED_INTERNAL_VALIDATION_LABEL":
        raise SystemExit(
            "[FAIL] nonaccepted truth label"
        )


    truth[
        sid
    ] = {
        "rank":
            int(
                label_row[
                    "validation_rank"
                ]
            ),

        "phi":
            float(
                label_row[
                    "phi_deg"
                ]
            ),

        "psi":
            float(
                label_row[
                    "psi_deg"
                ]
            ),

        "omega":
            float(
                label_row[
                    "omega_deg"
                ]
            ),

        "observed":
            float(
                label_row[
                    "absolute_QM_minus_M0_residual_kcal_mol"
                ]
            ),

        "qm_energy":
            float(
                qm_row[
                    "final_energy_Eh"
                ]
            ),
    }


emin = min(
    item[
        "qm_energy"
    ]
    for item in truth.values()
)


for item in truth.values():

    item[
        "delta_qm_kcal"
    ] = (
        item[
            "qm_energy"
        ]
        -
        emin
    ) * H2K


    item[
        "low_energy"
    ] = (
        item[
            "delta_qm_kcal"
        ]
        <=
        LOW_ENERGY_WINDOW
        +
        1.0e-12
    )


    omega = item[
        "omega"
    ]


    if periodic_error(
        omega,
        0.0,
    ) <= 1.0e-9:

        item[
            "omega_region"
        ] = "CIS"

    elif periodic_error(
        omega,
        180.0,
    ) <= 1.0e-9:

        item[
            "omega_region"
        ] = "TRANS"

    else:

        item[
            "omega_region"
        ] = "INTERMEDIATE_OMEGA"


region_counts = {
    region:
        sum(
            item[
                "omega_region"
            ]
            ==
            region
            for item in truth.values()
        )
    for region in (
        "CIS",
        "TRANS",
        "INTERMEDIATE_OMEGA",
    )
}


low_count = sum(
    item[
        "low_energy"
    ]
    for item in truth.values()
)


print(
    "low-energy<=5 count =",
    low_count,
)

print(
    "omega-region counts =",
    region_counts,
)


if low_count <= 0:
    raise SystemExit(
        "[FAIL] frozen low-energy subset is empty"
    )


if region_counts != {
    "CIS":
        12,

    "TRANS":
        12,

    "INTERMEDIATE_OMEGA":
        120,
}:
    raise SystemExit(
        "[FAIL] frozen omega-region counts differ"
    )


print(
    "[PASS] shared low-energy and exact12/12/120 omega subsets established"
)


print(
    "===== 3. LOAD EXACT24 FROZEN PREDICTION CELLS ====="
)


index_fields, index_rows = read_tsv(
    PRED_INDEX
)


if index_fields != INDEX_FIELDS:
    raise SystemExit(
        "[FAIL] prediction-index schema changed"
    )


if len(
    index_rows
) != 24:
    raise SystemExit(
        "[FAIL] prediction-index rows !=24"
    )


cells = []


for index_row in index_rows:

    if index_row[
        "fit_status"
    ] != "PASS":
        raise SystemExit(
            "[FAIL] frozen model cell not PASS"
        )


    prediction_path = Path(
        index_row[
            "prediction_file"
        ]
    ).resolve()


    if sha(
        prediction_path
    ) != index_row[
        "prediction_file_sha256"
    ]:
        raise SystemExit(
            "[FAIL] frozen prediction file changed"
        )


    pred_fields, pred_rows = read_tsv(
        prediction_path
    )


    if pred_fields != PREDICTION_FIELDS:
        raise SystemExit(
            "[FAIL] frozen prediction schema changed"
        )


    if len(
        pred_rows
    ) != 144:
        raise SystemExit(
            "[FAIL] frozen prediction rows !=144"
        )


    if [
        row[
            "state_id"
        ]
        for row in pred_rows
    ] != label_ids:
        raise SystemExit(
            "[FAIL] frozen prediction/truth identity order differs"
        )


    cells.append(
        (
            index_row,
            pred_rows,
        )
    )


print(
    "[PASS] exact24 frozen cells /3456 predictions loaded without refit"
)


print(
    "===== 4. MATERIALIZE EVALUATION OUTPUT BOUNDARY ====="
)


OUT.mkdir(
    parents=True,
    exist_ok=False,
)


PROTOCOL.write_text(
'''# NMe-Ala internal-validation144 frozen acquisition evaluation protocol v1

STATUS:
FROZEN-PREDICTION / CLOSED-TRUTH EVALUATION.

Prediction error:
epsilon = R_predicted - R_observed.

Primary metric:
unweighted RMSE across all144 validation states.

Mandatory secondary metrics:
MAE;
maximum absolute error;
QM-defined <=5 kcal/mol low-energy RMSE;
cis RMSE;
trans RMSE;
intermediate-omega RMSE.

Low-energy definition:
(E_QM - Emin_validation) * 627.509474 <= 5.0 kcal/mol.

Omega regions:
cis = omega 0 degrees;
trans = omega 180 degrees;
intermediate = all remaining locked validation states.

Budgets:
24, 48, 96, 192 new labels.

AULC:
x = log2(budget/24) = 0,1,2,3;
normalized trapezoidal AULC =
[0.5*(R24+R48)+0.5*(R48+R96)+0.5*(R96+R192)]/3.

Tie tolerance:
1e-6 kcal/mol.

FOURIER_ADVANTAGE:
GP Fourier AULC and B192 RMSE must each beat both Maximin and Random beyond tie tolerance.

GP_EVALUATOR_ADVANTAGE:
same gate using periodic-GP evaluator.

Decision mapping:
both TRUE -> CROSS_MODEL_ACQUISITION_ADVANTAGE_SUPPORTED;
Fourier only -> ORTHOGONAL_FOURIER_ACQUISITION_ADVANTAGE_ONLY;
periodic-GP only -> GP_MODEL_MATCHED_ADVANTAGE_ONLY;
both FALSE -> NO_ROBUST_ACQUISITION_ADVANTAGE_SUPPORTED.

Boundary:
no evaluator refit;
no retuning;
no prediction regeneration;
one deterministic acquisition trajectory per strategy;
no p-value claim;
PPII remains sealed.
''',
    encoding="utf-8",
)


print(
    "===== 5. WRITE SHARED LOW-ENERGY / OMEGA MEMBERSHIP ====="
)


low_rows = []


for sid in label_ids:

    item = truth[
        sid
    ]


    low_rows.append(
        {
            "validation_rank":
                str(
                    item[
                        "rank"
                    ]
                ),

            "state_id":
                sid,

            "QM_energy_Eh":
                f"{item['qm_energy']:.12f}",

            "delta_QM_from_validation_min_kcal_mol":
                f"{item['delta_qm_kcal']:.12f}",

            "low_energy_le5_kcal_mol":
                str(
                    item[
                        "low_energy"
                    ]
                ),

            "omega_region":
                item[
                    "omega_region"
                ],
        }
    )


write_tsv(
    LOW_ENERGY,
    [
        "validation_rank",
        "state_id",
        "QM_energy_Eh",
        "delta_QM_from_validation_min_kcal_mol",
        "low_energy_le5_kcal_mol",
        "omega_region",
    ],
    low_rows,
)


print(
    "[PASS] shared validation membership surface written"
)


print(
    "===== 6. COMPUTE EXACT24 PRECOMMITTED METRIC CELLS ====="
)


pointwise_rows = []
metric_rows = []

metric_lookup = {}


for index_row, pred_rows in cells:

    arm = index_row[
        "arm"
    ]

    budget = int(
        index_row[
            "budget_new_labels"
        ]
    )

    evaluator = index_row[
        "evaluator_class"
    ]


    errors = []
    low_errors = []
    cis_errors = []
    trans_errors = []
    intermediate_errors = []


    worst_state = None
    worst_abs = -1.0


    for pred_row in pred_rows:

        sid = pred_row[
            "state_id"
        ]

        item = truth[
            sid
        ]


        if pred_row[
            "prediction_status"
        ] != "PASS":
            raise SystemExit(
                "[FAIL] non-PASS prediction row"
            )


        if periodic_error(
            pred_row[
                "phi_target_deg"
            ],
            item[
                "phi"
            ],
        ) > 1.0e-9:
            raise SystemExit(
                "[FAIL] phi coordinate mismatch"
            )


        if periodic_error(
            pred_row[
                "psi_target_deg"
            ],
            item[
                "psi"
            ],
        ) > 1.0e-9:
            raise SystemExit(
                "[FAIL] psi coordinate mismatch"
            )


        if periodic_error(
            pred_row[
                "omega_target_deg"
            ],
            item[
                "omega"
            ],
        ) > 1.0e-9:
            raise SystemExit(
                "[FAIL] omega coordinate mismatch"
            )


        predicted = float(
            pred_row[
                "predicted_R_kcal_mol"
            ]
        )

        observed = item[
            "observed"
        ]

        error = (
            predicted
            -
            observed
        )

        absolute_error = abs(
            error
        )


        if not (
            math.isfinite(
                predicted
            )
            and
            math.isfinite(
                observed
            )
            and
            math.isfinite(
                error
            )
        ):
            raise SystemExit(
                "[FAIL] nonfinite evaluation quantity"
            )


        errors.append(
            error
        )


        if item[
            "low_energy"
        ]:
            low_errors.append(
                error
            )


        region = item[
            "omega_region"
        ]


        if region == "CIS":
            cis_errors.append(
                error
            )

        elif region == "TRANS":
            trans_errors.append(
                error
            )

        else:
            intermediate_errors.append(
                error
            )


        if absolute_error > worst_abs:

            worst_abs = absolute_error
            worst_state = sid


        pointwise_rows.append(
            {
                "fit_index":
                    index_row[
                        "fit_index"
                    ],

                "arm":
                    arm,

                "budget_new_labels":
                    str(
                        budget
                    ),

                "evaluator_class":
                    evaluator,

                "validation_rank":
                    pred_row[
                        "validation_rank"
                    ],

                "state_id":
                    sid,

                "predicted_R_kcal_mol":
                    f"{predicted:.17g}",

                "observed_R_kcal_mol":
                    f"{observed:.12f}",

                "prediction_error_kcal_mol":
                    f"{error:.12f}",

                "absolute_error_kcal_mol":
                    f"{absolute_error:.12f}",

                "low_energy_le5_kcal_mol":
                    str(
                        item[
                            "low_energy"
                        ]
                    ),

                "omega_region":
                    region,
            }
        )


    metrics = {
        "rmse":
            rmse(
                errors
            ),

        "mae":
            mae(
                errors
            ),

        "maxabs":
            max(
                abs(
                    value
                )
                for value in errors
            ),

        "low_rmse":
            rmse(
                low_errors
            ),

        "cis_rmse":
            rmse(
                cis_errors
            ),

        "trans_rmse":
            rmse(
                trans_errors
            ),

        "intermediate_rmse":
            rmse(
                intermediate_errors
            ),
    }


    metric_lookup[
        (
            arm,
            budget,
            evaluator,
        )
    ] = metrics


    metric_rows.append(
        {
            "fit_index":
                index_row[
                    "fit_index"
                ],

            "arm":
                arm,

            "budget_new_labels":
                str(
                    budget
                ),

            "evaluator_class":
                evaluator,

            "validation_count":
                "144",

            "low_energy_count":
                str(
                    low_count
                ),

            "cis_count":
                "12",

            "trans_count":
                "12",

            "intermediate_omega_count":
                "120",

            "RMSE_kcal_mol":
                f"{metrics['rmse']:.12f}",

            "MAE_kcal_mol":
                f"{metrics['mae']:.12f}",

            "MaxAbs_kcal_mol":
                f"{metrics['maxabs']:.12f}",

            "low_energy_RMSE_kcal_mol":
                f"{metrics['low_rmse']:.12f}",

            "cis_RMSE_kcal_mol":
                f"{metrics['cis_rmse']:.12f}",

            "trans_RMSE_kcal_mol":
                f"{metrics['trans_rmse']:.12f}",

            "intermediate_omega_RMSE_kcal_mol":
                f"{metrics['intermediate_rmse']:.12f}",

            "worst_state_id":
                worst_state,
        }
    )


    print(
        "METRIC",
        arm,
        "B"
        + str(
            budget
        ),
        evaluator,
        "RMSE=",
        f"{metrics['rmse']:.12f}",
    )


if len(
    pointwise_rows
) != 3456:
    raise SystemExit(
        "[FAIL] pointwise rows !=3456"
    )


if len(
    metric_rows
) != 24:
    raise SystemExit(
        "[FAIL] metric cells !=24"
    )


write_tsv(
    POINTWISE,
    [
        "fit_index",
        "arm",
        "budget_new_labels",
        "evaluator_class",
        "validation_rank",
        "state_id",
        "predicted_R_kcal_mol",
        "observed_R_kcal_mol",
        "prediction_error_kcal_mol",
        "absolute_error_kcal_mol",
        "low_energy_le5_kcal_mol",
        "omega_region",
    ],
    pointwise_rows,
)


write_tsv(
    CELL_METRICS,
    [
        "fit_index",
        "arm",
        "budget_new_labels",
        "evaluator_class",
        "validation_count",
        "low_energy_count",
        "cis_count",
        "trans_count",
        "intermediate_omega_count",
        "RMSE_kcal_mol",
        "MAE_kcal_mol",
        "MaxAbs_kcal_mol",
        "low_energy_RMSE_kcal_mol",
        "cis_RMSE_kcal_mol",
        "trans_RMSE_kcal_mol",
        "intermediate_omega_RMSE_kcal_mol",
        "worst_state_id",
    ],
    metric_rows,
)


print(
    "[PASS] exact24 cell metrics /3456 pointwise errors materialized"
)


print(
    "===== 7. COMPUTE FROZEN LOG-BUDGET AULC ====="
)


aulc_rows = []
aulc_lookup = {}


for evaluator in EVALUATORS:

    for arm in ARMS:

        r24 = metric_lookup[
            (
                arm,
                24,
                evaluator,
            )
        ][
            "rmse"
        ]

        r48 = metric_lookup[
            (
                arm,
                48,
                evaluator,
            )
        ][
            "rmse"
        ]

        r96 = metric_lookup[
            (
                arm,
                96,
                evaluator,
            )
        ][
            "rmse"
        ]

        r192 = metric_lookup[
            (
                arm,
                192,
                evaluator,
            )
        ][
            "rmse"
        ]


        aulc = (
            0.5
            *
            (
                r24
                +
                r48
            )
            +
            0.5
            *
            (
                r48
                +
                r96
            )
            +
            0.5
            *
            (
                r96
                +
                r192
            )
        ) / 3.0


        aulc_lookup[
            (
                arm,
                evaluator,
            )
        ] = aulc


        aulc_rows.append(
            {
                "arm":
                    arm,

                "evaluator_class":
                    evaluator,

                "RMSE_B24_kcal_mol":
                    f"{r24:.12f}",

                "RMSE_B48_kcal_mol":
                    f"{r48:.12f}",

                "RMSE_B96_kcal_mol":
                    f"{r96:.12f}",

                "RMSE_B192_kcal_mol":
                    f"{r192:.12f}",

                "normalized_log_budget_AULC_kcal_mol":
                    f"{aulc:.12f}",
            }
        )


        print(
            "AULC",
            arm,
            evaluator,
            "=",
            f"{aulc:.12f}",
        )


if len(
    aulc_rows
) != 6:
    raise SystemExit(
        "[FAIL] AULC rows !=6"
    )


write_tsv(
    AULC,
    [
        "arm",
        "evaluator_class",
        "RMSE_B24_kcal_mol",
        "RMSE_B48_kcal_mol",
        "RMSE_B96_kcal_mol",
        "RMSE_B192_kcal_mol",
        "normalized_log_budget_AULC_kcal_mol",
    ],
    aulc_rows,
)


print(
    "[PASS] exact6 arm-by-evaluator AULC rows materialized"
)


print(
    "===== 8. APPLY FROZEN ACQUISITION-ADVANTAGE CLAIM GATE ====="
)


def evaluator_advantage(
    evaluator,
):
    gp_aulc = aulc_lookup[
        (
            "GP",
            evaluator,
        )
    ]

    maximin_aulc = aulc_lookup[
        (
            "MAXIMIN",
            evaluator,
        )
    ]

    random_aulc = aulc_lookup[
        (
            "RANDOM",
            evaluator,
        )
    ]


    gp_b192 = metric_lookup[
        (
            "GP",
            192,
            evaluator,
        )
    ][
        "rmse"
    ]

    maximin_b192 = metric_lookup[
        (
            "MAXIMIN",
            192,
            evaluator,
        )
    ][
        "rmse"
    ]

    random_b192 = metric_lookup[
        (
            "RANDOM",
            192,
            evaluator,
        )
    ][
        "rmse"
    ]


    aulc_gate = (
        better_beyond_tolerance(
            gp_aulc,
            maximin_aulc,
        )
        and
        better_beyond_tolerance(
            gp_aulc,
            random_aulc,
        )
    )


    b192_gate = (
        better_beyond_tolerance(
            gp_b192,
            maximin_b192,
        )
        and
        better_beyond_tolerance(
            gp_b192,
            random_b192,
        )
    )


    return {
        "gp_aulc":
            gp_aulc,

        "maximin_aulc":
            maximin_aulc,

        "random_aulc":
            random_aulc,

        "gp_b192":
            gp_b192,

        "maximin_b192":
            maximin_b192,

        "random_b192":
            random_b192,

        "aulc_gate":
            aulc_gate,

        "b192_gate":
            b192_gate,

        "advantage":
            (
                aulc_gate
                and
                b192_gate
            ),
    }


fourier = evaluator_advantage(
    "FOURIER_K2"
)

gp_eval = evaluator_advantage(
    "PERIODIC_GP"
)


FOURIER_ADVANTAGE = fourier[
    "advantage"
]

GP_EVALUATOR_ADVANTAGE = gp_eval[
    "advantage"
]


if (
    FOURIER_ADVANTAGE
    and
    GP_EVALUATOR_ADVANTAGE
):

    decision = (
        "CROSS_MODEL_ACQUISITION_ADVANTAGE_SUPPORTED"
    )

elif (
    FOURIER_ADVANTAGE
    and
    not GP_EVALUATOR_ADVANTAGE
):

    decision = (
        "ORTHOGONAL_FOURIER_ACQUISITION_ADVANTAGE_ONLY"
    )

elif (
    not FOURIER_ADVANTAGE
    and
    GP_EVALUATOR_ADVANTAGE
):

    decision = (
        "GP_MODEL_MATCHED_ADVANTAGE_ONLY"
    )

else:

    decision = (
        "NO_ROBUST_ACQUISITION_ADVANTAGE_SUPPORTED"
    )


claim_rows = []


for evaluator, result in (
    (
        "FOURIER_K2",
        fourier,
    ),
    (
        "PERIODIC_GP",
        gp_eval,
    ),
):

    claim_rows.append(
        {
            "evaluator_class":
                evaluator,

            "GP_AULC_kcal_mol":
                f"{result['gp_aulc']:.12f}",

            "MAXIMIN_AULC_kcal_mol":
                f"{result['maximin_aulc']:.12f}",

            "RANDOM_AULC_kcal_mol":
                f"{result['random_aulc']:.12f}",

            "GP_B192_RMSE_kcal_mol":
                f"{result['gp_b192']:.12f}",

            "MAXIMIN_B192_RMSE_kcal_mol":
                f"{result['maximin_b192']:.12f}",

            "RANDOM_B192_RMSE_kcal_mol":
                f"{result['random_b192']:.12f}",

            "GP_beats_both_AULC_beyond_1e-6":
                str(
                    result[
                        "aulc_gate"
                    ]
                ),

            "GP_beats_both_B192_RMSE_beyond_1e-6":
                str(
                    result[
                        "b192_gate"
                    ]
                ),

            "evaluator_advantage":
                str(
                    result[
                        "advantage"
                    ]
                ),

            "final_precommitted_decision":
                decision,
        }
    )


write_tsv(
    CLAIM,
    [
        "evaluator_class",
        "GP_AULC_kcal_mol",
        "MAXIMIN_AULC_kcal_mol",
        "RANDOM_AULC_kcal_mol",
        "GP_B192_RMSE_kcal_mol",
        "MAXIMIN_B192_RMSE_kcal_mol",
        "RANDOM_B192_RMSE_kcal_mol",
        "GP_beats_both_AULC_beyond_1e-6",
        "GP_beats_both_B192_RMSE_beyond_1e-6",
        "evaluator_advantage",
        "final_precommitted_decision",
    ],
    claim_rows,
)


print(
    "FOURIER_ADVANTAGE =",
    FOURIER_ADVANTAGE,
)

print(
    "GP_EVALUATOR_ADVANTAGE =",
    GP_EVALUATOR_ADVANTAGE,
)

print(
    "PRECOMMITTED DECISION =",
    decision,
)


print(
    "[PASS] frozen deterministic acquisition claim gate applied exactly once"
)


print(
    "===== 9. WRITE SCIENTIFIC SUMMARY ====="
)


SUMMARY.write_text(
    "\n".join(
        [
            "# NMe-Ala internal-validation144 frozen acquisition evaluation summary v1",
            "",
            "STATUS:",
            "FORMAL FROZEN BENCHMARK EVALUATION COMPLETE; INDEPENDENT NUMERICAL CLOSURE REQUIRED.",
            "",
            "Validation states: 144.",
            "Low-energy <=5 kcal/mol states: "
            + str(
                low_count
            )
            + ".",
            "Cis/trans/intermediate counts: 12/12/120.",
            "",
            "Primary evaluator: orthogonal FOURIER_K2.",
            "Secondary evaluator: PERIODIC_GP.",
            "",
            "FOURIER_ADVANTAGE = "
            + str(
                FOURIER_ADVANTAGE
            ),
            "GP_EVALUATOR_ADVANTAGE = "
            + str(
                GP_EVALUATOR_ADVANTAGE
            ),
            "",
            "PRECOMMITTED DECISION:",
            decision,
            "",
            "Statistical-language boundary:",
            "one deterministic acquisition trajectory per strategy;",
            "no p-value or stochastic-method significance claim.",
            "",
            "Scientific boundary:",
            "no evaluator refit;",
            "no retuning;",
            "no prediction regeneration;",
            "PPII remains sealed.",
            "",
            "AUTHORIZED NEXT:",
            "one independent numerical reconstruction of all pointwise errors, metrics, AULC values, and claim gates before final scientific closure.",
        ]
    )
    + "\n",
    encoding="utf-8",
)


print(
    "===== 10. WRITE FORMAL EVALUATION GENERATION FREEZE ====="
)


output_members = {
    PROTOCOL.resolve(),
    LOW_ENERGY.resolve(),
    POINTWISE.resolve(),
    CELL_METRICS.resolve(),
    AULC.resolve(),
    CLAIM.resolve(),
    SUMMARY.resolve(),
}


external_members = {
    Path(
        __file__
    ).resolve(),

    SCHEMA3.resolve(),

    PRED_INDEX.resolve(),
    PRED_CLOSURE.resolve(),

    EVAL_PRECOMMIT.resolve(),
    EVAL_PRECOMMIT_FREEZE.resolve(),

    LABELS.resolve(),
    TRUTH_CLOSURE.resolve(),

    QM_RESULTS.resolve(),
}


freeze_members = (
    output_members
    |
    external_members
)


print(
    "evaluation freeze entries =",
    len(
        freeze_members
    ),
)


if len(
    freeze_members
) != 16:
    raise SystemExit(
        "[FAIL] evaluation freeze entries !=16"
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
        for path in sorted(
            freeze_members,
            key=lambda path:
                str(
                    path
                ),
        )
    )
    + "\n",
    encoding="utf-8",
)


print(
    "protocol SHA256 =",
    sha(
        PROTOCOL
    ),
)

print(
    "low-energy membership SHA256 =",
    sha(
        LOW_ENERGY
    ),
)

print(
    "pointwise errors SHA256 =",
    sha(
        POINTWISE
    ),
)

print(
    "cell metrics SHA256 =",
    sha(
        CELL_METRICS
    ),
)

print(
    "AULC summary SHA256 =",
    sha(
        AULC
    ),
)

print(
    "claim gate SHA256 =",
    sha(
        CLAIM
    ),
)

print(
    "summary SHA256 =",
    sha(
        SUMMARY
    ),
)

print(
    "evaluation freeze SHA256 =",
    sha(
        FREEZE
    ),
)


print(
    "===== 11. FINAL FORMAL EVALUATION STATE ====="
)


for path, expected in EXPECTED.items():

    if sha(
        path
    ) != expected:
        raise SystemExit(
            "[FAIL] frozen authority changed during evaluation: "
            + path.name
        )


for group in (
    prediction_closure_rows,
    truth_closure_rows,
    precommit_rows,
):

    for recorded, member in group:

        if sha(
            member
        ) != recorded:
            raise SystemExit(
                "[FAIL] closed-lineage member changed during evaluation"
            )


final_files = [
    path
    for path in OUT.iterdir()
    if path.is_file()
]


print(
    "final evaluation directory files =",
    len(
        final_files
    ),
)


if len(
    final_files
) != 8:
    raise SystemExit(
        "[FAIL] final evaluation directory !=8 files"
    )


print(
    "[PASS] FORMAL FROZEN ACQUISITION BENCHMARK EVALUATION COMPLETE"
)

print(
    "[PASS] exact3456 prediction-vs-truth errors generated"
)

print(
    "[PASS] exact24 mandatory metric cells generated"
)

print(
    "[PASS] exact6 AULC rows generated"
)

print(
    "[PASS] frozen Fourier / periodic-GP claim gate applied"
)

print(
    "[PASS] no evaluator refit/retuning/prediction regeneration occurred"
)

print(
    "[PASS] PPII remains SEALED"
)

print(
    "[HOLD] acquisition benchmark result is not yet SCIENTIFICALLY CLOSED"
)

print(
    "[AUTHORIZED-NEXT] one independent post-evaluation numerical reconstruction and final claim closure"
)

print(
    "===== FORMAL FROZEN INTERNAL-VALIDATION144 ACQUISITION EVALUATION END ====="
)
