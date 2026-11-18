from __future__ import annotations

import numpy as np


K = 2
N_COEFFICIENTS = 25
FIXED_LAMBDA = 1.0e-1

LAMBDA_GRID = 10.0 ** np.arange(
    -8.0,
    4.0 + 0.25,
    0.5,
)

PENALTY_WEIGHTS = np.array(
    [
        0.0,
        1, 1, 1, 1,
        16, 16, 16, 16,
        4, 4, 4, 4,
        1, 1,
        4, 4, 4, 4,
        4, 4, 4, 4,
        16, 16,
    ],
    dtype=float,
)


def design_k2(
    phi_deg,
    psi_deg,
    omega_deg,
):
    """Return the frozen 25-column gauge-fixed K=2 Fourier design."""

    phi = np.deg2rad(
        np.asarray(phi_deg, dtype=float).reshape(-1)
    )
    psi = np.deg2rad(
        np.asarray(psi_deg, dtype=float).reshape(-1)
    )
    omega = np.deg2rad(
        np.asarray(omega_deg, dtype=float).reshape(-1)
    )

    if not (
        phi.size == psi.size == omega.size
    ):
        raise ValueError("angle arrays must have equal length")

    c1 = np.cos(omega) + 1.0
    s1 = np.sin(omega)

    c2 = np.cos(2.0 * omega) - 1.0
    s2 = np.sin(2.0 * omega)

    x = np.column_stack(
        [
            np.ones(phi.size),

            np.cos(phi),
            np.sin(phi),
            np.cos(psi),
            np.sin(psi),

            np.cos(2.0 * phi),
            np.sin(2.0 * phi),
            np.cos(2.0 * psi),
            np.sin(2.0 * psi),

            np.cos(phi) * np.cos(psi),
            np.cos(phi) * np.sin(psi),
            np.sin(phi) * np.cos(psi),
            np.sin(phi) * np.sin(psi),

            c1,
            s1,

            np.cos(phi) * c1,
            np.cos(phi) * s1,
            np.sin(phi) * c1,
            np.sin(phi) * s1,

            np.cos(psi) * c1,
            np.cos(psi) * s1,
            np.sin(psi) * c1,
            np.sin(psi) * s1,

            c2,
            s2,
        ]
    )

    if x.shape[1] != N_COEFFICIENTS:
        raise RuntimeError(
            f"unexpected design width: {x.shape}"
        )

    return x


def fit_ridge(
    x,
    y,
    lam,
):
    """Fit the frozen frequency-weighted ridge objective."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    lam = float(lam)

    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")

    if x.shape[1] != N_COEFFICIENTS:
        raise ValueError(
            f"x must have {N_COEFFICIENTS} columns"
        )

    if x.shape[0] != y.size:
        raise ValueError("x/y row mismatch")

    if not np.all(np.isfinite(x)):
        raise ValueError("nonfinite design matrix")

    if not np.all(np.isfinite(y)):
        raise ValueError("nonfinite target")

    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lambda must be finite and positive")

    n = y.size

    penalty = np.diag(
        np.sqrt(PENALTY_WEIGHTS)
    )

    augmented_x = np.vstack(
        [
            x,
            np.sqrt(n * lam) * penalty,
        ]
    )

    augmented_y = np.concatenate(
        [
            y,
            np.zeros(N_COEFFICIENTS),
        ]
    )

    rank = int(
        np.linalg.matrix_rank(augmented_x)
    )

    if rank != N_COEFFICIENTS:
        raise RuntimeError(
            f"augmented rank={rank}/{N_COEFFICIENTS}"
        )

    singular_values = np.linalg.svd(
        augmented_x,
        compute_uv=False,
    )

    beta = np.linalg.lstsq(
        augmented_x,
        augmented_y,
        rcond=None,
    )[0]

    diagnostics = {
        "augmented_rank": rank,
        "minimum_augmented_singular_value":
            float(singular_values[-1]),
        "augmented_condition_number":
            float(
                singular_values[0]
                / singular_values[-1]
            ),
    }

    return beta, diagnostics


def predict(
    x,
    beta,
):
    x = np.asarray(x, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)

    if beta.size != N_COEFFICIENTS:
        raise ValueError("beta length mismatch")

    return x @ beta
