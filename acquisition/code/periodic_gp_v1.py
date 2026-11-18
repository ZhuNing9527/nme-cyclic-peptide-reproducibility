from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


NUGGET = 1.0e-8

SIGMA_BOUNDS = (
    0.10,
    20.0,
)

ELL_BOUNDS = (
    0.15,
    3.0,
)

OPT_MAXITER = 1000
OPT_MAXLS = 50
OPT_FTOL = 1.0e-12
OPT_GTOL = 1.0e-8

NLL_TIE_TOL = 1.0e-10

SCORE_TIE_TOL = 1.0e-12
DISTANCE_TIE_TOL_DEG = 1.0e-10


ELL_STARTS = (
    (0.25, 0.25, 0.25),
    (0.50, 0.50, 0.50),
    (0.90, 0.90, 0.90),
    (1.50, 1.50, 1.50),
    (2.50, 2.50, 2.50),

    (0.35, 0.70, 1.40),
    (0.35, 1.40, 0.70),
    (0.70, 0.35, 1.40),
    (0.70, 1.40, 0.35),
    (1.40, 0.35, 0.70),
    (1.40, 0.70, 0.35),

    (0.70, 1.40, 2.40),
    (0.70, 2.40, 1.40),
    (1.40, 0.70, 2.40),
    (1.40, 2.40, 0.70),
    (2.40, 0.70, 1.40),
    (2.40, 1.40, 0.70),
)


def as_X_deg(X):

    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2 or X.shape[1] != 3:

        raise ValueError(
            "X must have shape (n,3) in phi/psi/omega order"
        )

    if not np.all(
        np.isfinite(
            X
        )
    ):

        raise ValueError(
            "non-finite angular coordinate"
        )

    return X


def to_radians(X_deg):

    return np.deg2rad(
        as_X_deg(
            X_deg
        )
    )


def kernel_rad(
    X1_rad,
    X2_rad,
    sigma_f,
    ell,
):

    X1 = np.asarray(
        X1_rad,
        dtype=float,
    )

    X2 = np.asarray(
        X2_rad,
        dtype=float,
    )

    ell = np.asarray(
        ell,
        dtype=float,
    )


    if ell.shape != (
        3,
    ):

        raise ValueError(
            "ell must have length 3"
        )


    delta = (
        X1[
            :,
            None,
            :
        ]
        -
        X2[
            None,
            :,
            :
        ]
    )


    periodic_sq = (
        np.sin(
            0.5
            *
            delta
        )
        **
        2
    )


    exponent = (
        -2.0
        *
        np.sum(
            periodic_sq
            /
            (
                ell[
                    None,
                    None,
                    :
                ]
                **
                2
            ),
            axis=2,
        )
    )


    return (
        float(
            sigma_f
        )
        **
        2
    ) * np.exp(
        exponent
    )


def kernel_deg(
    X1_deg,
    X2_deg,
    sigma_f,
    ell,
):

    return kernel_rad(
        to_radians(
            X1_deg
        ),
        to_radians(
            X2_deg
        ),
        sigma_f,
        ell,
    )


def center_targets(
    y,
):

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(
        -1
    )


    if not np.all(
        np.isfinite(
            y
        )
    ):

        raise ValueError(
            "non-finite GP target"
        )


    mean = float(
        np.mean(
            y
        )
    )


    yc = (
        y
        -
        mean
    )


    return (
        mean,
        yc,
    )


def unpack_logtheta(
    logtheta,
):

    p = np.exp(
        np.asarray(
            logtheta,
            dtype=float,
        )
    )


    return (
        float(
            p[
                0
            ]
        ),
        np.asarray(
            p[
                1:
            ],
            dtype=float,
        ),
    )


def negative_log_marginal_likelihood(
    logtheta,
    X_deg,
    y_centered,
):

    X = to_radians(
        X_deg
    )

    y = np.asarray(
        y_centered,
        dtype=float,
    ).reshape(
        -1
    )


    sigma_f,ell = unpack_logtheta(
        logtheta
    )


    K = kernel_rad(
        X,
        X,
        sigma_f,
        ell,
    )


    K = K.copy()

    K[
        np.diag_indices_from(
            K
        )
    ] += NUGGET


    try:

        L = np.linalg.cholesky(
            K
        )

    except np.linalg.LinAlgError:

        return 1.0e100


    z = solve_triangular(
        L,
        y,
        lower=True,
        check_finite=False,
    )


    alpha = solve_triangular(
        L.T,
        z,
        lower=False,
        check_finite=False,
    )


    n = len(
        y
    )


    nll = (
        0.5
        *
        float(
            np.dot(
                y,
                alpha,
            )
        )
        +
        float(
            np.sum(
                np.log(
                    np.diag(
                        L
                    )
                )
            )
        )
        +
        0.5
        *
        n
        *
        math.log(
            2.0
            *
            math.pi
        )
    )


    if not math.isfinite(
        nll
    ):

        return 1.0e100


    return float(
        nll
    )


def fit_hyperparameters(
    X_deg,
    y_absolute,
):

    X = as_X_deg(
        X_deg
    )

    y = np.asarray(
        y_absolute,
        dtype=float,
    ).reshape(
        -1
    )


    if len(
        X
    ) != len(
        y
    ):

        raise ValueError(
            "X/y length mismatch"
        )


    target_mean,yc = center_targets(
        y
    )


    centered_rms = math.sqrt(
        float(
            np.mean(
                yc
                **
                2
            )
        )
    )


    sigma_start = min(
        SIGMA_BOUNDS[
            1
        ],
        max(
            SIGMA_BOUNDS[
                0
            ],
            centered_rms,
        ),
    )


    bounds = [
        (
            math.log(
                SIGMA_BOUNDS[
                    0
                ]
            ),
            math.log(
                SIGMA_BOUNDS[
                    1
                ]
            ),
        ),
        (
            math.log(
                ELL_BOUNDS[
                    0
                ]
            ),
            math.log(
                ELL_BOUNDS[
                    1
                ]
            ),
        ),
        (
            math.log(
                ELL_BOUNDS[
                    0
                ]
            ),
            math.log(
                ELL_BOUNDS[
                    1
                ]
            ),
        ),
        (
            math.log(
                ELL_BOUNDS[
                    0
                ]
            ),
            math.log(
                ELL_BOUNDS[
                    1
                ]
            ),
        ),
    ]


    attempts = []


    for start_index,ells in enumerate(
        ELL_STARTS,
        start=1,
    ):

        x0 = np.log(
            np.asarray(
                (
                    sigma_start,
                    *ells,
                ),
                dtype=float,
            )
        )


        result = minimize(
            negative_log_marginal_likelihood,
            x0,
            args=(
                X,
                yc,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter":
                    OPT_MAXITER,

                "maxls":
                    OPT_MAXLS,

                "ftol":
                    OPT_FTOL,

                "gtol":
                    OPT_GTOL,
            },
        )


        sigma_f,ell = unpack_logtheta(
            result.x
        )


        attempt = {
            "start_index":
                start_index,

            "success":
                bool(
                    result.success
                ),

            "nll":
                float(
                    result.fun
                ),

            "sigma_f":
                sigma_f,

            "ell_phi":
                float(
                    ell[
                        0
                    ]
                ),

            "ell_psi":
                float(
                    ell[
                        1
                    ]
                ),

            "ell_omega":
                float(
                    ell[
                        2
                    ]
                ),

            "nit":
                int(
                    getattr(
                        result,
                        "nit",
                        -1,
                    )
                ),

            "message":
                str(
                    result.message
                ),
        }


        attempts.append(
            attempt
        )


    eligible = [
        a
        for a in attempts
        if (
            a[
                "success"
            ]
            and
            math.isfinite(
                a[
                    "nll"
                ]
            )
        )
    ]


    if not eligible:

        raise RuntimeError(
            "all deterministic GP hyperparameter starts failed"
        )


    best = None


    for a in eligible:

        if best is None:

            best = a
            continue


        if (
            a[
                "nll"
            ]
            <
            best[
                "nll"
            ]
            -
            NLL_TIE_TOL
        ):

            best = a
            continue


        if abs(
            a[
                "nll"
            ]
            -
            best[
                "nll"
            ]
        ) <= NLL_TIE_TOL:

            ta = (
                a[
                    "sigma_f"
                ],
                a[
                    "ell_phi"
                ],
                a[
                    "ell_psi"
                ],
                a[
                    "ell_omega"
                ],
            )

            tb = (
                best[
                    "sigma_f"
                ],
                best[
                    "ell_phi"
                ],
                best[
                    "ell_psi"
                ],
                best[
                    "ell_omega"
                ],
            )

            if ta < tb:

                best = a


    return {
        "target_mean":
            target_mean,

        "centered_rms":
            centered_rms,

        "sigma_start":
            sigma_start,

        "best":
            dict(
                best
            ),

        "attempts":
            attempts,
    }


def _conditioning_factor(
    Xtrain_deg,
    sigma_f,
    ell,
    extra_condition_deg=None,
):

    Xtrain = as_X_deg(
        Xtrain_deg
    )


    if extra_condition_deg is None:

        Xcond = Xtrain

    else:

        Xextra = as_X_deg(
            extra_condition_deg
        )

        Xcond = np.vstack(
            (
                Xtrain,
                Xextra,
            )
        )


    Xcond_rad = to_radians(
        Xcond
    )


    K = kernel_rad(
        Xcond_rad,
        Xcond_rad,
        sigma_f,
        ell,
    )


    K = K.copy()

    K[
        np.diag_indices_from(
            K
        )
    ] += NUGGET


    L = np.linalg.cholesky(
        K
    )


    return (
        Xcond,
        Xcond_rad,
        L,
    )


def posterior_mean(
    Xtrain_deg,
    y_absolute,
    Xquery_deg,
    sigma_f,
    ell,
):

    Xtrain = as_X_deg(
        Xtrain_deg
    )

    Xquery = as_X_deg(
        Xquery_deg
    )

    target_mean,yc = center_targets(
        y_absolute
    )


    Xtrain_rad = to_radians(
        Xtrain
    )

    Xquery_rad = to_radians(
        Xquery
    )


    K = kernel_rad(
        Xtrain_rad,
        Xtrain_rad,
        sigma_f,
        ell,
    )

    K = K.copy()

    K[
        np.diag_indices_from(
            K
        )
    ] += NUGGET


    L = np.linalg.cholesky(
        K
    )


    z = solve_triangular(
        L,
        yc,
        lower=True,
        check_finite=False,
    )

    alpha = solve_triangular(
        L.T,
        z,
        lower=False,
        check_finite=False,
    )


    Kqt = kernel_rad(
        Xquery_rad,
        Xtrain_rad,
        sigma_f,
        ell,
    )


    return (
        target_mean
        +
        Kqt
        @
        alpha
    )


def acquisition_scores(
    Xtrain_deg,
    Xcandidate_deg,
    sigma_f,
    ell,
    extra_condition_deg=None,
):

    Xcandidate = as_X_deg(
        Xcandidate_deg
    )


    (
        Xcond,
        Xcond_rad,
        L,
    ) = _conditioning_factor(
        Xtrain_deg,
        sigma_f,
        ell,
        extra_condition_deg,
    )


    Xcand_rad = to_radians(
        Xcandidate
    )


    Xtrans_deg = Xcandidate.copy()

    Xtrans_deg[
        :,
        2
    ] = 180.0


    Xtrans_rad = to_radians(
        Xtrans_deg
    )


    Kcx = kernel_rad(
        Xcond_rad,
        Xcand_rad,
        sigma_f,
        ell,
    )


    Kct = kernel_rad(
        Xcond_rad,
        Xtrans_rad,
        sigma_f,
        ell,
    )


    Vx = solve_triangular(
        L,
        Kcx,
        lower=True,
        check_finite=False,
    )


    Vt = solve_triangular(
        L,
        Kct,
        lower=True,
        check_finite=False,
    )


    prior_var = (
        float(
            sigma_f
        )
        **
        2
    )


    var_R = (
        prior_var
        -
        np.sum(
            Vx
            *
            Vx,
            axis=0,
        )
    )


    var_Tref = (
        prior_var
        -
        np.sum(
            Vt
            *
            Vt,
            axis=0,
        )
    )


    prior_cov_xt = np.diag(
        kernel_rad(
            Xcand_rad,
            Xtrans_rad,
            sigma_f,
            ell,
        )
    )


    posterior_cov_xt = (
        prior_cov_xt
        -
        np.sum(
            Vx
            *
            Vt,
            axis=0,
        )
    )


    var_R = np.maximum(
        var_R,
        0.0,
    )

    var_Tref = np.maximum(
        var_Tref,
        0.0,
    )


    var_J = (
        var_R
        +
        var_Tref
        -
        2.0
        *
        posterior_cov_xt
    )


    var_J = np.maximum(
        var_J,
        0.0,
    )


    sd_R = np.sqrt(
        var_R
    )

    sd_J = np.sqrt(
        var_J
    )


    score = (
        sd_R
        +
        sd_J
    )


    return {
        "sd_R":
            sd_R,

        "sd_J":
            sd_J,

        "score":
            score,

        "var_R":
            var_R,

        "var_J":
            var_J,
    }


def wrap_deg_delta(
    x,
):

    return (
        (
            x
            +
            180.0
        )
        %
        360.0
        -
        180.0
    )


def periodic_distance_deg(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )


    d = wrap_deg_delta(
        a-b
    )


    return float(
        math.sqrt(
            float(
                np.dot(
                    d,
                    d,
                )
            )
        )
    )


def select_fantasy_batch(
    Xtrain_deg,
    candidate_records,
    sigma_f,
    ell,
    batch_size,
):

    Xtrain = as_X_deg(
        Xtrain_deg
    )


    remaining = {
        r[
            "state_id"
        ]:
            {
                "state_id":
                    r[
                        "state_id"
                    ],

                "coord":
                    np.asarray(
                        (
                            float(
                                r[
                                    "phi"
                                ]
                            ),
                            float(
                                r[
                                    "psi"
                                ]
                            ),
                            float(
                                r[
                                    "omega"
                                ]
                            ),
                        ),
                        dtype=float,
                    ),
            }
        for r in candidate_records
    }


    selected = []


    for rank in range(
        1,
        batch_size+1,
    ):

        ids = sorted(
            remaining
        )


        coords = np.asarray(
            [
                remaining[
                    sid
                ][
                    "coord"
                ]
                for sid in ids
            ],
            dtype=float,
        )


        extra = (
            np.asarray(
                [
                    x[
                        "coord"
                    ]
                    for x in selected
                ],
                dtype=float,
            )
            if selected
            else None
        )


        scores = acquisition_scores(
            Xtrain,
            coords,
            sigma_f,
            ell,
            extra_condition_deg=extra,
        )


        current_coords = (
            np.vstack(
                (
                    Xtrain,
                    extra,
                )
            )
            if extra is not None
            else Xtrain
        )


        metrics = []


        for i,sid in enumerate(
            ids
        ):

            coord = coords[
                i
            ]


            min_dist = min(
                periodic_distance_deg(
                    coord,
                    ref,
                )
                for ref in current_coords
            )


            metrics.append({
                "state_id":
             
       sid,

                "coord":
                    coord,

                "score":
                    float(
                        scores[
                            "score"
                        ][
                            i
                        ]
                    ),

                "sd_R":
                    float(
                        scores[
                            "sd_R"
                        ][
                            i
                        ]
                    ),

                "sd_J":
                    float(
                        scores[
                            "sd_J"
                        ][
                            i
                        ]
                    ),

                "min_dist":
                    min_dist,
            })


        best_score = max(
            x[
                "score"
            ]
            for x in metrics
        )


        tied_score = [
            x
            for x in metrics
            if (
                best_score
                -
                x[
                    "score"
                ]
            )
            <=
            SCORE_TIE_TOL
        ]


        best_dist = max(
            x[
                "min_dist"
            ]
            for x in tied_score
        )


        tied_dist = [
            x
            for x in tied_score
            if (
                best_dist
                -
                x[
                    "min_dist"
                ]
            )
            <=
            DISTANCE_TIE_TOL_DEG
        ]


        chosen = min(
            tied_dist,
            key=lambda x:
                x[
                    "state_id"
                ],
        )


        selected.append({
            "selection_rank":
                rank,

            **chosen,
        })


        del remaining[
            chosen[
                "state_id"
            ]
        ]


    return selected
