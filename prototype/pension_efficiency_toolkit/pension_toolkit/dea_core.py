"""DEA core: CCR (CRS) and BCC (VRS) input-oriented models using PuLP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pulp

from .utils import get_logger

logger = get_logger(__name__)

pulp.LpSolverDefault.msg = 0  # suppress solver output


@dataclass
class DEAResult:
    """Container for DEA results (one row per DMU)."""

    fund_ids: list[str]
    theta: np.ndarray          # efficiency scores, shape (n,)
    lambdas: np.ndarray        # lambda weights, shape (n, n)
    slacks_in: np.ndarray      # input slacks, shape (n, m)
    slacks_out: np.ndarray     # output slacks, shape (n, s)
    model: str = "CCR"         # "CCR" or "BCC"
    peer_ids: list[list[str]] = field(default_factory=list)


def _solve_dmu(
    k: int,
    X: np.ndarray,
    Y: np.ndarray,
    vrs: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the input-oriented DEA LP for DMU k using two-phase approach.

    Phase I  — minimise theta (radial efficiency score, bounded to [0,1]).
    Phase II — maximise sum of slacks with theta fixed at Phase-I optimum.

    Returns
    -------
    (theta, lambdas, slack_in, slack_out)
    """
    n, m = X.shape
    _, s = Y.shape

    # ---- Phase I: minimise theta ----------------------------------------
    p1 = pulp.LpProblem(f"DEA_P1_DMU_{k}", pulp.LpMinimize)
    theta = pulp.LpVariable("theta", lowBound=0, upBound=1)
    lam = [pulp.LpVariable(f"lam_{j}", lowBound=0) for j in range(n)]

    p1 += theta  # minimise radial efficiency

    # Inequality form avoids bilinear issues; slacks handled in Phase II
    for i in range(m):
        p1 += pulp.lpSum(lam[j] * X[j, i] for j in range(n)) <= theta * X[k, i]

    for r in range(s):
        p1 += pulp.lpSum(lam[j] * Y[j, r] for j in range(n)) >= Y[k, r]

    if vrs:
        p1 += pulp.lpSum(lam) == 1

    status1 = p1.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status1] != "Optimal":
        logger.warning("DMU %d Phase-I: solver status %s", k, pulp.LpStatus[status1])
        return 1.0, np.zeros(n), np.zeros(m), np.zeros(s)

    theta_star = float(pulp.value(theta))

    # ---- Phase II: maximise slacks with theta fixed ----------------------
    p2 = pulp.LpProblem(f"DEA_P2_DMU_{k}", pulp.LpMaximize)
    lam2 = [pulp.LpVariable(f"lam2_{j}", lowBound=0) for j in range(n)]
    si = [pulp.LpVariable(f"si_{i}", lowBound=0) for i in range(m)]
    sr = [pulp.LpVariable(f"sr_{r}", lowBound=0) for r in range(s)]

    p2 += pulp.lpSum(si) + pulp.lpSum(sr)

    for i in range(m):
        p2 += (
            pulp.lpSum(lam2[j] * X[j, i] for j in range(n)) + si[i]
            == theta_star * X[k, i]
        )

    for r in range(s):
        p2 += (
            pulp.lpSum(lam2[j] * Y[j, r] for j in range(n)) - sr[r]
            == Y[k, r]
        )

    if vrs:
        p2 += pulp.lpSum(lam2) == 1

    status2 = p2.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status2] != "Optimal":
        logger.warning("DMU %d Phase-II: solver status %s", k, pulp.LpStatus[status2])
        lam_vals = np.zeros(n)
        si_vals = np.zeros(m)
        sr_vals = np.zeros(s)
    else:
        lam_vals = np.array([pulp.value(lv) for lv in lam2])
        si_vals = np.array([pulp.value(v) for v in si])
        sr_vals = np.array([pulp.value(v) for v in sr])

    lam_vals = np.clip(lam_vals, 0, None)
    si_vals = np.clip(si_vals, 0, None)
    sr_vals = np.clip(sr_vals, 0, None)

    return theta_star, lam_vals, si_vals, sr_vals


def _run_dea(
    X_in: np.ndarray,
    Y_out: np.ndarray,
    fund_ids: list[str],
    vrs: bool,
    model_name: str,
) -> DEAResult:
    """Internal DEA runner used by both CCR and BCC wrappers."""
    n = X_in.shape[0]
    if n < 2:
        raise ValueError("DEA requires at least 2 DMUs.")

    thetas, lambdas_list, si_list, sr_list = [], [], [], []

    for k in range(n):
        theta_k, lam_k, si_k, sr_k = _solve_dmu(k, X_in, Y_out, vrs=vrs)
        thetas.append(theta_k)
        lambdas_list.append(lam_k)
        si_list.append(si_k)
        sr_list.append(sr_k)

    theta_arr = np.array(thetas)
    lambda_arr = np.vstack(lambdas_list)
    slack_in_arr = np.vstack(si_list)
    slack_out_arr = np.vstack(sr_list)

    # Identify peer sets (lambda > threshold)
    PEER_THRESHOLD = 1e-4
    peer_ids: list[list[str]] = []
    for k in range(n):
        peers = [fund_ids[j] for j in range(n) if lambda_arr[k, j] > PEER_THRESHOLD]
        peer_ids.append(peers)

    logger.info("%s DEA complete. Mean efficiency: %.4f", model_name, theta_arr.mean())
    return DEAResult(
        fund_ids=fund_ids,
        theta=theta_arr,
        lambdas=lambda_arr,
        slacks_in=slack_in_arr,
        slacks_out=slack_out_arr,
        model=model_name,
        peer_ids=peer_ids,
    )


def dea_ccr_input_oriented(
    X_in: np.ndarray,
    Y_out: np.ndarray,
    fund_ids: list[str] | None = None,
) -> DEAResult:
    """Input-oriented CCR (CRS) DEA.

    Parameters
    ----------
    X_in:
        Input matrix, shape (n, m).
    Y_out:
        Output matrix, shape (n, s).
    fund_ids:
        Optional list of DMU identifiers.

    Returns
    -------
    DEAResult with model="CCR".
    """
    n = X_in.shape[0]
    if fund_ids is None:
        fund_ids = [f"DMU{i}" for i in range(n)]
    return _run_dea(X_in, Y_out, fund_ids, vrs=False, model_name="CCR")


def dea_bcc_input_oriented(
    X_in: np.ndarray,
    Y_out: np.ndarray,
    fund_ids: list[str] | None = None,
) -> DEAResult:
    """Input-oriented BCC (VRS) DEA.

    Parameters
    ----------
    X_in:
        Input matrix, shape (n, m).
    Y_out:
        Output matrix, shape (n, s).
    fund_ids:
        Optional list of DMU identifiers.

    Returns
    -------
    DEAResult with model="BCC".
    """
    n = X_in.shape[0]
    if fund_ids is None:
        fund_ids = [f"DMU{i}" for i in range(n)]
    return _run_dea(X_in, Y_out, fund_ids, vrs=True, model_name="BCC")
