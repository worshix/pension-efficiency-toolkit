"""Scale efficiency computation and RTS classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dea_core import DEAResult
from .utils import get_logger

logger = get_logger(__name__)

RTS_IRS = "IRS"  # Increasing Returns to Scale
RTS_DRS = "DRS"  # Decreasing Returns to Scale
RTS_CRS = "CRS"  # Constant Returns to Scale (efficient scale)


@dataclass
class ScaleResult:
    """Container for scale efficiency results."""

    fund_ids: list[str]
    scale_efficiency: np.ndarray      # SE = theta_ccr / theta_bcc
    rts_class: list[str]              # IRS, DRS, or CRS per DMU


def compute_scale_efficiency(
    ccr_result: DEAResult,
    bcc_result: DEAResult,
) -> ScaleResult:
    """Compute scale efficiency and classify returns-to-scale.

    Scale Efficiency = theta_CCR / theta_BCC.

    A DMU is:
    - CRS-efficient (SE ≈ 1) if both CCR and BCC are equal
    - IRS if the sum of BCC lambdas < 1 (for the optimal CCR solution)
    - DRS if the sum of BCC lambdas > 1

    Parameters
    ----------
    ccr_result:
        DEAResult from :func:`dea_ccr_input_oriented`.
    bcc_result:
        DEAResult from :func:`dea_bcc_input_oriented`.

    Returns
    -------
    ScaleResult
    """
    if list(ccr_result.fund_ids) != list(bcc_result.fund_ids):
        raise ValueError("CCR and BCC results must cover the same DMUs in the same order.")

    fund_ids = ccr_result.fund_ids
    theta_ccr = ccr_result.theta
    theta_bcc = bcc_result.theta

    # Avoid division by zero: if BCC score is 0 (degenerate), SE = 1
    se = np.where(theta_bcc > 1e-9, theta_ccr / theta_bcc, 1.0)
    se = np.clip(se, 0.0, 1.0)

    # RTS classification via sum of BCC lambdas for each DMU's optimal solution
    rts_class: list[str] = []
    for k in range(len(fund_ids)):
        lam_sum = bcc_result.lambdas[k].sum()
        if abs(lam_sum - 1.0) < 1e-3 and abs(se[k] - 1.0) < 1e-3:
            rts_class.append(RTS_CRS)
        elif lam_sum < 1.0 - 1e-3:
            rts_class.append(RTS_IRS)
        else:
            rts_class.append(RTS_DRS)

    logger.info(
        "Scale efficiency: mean=%.4f | IRS=%d DRS=%d CRS=%d",
        se.mean(),
        rts_class.count(RTS_IRS),
        rts_class.count(RTS_DRS),
        rts_class.count(RTS_CRS),
    )

    return ScaleResult(fund_ids=fund_ids, scale_efficiency=se, rts_class=rts_class)


def scale_to_dataframe(result: ScaleResult) -> pd.DataFrame:
    """Convert ScaleResult to a DataFrame for export."""
    return pd.DataFrame(
        {
            "fund_id": result.fund_ids,
            "scale_efficiency": result.scale_efficiency,
            "rts_classification": result.rts_class,
        }
    )
