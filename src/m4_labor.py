"""Module M4: labor-market simulation under AI and automation.

The final-exam problem asks students to translate AI adoption into job creation,
job upgrading, displacement, and retraining capacity.  This module keeps the
model linear so that it can be solved with scipy.optimize.linprog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog


LABOR_SECTORS = [
    "Nông-Lâm-Thủy sản",
    "CN chế biến chế tạo",
    "Xây dựng",
    "Bán buôn-bán lẻ",
    "Tài chính-Ngân hàng",
    "Logistics-Vận tải",
    "CNTT-Truyền thông",
    "Giáo dục-Đào tạo",
]

LABOR_MILLION = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22], dtype=float) / 100.0
A1_NEW_AI = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1_UPGRADE_H = np.array([45, 28, 35, 32, 22, 30, 20, 55], dtype=float)
C1_DISPLACE = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1_RETRAIN = np.array([50, 32, 42, 38, 26, 36, 24, 62], dtype=float)


@dataclass
class LaborModelResult:
    """Container for labor optimization results."""

    success: bool
    objective: float | None
    allocation: pd.DataFrame
    message: str


def job_components(x_ai: np.ndarray, x_h: np.ndarray) -> pd.DataFrame:
    """Compute NewJob, UpgradeJob, DisplacedJob, Capacity, and NetJob."""
    new_jobs = A1_NEW_AI * x_ai
    upgraded = B1_UPGRADE_H * x_h
    displaced = C1_DISPLACE * RISK * x_ai
    retrain = D1_RETRAIN * x_h
    net = new_jobs + upgraded - displaced
    return pd.DataFrame(
        {
            "sector": LABOR_SECTORS,
            "x_AI": x_ai,
            "x_H": x_h,
            "NewJob": new_jobs,
            "UpgradeJob": upgraded,
            "DisplacedJob": displaced,
            "RetrainingCapacity": retrain,
            "NetJob": net,
        }
    )


def build_labor_lp(total_budget: float = 30000.0, sector_cap: float | None = 9000.0, max_displacement_share: float | None = None):
    """Build matrices for the labor LP."""
    n = len(LABOR_SECTORS)
    net_ai = A1_NEW_AI - C1_DISPLACE * RISK
    c = -np.r_[net_ai, B1_UPGRADE_H]
    a_ub = []
    b_ub = []
    row = np.ones(2 * n)
    a_ub.append(row)
    b_ub.append(total_budget)
    for i in range(n):
        row = np.zeros(2 * n)
        row[i] = -net_ai[i]
        row[n + i] = -B1_UPGRADE_H[i]
        a_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(2 * n)
        row[i] = C1_DISPLACE[i] * RISK[i]
        row[n + i] = -D1_RETRAIN[i]
        a_ub.append(row)
        b_ub.append(0.0)
        if sector_cap is not None:
            row = np.zeros(2 * n)
            row[i] = 1.0
            row[n + i] = 1.0
            a_ub.append(row)
            b_ub.append(sector_cap)
        if max_displacement_share is not None:
            row = np.zeros(2 * n)
            row[i] = C1_DISPLACE[i] * RISK[i]
            a_ub.append(row)
            b_ub.append(max_displacement_share * LABOR_MILLION[i] * 1_000_000.0)
    return c, np.array(a_ub), np.array(b_ub)


def solve_labor_budget(total_budget: float = 30000.0, sector_cap: float | None = 9000.0, max_displacement_share: float | None = None) -> LaborModelResult:
    """Solve the AI/H labor allocation LP."""
    c, a_ub, b_ub = build_labor_lp(total_budget, sector_cap, max_displacement_share)
    n = len(LABOR_SECTORS)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * (2 * n), method="highs")
    if not res.success:
        empty = pd.DataFrame(columns=["sector", "x_AI", "x_H", "NetJob"])
        return LaborModelResult(False, None, empty, res.message)
    x_ai = res.x[:n]
    x_h = res.x[n:]
    allocation = job_components(x_ai, x_h)
    return LaborModelResult(True, -float(res.fun), allocation, res.message)


def retraining_threshold_for_sector(sector_index: int, x_ai: float) -> float:
    """Minimum H investment needed so retraining capacity covers displacement."""
    displaced = C1_DISPLACE[sector_index] * RISK[sector_index] * x_ai
    return float(displaced / D1_RETRAIN[sector_index])


def netjob_threshold_for_sector(sector_index: int, x_ai: float) -> float:
    """Minimum H investment needed so NetJob is non-negative."""
    net_ai = (A1_NEW_AI[sector_index] - C1_DISPLACE[sector_index] * RISK[sector_index]) * x_ai
    if net_ai >= 0:
        return 0.0
    return float(-net_ai / B1_UPGRADE_H[sector_index])


def vulnerable_labor_flow_table() -> pd.DataFrame:
    """Return a simple table for sectors with high vulnerable-labor exposure."""
    vulnerable_indices = [0, 2, 3]
    rows = []
    for i in vulnerable_indices:
        rows.append(
            {
                "sector": LABOR_SECTORS[i],
                "labor_million": LABOR_MILLION[i],
                "automation_risk_pct": RISK[i] * 100,
                "policy_note": "Ưu tiên đào tạo lại và hỗ trợ chuyển dịch việc làm",
            }
        )
    return pd.DataFrame(rows)


def solve_labor_scenarios() -> pd.DataFrame:
    """Compare labor model results under several policy caps."""
    rows = []
    for cap in [None, 12000, 9000, 6000]:
        result = solve_labor_budget(sector_cap=cap)
        rows.append(
            {
                "sector_cap": "No cap" if cap is None else cap,
                "success": result.success,
                "objective_NetJob": result.objective,
                "active_sectors": int((result.allocation.get("x_AI", pd.Series(dtype=float)) + result.allocation.get("x_H", pd.Series(dtype=float)) > 1e-6).sum()) if result.success else 0,
            }
        )
    return pd.DataFrame(rows)


def run_m4() -> Dict[str, object]:
    """Run the labor module for the report."""
    base = solve_labor_budget(sector_cap=9000)
    return {
        "base": base,
        "scenarios": solve_labor_scenarios(),
        "vulnerable": vulnerable_labor_flow_table(),
        "manufacturing_retraining_at_9000_ai": retraining_threshold_for_sector(1, 9000),
        "manufacturing_netjob_threshold_at_9000_ai": netjob_threshold_for_sector(1, 9000),
    }
