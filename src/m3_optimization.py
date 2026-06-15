"""Module M3: deterministic optimization models.

This module contains three major model types from the assignment:

1. A simple four-variable LP for digital investment categories.
2. A region-by-item LP for spatial budget allocation.
3. A binary project-selection MIP solved by enumeration for transparency.

The code uses scipy.optimize.linprog for LPs and full enumeration for the small
binary project model.  Enumeration is acceptable here because there are only 15
projects, so 2^15 = 32768 combinations can be checked instantly on a laptop.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import BETA_REGION_ITEM, INITIAL_DIGITAL_INDEX, INVESTMENT_ITEMS, REGION_NAMES
from .data_loader import load_projects


@dataclass
class LPResult:
    """Readable wrapper around a scipy LP result."""

    status: str
    success: bool
    objective: float | None
    variables: pd.DataFrame | None
    message: str


def solve_simple_budget(total_budget: float = 100.0, min_human: float = 20.0) -> Dict[str, object]:
    """Solve the four-variable LP for digital investment categories."""
    c = [-0.85, -1.20, -0.95, -1.35]
    a_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65],
    ]
    b_ub = [total_budget, -25, -15, -min_human, -10, 0]
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    if not res.success:
        return {"success": False, "message": res.message}
    return {
        "success": True,
        "x": res.x,
        "objective": -float(res.fun),
        "shadow_budget": -float(res.ineqlin.marginals[0]),
        "labels": ["Hạ tầng số", "AI & dữ liệu", "Nhân lực số", "R&D"],
    }


def budget_sensitivity(budgets: Iterable[float]) -> pd.DataFrame:
    """Solve the simple LP for a list of budget levels."""
    rows = []
    for budget in budgets:
        result = solve_simple_budget(total_budget=float(budget))
        if result["success"]:
            x = result["x"]
            rows.append(
                {
                    "budget": budget,
                    "I": x[0],
                    "AI": x[1],
                    "H": x[2],
                    "R&D": x[3],
                    "objective": result["objective"],
                    "shadow_budget": result["shadow_budget"],
                }
            )
    return pd.DataFrame(rows)


def build_region_lp_matrices(lam: float = 0.70, fair: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[float | None, float | None]]]:
    """Construct c, A_ub, b_ub, and bounds for the regional allocation LP."""
    nvar = 25
    c = np.r_[-BETA_REGION_ITEM.flatten(), 0.0]
    a_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    row = np.zeros(nvar)
    row[:24] = 1.0
    a_ub.append(row)
    b_ub.append(50000.0)
    for region in range(6):
        row = np.zeros(nvar)
        row[region * 4 : (region + 1) * 4] = -1.0
        a_ub.append(row)
        b_ub.append(-5000.0)
        row = np.zeros(nvar)
        row[region * 4 : (region + 1) * 4] = 1.0
        a_ub.append(row)
        b_ub.append(12000.0)
    row = np.zeros(nvar)
    row[3:24:4] = -1.0
    a_ub.append(row)
    b_ub.append(-12000.0)
    if fair:
        gamma = 0.002
        for region in range(6):
            row = np.zeros(nvar)
            row[region * 4 + 1] = gamma
            row[-1] = -1.0
            a_ub.append(row)
            b_ub.append(-INITIAL_DIGITAL_INDEX[region])
        for region in range(6):
            row = np.zeros(nvar)
            row[region * 4 + 1] = -gamma
            row[-1] = lam
            a_ub.append(row)
            b_ub.append(INITIAL_DIGITAL_INDEX[region])
    bounds = [(0, None)] * 24 + [(0, None)]
    return c, np.array(a_ub), np.array(b_ub), bounds


def solve_region_budget(lam: float = 0.70, fair: bool = True) -> LPResult:
    """Solve the regional budget allocation LP."""
    c, a_ub, b_ub, bounds = build_region_lp_matrices(lam=lam, fair=fair)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return LPResult("infeasible_or_failed", False, None, None, res.message)
    matrix = res.x[:24].reshape(6, 4)
    table = pd.DataFrame(matrix, columns=INVESTMENT_ITEMS)
    table.insert(0, "region", REGION_NAMES)
    return LPResult("optimal", True, -float(res.fun), table, res.message)


def diagnose_region_fairness(lam: float = 0.70) -> pd.DataFrame:
    """Diagnose why the fairness constraint can be infeasible."""
    max_index = float(INITIAL_DIGITAL_INDEX.max())
    threshold = lam * max_index
    rows = []
    for region, initial in zip(REGION_NAMES, INITIAL_DIGITAL_INDEX):
        required_d = max(0.0, (threshold - initial) / 0.002)
        rows.append(
            {
                "region": region,
                "initial_digital_index": initial,
                "required_min_index": threshold,
                "required_D_investment": required_d,
                "above_region_cap_12000": required_d > 12000,
            }
        )
    return pd.DataFrame(rows)


def project_feasible(
    y: np.ndarray,
    projects: pd.DataFrame,
    budget: float = 80000.0,
    budget_year_1_2: float = 40000.0,
    require_p1_p2: bool = False,
) -> bool:
    """Check all MIP constraints for a binary project vector."""
    if projects["cost_billion_VND"].to_numpy(float) @ y > budget:
        return False
    if projects["cost_year_1_2"].to_numpy(float) @ y > budget_year_1_2:
        return False
    if require_p1_p2:
        if not (y[0] == 1 and y[1] == 1):
            return False
    else:
        if y[0] + y[1] > 1:
            return False
    if y[7] > y[11]:
        return False
    if y[12] > y[11]:
        return False
    if y[3] + y[4] < 1:
        return False
    if y[13] < 1:
        return False
    if y.sum() < 7 or y.sum() > 11:
        return False
    return True


def project_success_probabilities(projects: pd.DataFrame) -> np.ndarray:
    """Assign simple completion probabilities by project field."""
    values = []
    for field in projects["field"]:
        if field == "Hạ tầng":
            values.append(0.85)
        elif field == "Chính phủ số":
            values.append(0.75)
        elif field in ["AI", "Bán dẫn"]:
            values.append(0.65)
        else:
            values.append(0.80)
    return np.array(values, dtype=float)


def solve_project_selection(
    budget: float = 80000.0,
    budget_year_1_2: float = 40000.0,
    require_p1_p2: bool = False,
    expected_value: bool = False,
) -> Dict[str, object]:
    """Solve the 15-project binary selection problem by full enumeration."""
    projects = load_projects()
    benefits = projects["benefit_NPV_billion_VND"].to_numpy(float)
    if expected_value:
        benefits = benefits * project_success_probabilities(projects)
    best_value = -np.inf
    best_y = None
    feasible_count = 0
    for bits in product([0, 1], repeat=len(projects)):
        y = np.array(bits, dtype=int)
        if project_feasible(y, projects, budget, budget_year_1_2, require_p1_p2):
            feasible_count += 1
            value = benefits @ y
            if value > best_value:
                best_value = float(value)
                best_y = y.copy()
    if best_y is None:
        return {"success": False, "message": "No feasible project set"}
    selected = projects.loc[best_y == 1].copy()
    return {
        "success": True,
        "objective": best_value,
        "selected": selected,
        "binary_vector": best_y,
        "total_cost": float(projects["cost_billion_VND"].to_numpy(float) @ best_y),
        "total_cost_year_1_2": float(projects["cost_year_1_2"].to_numpy(float) @ best_y),
        "feasible_count": feasible_count,
    }


def project_scenario_table() -> pd.DataFrame:
    """Return project-selection results for all report scenarios."""
    scenarios = [
        ("Cơ sở 80.000 tỷ", 80000, 40000, False, False),
        ("Nới 100.000 tỷ", 100000, 50000, False, False),
        ("Bắt buộc P1+P2", 80000, 40000, True, False),
        ("Lợi ích kỳ vọng rủi ro", 80000, 40000, False, True),
    ]
    rows = []
    for label, budget, budget12, require, expected in scenarios:
        result = solve_project_selection(budget, budget12, require, expected)
        if result["success"]:
            rows.append(
                {
                    "scenario": label,
                    "objective": result["objective"],
                    "total_cost": result["total_cost"],
                    "cost_year_1_2": result["total_cost_year_1_2"],
                    "n_projects": len(result["selected"]),
                    "selected_ids": ", ".join(result["selected"]["project_id"].tolist()),
                }
            )
        else:
            rows.append({"scenario": label, "objective": None, "selected_ids": result["message"]})
    return pd.DataFrame(rows)


def run_m3() -> Dict[str, object]:
    """Run all deterministic optimization models."""
    return {
        "simple_lp": solve_simple_budget(),
        "budget_sensitivity": budget_sensitivity([100, 120, 140]),
        "region_fair_original": solve_region_budget(lam=0.70, fair=True),
        "region_fair_relaxed": solve_region_budget(lam=0.68, fair=True),
        "region_no_fairness": solve_region_budget(fair=False),
        "fairness_diagnostics": diagnose_region_fairness(0.70),
        "project_scenarios": project_scenario_table(),
    }
