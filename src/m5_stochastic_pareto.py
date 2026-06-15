"""Module M5: risk, stochastic programming, and approximate Pareto search.

The exam includes stochastic programming and Pareto/NSGA-II.  To keep this
repository runnable on a standard laptop without commercial solvers, the code
uses scipy.optimize.linprog for the two-stage stochastic LP and a transparent
random-search approximation for the multi-objective Pareto frontier.  The
random-search version does not replace NSGA-II in research, but it provides a
local dashboard-friendly result that can be inspected and extended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import BETA_REGION_ITEM, INVESTMENT_ITEMS, REGION_NAMES


SCENARIO_NAMES = ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"]
SCENARIO_PROBABILITIES = np.array([0.30, 0.45, 0.20, 0.05])
BETA_FIRST_STAGE = np.array([1.00, 1.10, 1.25, 0.95])
BETA_SECOND_STAGE = np.array(
    [
        [1.25, 1.35, 1.55, 1.05],
        [1.00, 1.10, 1.25, 0.95],
        [0.75, 0.85, 0.90, 1.00],
        [0.40, 0.50, 0.55, 1.10],
    ],
    dtype=float,
)
EMISSION_INTENSITY = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
DATA_RISK_AI = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
DATA_RISK_REDUCTION_H = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])


@dataclass
class StochasticResult:
    """Container for the two-stage stochastic programming solution."""

    success: bool
    objective: float | None
    first_stage: pd.DataFrame
    second_stage: pd.DataFrame
    message: str


def solve_two_stage_sp() -> StochasticResult:
    """Solve the simplified two-stage stochastic programming model."""
    c = -np.r_[BETA_FIRST_STAGE, (SCENARIO_PROBABILITIES[:, None] * BETA_SECOND_STAGE).flatten()]
    a_ub = []
    b_ub = []
    row = np.zeros(20)
    row[:4] = 1.0
    a_ub.append(row)
    b_ub.append(65000.0)
    for s in range(4):
        row = np.zeros(20)
        row[4 + s * 4 : 4 + (s + 1) * 4] = 1.0
        a_ub.append(row)
        b_ub.append(15000.0)
        row = np.zeros(20)
        row[4 + s * 4 + 2] = 1.0
        row[3] = -0.5
        a_ub.append(row)
        b_ub.append(0.0)
    res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub), bounds=[(0, None)] * 20, method="highs")
    if not res.success:
        return StochasticResult(False, None, pd.DataFrame(), pd.DataFrame(), res.message)
    first = pd.DataFrame({"item": INVESTMENT_ITEMS, "x_first_stage": res.x[:4]})
    second = pd.DataFrame(res.x[4:].reshape(4, 4), columns=INVESTMENT_ITEMS)
    second.insert(0, "scenario", SCENARIO_NAMES)
    return StochasticResult(True, -float(res.fun), first, second, res.message)


def evaluate_fixed_first_stage(x: np.ndarray) -> float:
    """Evaluate expected value of a fixed first-stage decision with optimal recourse."""
    x = np.asarray(x, dtype=float)
    if x.sum() > 65000 + 1e-6:
        raise ValueError("First-stage budget exceeded")
    value = float(BETA_FIRST_STAGE @ x)
    for s in range(4):
        best = max(BETA_SECOND_STAGE[s, 0], BETA_SECOND_STAGE[s, 1], BETA_SECOND_STAGE[s, 3])
        ai_capacity = 0.5 * x[3]
        ai_budget = min(15000.0, ai_capacity) if BETA_SECOND_STAGE[s, 2] > best else 0.0
        non_ai_budget = 15000.0 - ai_budget
        recourse = ai_budget * BETA_SECOND_STAGE[s, 2] + non_ai_budget * best
        value += SCENARIO_PROBABILITIES[s] * recourse
    return value


def deterministic_equivalent_table() -> pd.DataFrame:
    """Construct a compact comparison of deterministic and stochastic strategies."""
    sp = solve_two_stage_sp()
    strategies = {
        "All AI": np.array([0, 0, 65000, 0], dtype=float),
        "All D": np.array([0, 65000, 0, 0], dtype=float),
        "All H": np.array([0, 0, 0, 65000], dtype=float),
        "Balanced": np.array([16250, 16250, 16250, 16250], dtype=float),
    }
    if sp.success:
        strategies["SP optimum"] = sp.first_stage["x_first_stage"].to_numpy(float)
    rows = []
    for name, x in strategies.items():
        rows.append({"strategy": name, "expected_value": evaluate_fixed_first_stage(x), "x_I": x[0], "x_D": x[1], "x_AI": x[2], "x_H": x[3]})
    return pd.DataFrame(rows)


def allocation_objectives(x: np.ndarray) -> Dict[str, float]:
    """Evaluate growth, inequality, emission, and data-risk objectives."""
    matrix = np.asarray(x, dtype=float).reshape(6, 4)
    growth = float((BETA_REGION_ITEM * matrix).sum())
    regional_sum = matrix.sum(axis=1)
    inequality = float(np.abs(regional_sum - regional_sum.mean()).mean())
    emission = float((EMISSION_INTENSITY * (matrix[:, 0] + matrix[:, 2])).sum())
    data_risk = float((DATA_RISK_AI * matrix[:, 2]).sum() - (DATA_RISK_REDUCTION_H * matrix[:, 3]).sum())
    return {
        "growth": growth,
        "inequality": inequality,
        "emission": emission,
        "data_risk": data_risk,
    }


def random_feasible_allocation(rng: np.random.Generator, total_budget: float = 50000.0) -> np.ndarray:
    """Generate a simple random allocation satisfying rough region floors."""
    region_budgets = rng.dirichlet(np.ones(6)) * total_budget
    region_budgets = np.maximum(region_budgets, 5000.0)
    region_budgets = region_budgets / region_budgets.sum() * total_budget
    region_budgets = np.minimum(region_budgets, 12000.0)
    region_budgets = region_budgets / region_budgets.sum() * total_budget
    matrix = np.zeros((6, 4))
    for r in range(6):
        shares = rng.dirichlet(np.ones(4))
        matrix[r] = region_budgets[r] * shares
    return matrix.flatten()


def dominates(a: pd.Series, b: pd.Series) -> bool:
    """Return True if row a Pareto-dominates row b.

    Growth is maximized; inequality, emission and data risk are minimized.
    """
    better_or_equal = (
        a["growth"] >= b["growth"]
        and a["inequality"] <= b["inequality"]
        and a["emission"] <= b["emission"]
        and a["data_risk"] <= b["data_risk"]
    )
    strictly_better = (
        a["growth"] > b["growth"]
        or a["inequality"] < b["inequality"]
        or a["emission"] < b["emission"]
        or a["data_risk"] < b["data_risk"]
    )
    return bool(better_or_equal and strictly_better)


def approximate_pareto(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Approximate a Pareto set through random feasible allocations."""
    rng = np.random.default_rng(seed)
    rows = []
    allocations = []
    for i in range(n_samples):
        x = random_feasible_allocation(rng)
        obj = allocation_objectives(x)
        obj["sample_id"] = i
        rows.append(obj)
        allocations.append(x)
    df = pd.DataFrame(rows)
    pareto = []
    for i, row in df.iterrows():
        is_dominated = False
        for j, other in df.iterrows():
            if i != j and dominates(other, row):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(i)
    out = df.loc[pareto].copy().reset_index(drop=True)
    return out


def compromise_solution(pareto: pd.DataFrame, weights: Tuple[float, float, float, float] = (0.40, 0.25, 0.20, 0.15)) -> pd.DataFrame:
    """Select one compromise solution from a Pareto table using normalized TOPSIS-like score."""
    if pareto.empty:
        return pareto
    p = pareto.copy()
    benefit = (p["growth"] - p["growth"].min()) / (p["growth"].max() - p["growth"].min() + 1e-12)
    inequality = (p["inequality"].max() - p["inequality"]) / (p["inequality"].max() - p["inequality"].min() + 1e-12)
    emission = (p["emission"].max() - p["emission"]) / (p["emission"].max() - p["emission"].min() + 1e-12)
    risk = (p["data_risk"].max() - p["data_risk"]) / (p["data_risk"].max() - p["data_risk"].min() + 1e-12)
    p["compromise_score"] = weights[0] * benefit + weights[1] * inequality + weights[2] * emission + weights[3] * risk
    return p.sort_values("compromise_score", ascending=False).head(1)


def run_m5() -> Dict[str, object]:
    """Run stochastic and Pareto-risk modules."""
    sp = solve_two_stage_sp()
    pareto = approximate_pareto(n_samples=600, seed=2026)
    return {
        "sp": sp,
        "deterministic_comparison": deterministic_equivalent_table(),
        "pareto": pareto,
        "compromise": compromise_solution(pareto),
    }
