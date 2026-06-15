"""Module M1: Cobb-Douglas production, TFP, and macro forecasting.

The module implements the first group of exercises in the final-exam problem.
It is intentionally verbose, because the source code itself is part of the
submission.  Each function is small enough to be tested, explained in the
report, and reused in the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .config import COBB_DOUGLAS_COEFFICIENTS
from .data_loader import load_macro


@dataclass
class CobbDouglasResult:
    """Container for a fitted Cobb-Douglas teaching model."""

    data: pd.DataFrame
    average_tfp: float
    mape: float
    coefficients: Tuple[float, float, float, float, float]


def _extract_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract model arrays in the order Y, K, L, D, AI, H."""
    y = df["GDP_trillion_VND"].to_numpy(float)
    k = df["K_accum_trillion_VND"].to_numpy(float)
    l = df["L_million"].to_numpy(float)
    d = df["digital_GDP_pct"].to_numpy(float)
    ai = df["AI_thousand_digital_firms"].to_numpy(float)
    h = df["trained_labor_pct"].to_numpy(float)
    return y, k, l, d, ai, h


def production_function(
    tfp: float | np.ndarray,
    capital: float | np.ndarray,
    labor: float | np.ndarray,
    digital: float | np.ndarray,
    ai_capacity: float | np.ndarray,
    human_capital: float | np.ndarray,
    coefficients: Tuple[float, float, float, float, float] = COBB_DOUGLAS_COEFFICIENTS,
) -> float | np.ndarray:
    """Evaluate the extended Cobb-Douglas production function."""
    a, b, g, d, t = coefficients
    return tfp * capital**a * labor**b * digital**g * ai_capacity**d * human_capital**t


def estimate_tfp(
    df: pd.DataFrame | None = None,
    coefficients: Tuple[float, float, float, float, float] = COBB_DOUGLAS_COEFFICIENTS,
) -> pd.DataFrame:
    """Solve TFP A_t from observed GDP and inputs."""
    df = load_macro() if df is None else df.copy()
    y, k, l, digital, ai, human = _extract_arrays(df)
    a, b, g, d, t = coefficients
    denominator = k**a * l**b * digital**g * ai**d * human**t
    df["TFP_A"] = y / denominator
    return df


def fit_average_tfp_model(
    df: pd.DataFrame | None = None,
    coefficients: Tuple[float, float, float, float, float] = COBB_DOUGLAS_COEFFICIENTS,
) -> CobbDouglasResult:
    """Fit a teaching model using the average TFP over 2020-2025."""
    out = estimate_tfp(df, coefficients)
    y, k, l, digital, ai, human = _extract_arrays(out)
    average_tfp = float(out["TFP_A"].mean())
    y_hat = production_function(average_tfp, k, l, digital, ai, human, coefficients)
    out["GDP_hat_Abar"] = y_hat
    out["APE_pct"] = np.abs(y_hat / y - 1.0) * 100.0
    mape = float(out["APE_pct"].mean())
    return CobbDouglasResult(out, average_tfp, mape, coefficients)


def growth_accounting(
    df: pd.DataFrame | None = None,
    coefficients: Tuple[float, float, float, float, float] = COBB_DOUGLAS_COEFFICIENTS,
) -> pd.DataFrame:
    """Decompose average log growth into input and TFP contributions."""
    fitted = estimate_tfp(df, coefficients)
    a, b, g, d, t = coefficients
    years = int(fitted["year"].iloc[-1] - fitted["year"].iloc[0])
    total_log_growth = np.log(fitted["GDP_trillion_VND"].iloc[-1] / fitted["GDP_trillion_VND"].iloc[0]) / years
    items = [
        ("Vốn vật chất K", a, "K_accum_trillion_VND"),
        ("Lao động L", b, "L_million"),
        ("Số hóa D", g, "digital_GDP_pct"),
        ("Năng lực AI", d, "AI_thousand_digital_firms"),
        ("Nhân lực số H", t, "trained_labor_pct"),
        ("TFP", 1.0, "TFP_A"),
    ]
    rows = []
    for name, coef, col in items:
        contribution = coef * np.log(fitted[col].iloc[-1] / fitted[col].iloc[0]) / years
        rows.append(
            {
                "factor": name,
                "log_point_per_year_pct": contribution * 100,
                "share_pct": contribution / total_log_growth * 100,
            }
        )
    return pd.DataFrame(rows)


def forecast_2030_policy_target(
    df: pd.DataFrame | None = None,
    digital_target: float = 30.0,
    ai_target: float = 100.0,
    human_target: float = 35.0,
    capital_growth: float = 0.06,
    labor_growth: float = 0.06,
    tfp_growth: float = 0.012,
) -> float:
    """Forecast GDP in 2030 under the policy target described in the problem."""
    fitted = estimate_tfp(df)
    last = fitted.iloc[-1]
    years = 5
    a, b, g, d, t = COBB_DOUGLAS_COEFFICIENTS
    tfp = last["TFP_A"] * (1 + tfp_growth) ** years
    capital = last["K_accum_trillion_VND"] * (1 + capital_growth) ** years
    labor = last["L_million"] * (1 + labor_growth) ** years
    return float(production_function(tfp, capital, labor, digital_target, ai_target, human_target, COBB_DOUGLAS_COEFFICIENTS))


def scenario_forecast(
    df: pd.DataFrame | None,
    allocation: np.ndarray,
    years: Iterable[int] = range(2026, 2031),
    annual_budget: float = 1000.0,
    private_capital_growth: float = 0.04,
    labor_growth: float = 0.006,
) -> Dict[str, float]:
    """Simulate a simple 2030 forecast for one allocation scenario."""
    fitted = estimate_tfp(df)
    last = fitted.iloc[-1]
    capital = float(last["K_accum_trillion_VND"])
    labor = float(last["L_million"])
    digital = float(last["digital_GDP_pct"])
    ai = float(last["AI_thousand_digital_firms"])
    human = float(last["trained_labor_pct"])
    tfp = float(last["TFP_A"])
    for _year in years:
        capital = capital * (1 + private_capital_growth) + allocation[0] * annual_budget
        labor = labor * (1 + labor_growth)
        digital = digital + allocation[1] * annual_budget / 100.0
        ai = ai + allocation[2] * annual_budget / 20.0
        human = human + allocation[3] * annual_budget / 200.0
        tfp = tfp * (1 + 0.006 + 0.00025 * digital + 0.00015 * ai / 100.0 + 0.00035 * human / 100.0)
    gdp = production_function(tfp, capital, labor, digital, ai, human)
    return {
        "GDP_2030": float(gdp),
        "K_2030": float(capital),
        "L_2030": float(labor),
        "D_2030": float(digital),
        "AI_2030": float(ai),
        "H_2030": float(human),
        "TFP_2030": float(tfp),
    }


def elasticity_summary() -> pd.DataFrame:
    """Return the elasticities used in the teaching production function."""
    names = ["K", "L", "D", "AI", "H"]
    rows = []
    for name, value in zip(names, COBB_DOUGLAS_COEFFICIENTS):
        rows.append({"input": name, "elasticity": value, "interpretation": f"1% tăng {name} làm GDP tăng khoảng {value:.2f}%"})
    rows.append({"input": "Total", "elasticity": sum(COBB_DOUGLAS_COEFFICIENTS), "interpretation": "Lợi suất không đổi theo quy mô"})
    return pd.DataFrame(rows)


def run_m1() -> Dict[str, object]:
    """Run the complete M1 workflow."""
    fit = fit_average_tfp_model()
    accounting = growth_accounting(fit.data)
    forecast = forecast_2030_policy_target(fit.data)
    return {
        "fit": fit,
        "accounting": accounting,
        "forecast_2030": forecast,
        "elasticities": elasticity_summary(),
    }
