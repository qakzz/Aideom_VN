"""Module M2: sector priority index and TOPSIS regional ranking.

This module covers the multi-criteria decision-making part of the exam.  It
contains both a sector-level priority index and a region-level TOPSIS model for
AI investment readiness.  The code keeps the normalization steps explicit so
that students can discuss how value choices enter the ranking.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import SECTOR_PRIORITY_DEFAULT_WEIGHTS, TOPSIS_EXPERT_WEIGHTS
from .data_loader import load_regions, load_sectors


def minmax_good(series: pd.Series) -> pd.Series:
    """Normalize a benefit criterion to [0, 1]."""
    denominator = series.max() - series.min()
    if denominator == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / denominator


def minmax_cost(series: pd.Series) -> pd.Series:
    """Normalize a cost criterion to [0, 1] where higher means worse."""
    denominator = series.max() - series.min()
    if denominator == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / denominator


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize weights to sum to one and guard against invalid inputs."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return w / total


def sector_priority(
    df: pd.DataFrame | None = None,
    weights: Sequence[float] = SECTOR_PRIORITY_DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """Compute the sector priority index from growth, productivity, spillover and risk."""
    df = load_sectors() if df is None else df.copy()
    good_cols = [
        "growth_rate_2024_pct",
        "productivity_million_VND_per_worker",
        "spillover_coef_0_1",
        "export_billion_USD",
        "labor_million",
        "ai_readiness_0_100",
    ]
    normalized = pd.DataFrame({col: minmax_good(df[col]) for col in good_cols})
    normalized["automation_risk_norm"] = minmax_cost(df["automation_risk_pct"])
    w = normalize_weights(weights)
    score = normalized[good_cols].to_numpy() @ w[:6] - normalized["automation_risk_norm"].to_numpy() * w[6]
    out = df.copy()
    out["Priority"] = score
    out["rank"] = out["Priority"].rank(ascending=False, method="dense").astype(int)
    return out.sort_values("Priority", ascending=False).reset_index(drop=True)


def sector_priority_matrix(weights_list: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """Compare multiple policy-weight systems in one table."""
    base = load_sectors()[["sector_name_vi"]].copy()
    out = base.copy()
    for label, weights in weights_list.items():
        ranking = sector_priority(weights=weights)[["sector_name_vi", "Priority"]]
        out = out.merge(ranking.rename(columns={"Priority": label}), on="sector_name_vi", how="left")
    return out


def ai_weight_sensitivity(start: float = 0.05, stop: float = 0.40, step: float = 0.05) -> pd.DataFrame:
    """Vary the AI-readiness weight and record the sector scores."""
    base = np.array(SECTOR_PRIORITY_DEFAULT_WEIGHTS, dtype=float)
    rows = []
    values = np.arange(start, stop + 1e-9, step)
    for ai_weight in values:
        other = base.copy()
        other[5] = 0.0
        other = other / other.sum() * (1.0 - ai_weight)
        weights = other.copy()
        weights[5] = ai_weight
        result = sector_priority(weights=weights)
        for _, row in result.iterrows():
            rows.append(
                {
                    "AI_weight": round(float(ai_weight), 3),
                    "sector_name_vi": row["sector_name_vi"],
                    "Priority": row["Priority"],
                    "rank": row["rank"],
                }
            )
    return pd.DataFrame(rows)


def vector_normalize(matrix: np.ndarray) -> np.ndarray:
    """Vector-normalize a TOPSIS decision matrix."""
    denom = np.sqrt((matrix**2).sum(axis=0))
    denom[denom == 0] = 1.0
    return matrix / denom


def topsis_score(
    matrix: np.ndarray,
    weights: Sequence[float],
    is_benefit: Sequence[bool],
) -> np.ndarray:
    """Compute TOPSIS closeness coefficient for all alternatives."""
    weights = normalize_weights(weights)
    benefit_flags = np.asarray(is_benefit, dtype=bool)
    r = vector_normalize(matrix.astype(float))
    v = r * weights
    ideal = np.where(benefit_flags, v.max(axis=0), v.min(axis=0))
    anti = np.where(benefit_flags, v.min(axis=0), v.max(axis=0))
    s_pos = np.sqrt(((v - ideal) ** 2).sum(axis=1))
    s_neg = np.sqrt(((v - anti) ** 2).sum(axis=1))
    return s_neg / (s_pos + s_neg)


def topsis_regions(
    df: pd.DataFrame | None = None,
    weights: Sequence[float] = TOPSIS_EXPERT_WEIGHTS,
) -> pd.DataFrame:
    """Rank Vietnamese regions by AI investment readiness using TOPSIS."""
    df = load_regions() if df is None else df.copy()
    criteria = [
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ]
    is_benefit = [True, True, True, True, True, True, True, False]
    matrix = df[criteria].to_numpy(float)
    df["TOPSIS_score"] = topsis_score(matrix, weights, is_benefit)
    df["rank"] = df["TOPSIS_score"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """Compute objective weights using the entropy method."""
    x = matrix.astype(float)
    col_sums = x.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    p = x / col_sums
    k = 1.0 / np.log(len(x))
    entropy = -k * np.nansum(p * np.log(p + 1e-12), axis=0)
    diversity = 1 - entropy
    if diversity.sum() == 0:
        return np.ones(x.shape[1]) / x.shape[1]
    return diversity / diversity.sum()


def topsis_regions_entropy(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run TOPSIS with entropy-derived weights."""
    df = load_regions() if df is None else df.copy()
    criteria = [
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ]
    x = df[criteria].copy()
    x["gini_coef"] = x["gini_coef"].max() - x["gini_coef"] + x["gini_coef"].min()
    weights = entropy_weights(x.to_numpy(float))
    out = topsis_regions(df, weights=weights)
    out["weighting_method"] = "entropy"
    return out


def compare_topsis_methods() -> pd.DataFrame:
    """Combine expert TOPSIS and entropy TOPSIS in one table."""
    expert = topsis_regions()[["region_name_vi", "TOPSIS_score", "rank"]]
    entropy = topsis_regions_entropy()[["region_name_vi", "TOPSIS_score", "rank"]]
    return expert.merge(
        entropy,
        on="region_name_vi",
        suffixes=("_expert", "_entropy"),
        how="outer",
    )


def region_ai_weight_sensitivity(values: Iterable[float]) -> pd.DataFrame:
    """Track regional TOPSIS rankings as AI-readiness weight changes."""
    base = np.array(TOPSIS_EXPERT_WEIGHTS, dtype=float)
    rows = []
    for ai_weight in values:
        other = base.copy()
        other[3] = 0.0
        other = other / other.sum() * (1.0 - ai_weight)
        weights = other.copy()
        weights[3] = ai_weight
        result = topsis_regions(weights=weights)
        for _, row in result.iterrows():
            rows.append(
                {
                    "AI_weight": float(ai_weight),
                    "region_name_vi": row["region_name_vi"],
                    "TOPSIS_score": row["TOPSIS_score"],
                    "rank": row["rank"],
                }
            )
    return pd.DataFrame(rows)


def run_m2() -> Dict[str, pd.DataFrame]:
    """Run all MCDM outputs used in the report and dashboard."""
    growth_weights = np.array([0.25, 0.25, 0.10, 0.25, 0.05, 0.05, 0.05])
    inclusive_weights = np.array([0.10, 0.10, 0.25, 0.05, 0.25, 0.10, 0.15])
    return {
        "sector_default": sector_priority(),
        "sector_comparison": sector_priority_matrix(
            {
                "growth_orientation": growth_weights,
                "inclusive_orientation": inclusive_weights,
            }
        ),
        "sector_ai_sensitivity": ai_weight_sensitivity(),
        "region_topsis_expert": topsis_regions(),
        "region_topsis_entropy": topsis_regions_entropy(),
        "region_topsis_compare": compare_topsis_methods(),
    }
