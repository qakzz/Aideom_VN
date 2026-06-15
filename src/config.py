"""Configuration layer for AIDEOM-VN.

This module keeps the mathematical constants, file paths, labels, and policy
scenario definitions in one place.  The report and the dashboard import from
this module instead of hard-coding values in many locations.  That makes the
project easier to audit, easier to explain to a lecturer, and safer to extend
when new data arrives.

The constants below are intentionally transparent.  They reproduce the values
in the final-exam problem statement and the teaching data set.  In a research
version, the same fields can be populated from an official data warehouse or
from parameter-estimation scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

COBB_DOUGLAS_COEFFICIENTS: Tuple[float, float, float, float, float] = (
    0.33,
    0.42,
    0.10,
    0.08,
    0.07,
)

COBB_DOUGLAS_NAMES: Tuple[str, str, str, str, str] = (
    "capital_K",
    "labor_L",
    "digital_D",
    "ai_capacity",
    "human_capital_H",
)

SECTOR_PRIORITY_DEFAULT_WEIGHTS = np.array(
    [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15], dtype=float
)

TOPSIS_EXPERT_WEIGHTS = np.array(
    [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10], dtype=float
)

REGION_NAMES = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]

REGION_SHORT_NAMES = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]

INVESTMENT_ITEMS = ["I", "D", "AI", "H"]

BETA_REGION_ITEM = np.array(
    [
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ],
    dtype=float,
)

INITIAL_DIGITAL_INDEX = np.array([38, 78, 55, 32, 82, 48], dtype=float)

SCENARIOS: Dict[str, np.ndarray] = {
    "S1 Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
    "S2 Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
    "S3 AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
    "S4 Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
    "S5 Tối ưu cân bằng": np.array([0.40, 0.25, 0.15, 0.20]),
}

SCENARIO_DESCRIPTIONS: Dict[str, str] = {
    "S1 Truyền thống": "Tập trung vốn vật chất, hạ tầng truyền thống và FDI.",
    "S2 Số hóa nhanh": "Ưu tiên chính phủ số, doanh nghiệp số và hạ tầng dữ liệu.",
    "S3 AI dẫn dắt": "Ưu tiên AI, dữ liệu lớn, bán dẫn và trung tâm dữ liệu.",
    "S4 Bao trùm số": "Ưu tiên vùng yếu, SME, giáo dục số và nông nghiệp số.",
    "S5 Tối ưu cân bằng": "Kịch bản thỏa hiệp giữa tăng trưởng, việc làm và rủi ro.",
}

@dataclass(frozen=True)
class BudgetPolicy:
    """Container for simple annual public investment assumptions."""

    annual_budget_trillion: float = 1000.0
    private_capital_growth: float = 0.04
    labor_growth: float = 0.006
    base_tfp_growth: float = 0.006
    digital_to_index_scale: float = 100.0
    ai_to_capacity_scale: float = 20.0
    human_to_percentage_scale: float = 200.0


@dataclass(frozen=True)
class FairnessPolicy:
    """Container for regional fairness constraints in the LP model."""

    gamma: float = 0.002
    lambda_target: float = 0.70
    total_budget: float = 50000.0
    region_floor: float = 5000.0
    region_cap: float = 12000.0
    human_floor: float = 12000.0


@dataclass(frozen=True)
class StochasticPolicy:
    """Container for two-stage stochastic programming scenario data."""

    first_stage_budget: float = 65000.0
    second_stage_budget: float = 15000.0
    probabilities: Tuple[float, float, float, float] = (0.30, 0.45, 0.20, 0.05)
    scenario_names: Tuple[str, str, str, str] = (
        "Lạc quan",
        "Cơ sở",
        "Bi quan",
        "Khủng hoảng",
    )


def ensure_directories() -> None:
    """Create output directories when running the project locally."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
