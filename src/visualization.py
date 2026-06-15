"""Visualization functions for reports and dashboard outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .config import OUTPUT_DIR, ensure_directories


def save_tfp_plot(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    path = path or OUTPUT_DIR / "fig_tfp_trend.png"
    plt.figure(figsize=(6, 4))
    plt.plot(df["year"], df["TFP_A"], marker="o")
    plt.xlabel("Năm")
    plt.ylabel("TFP A_t")
    plt.title("Xu hướng TFP 2020-2025")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_growth_accounting_plot(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    path = path or OUTPUT_DIR / "fig_growth_accounting.png"
    plt.figure(figsize=(7, 4))
    plt.bar(df["factor"], df["share_pct"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Tỷ trọng đóng góp (%)")
    plt.title("Phân rã tăng trưởng bình quân")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_priority_plot(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    path = path or OUTPUT_DIR / "fig_sector_priority.png"
    d = df.sort_values("Priority").copy()
    plt.figure(figsize=(8, 4))
    plt.barh(d["sector_name_vi"], d["Priority"])
    plt.xlabel("Priority")
    plt.title("Xếp hạng ưu tiên ngành")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_topsis_plot(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    path = path or OUTPUT_DIR / "fig_topsis_regions.png"
    plt.figure(figsize=(7, 4))
    plt.bar(df["region_name_vi"], df["TOPSIS_score"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("C*")
    plt.title("TOPSIS ưu tiên đầu tư AI theo vùng")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_scenario_plot(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    path = path or OUTPUT_DIR / "fig_scenario_gdp2030.png"
    plt.figure(figsize=(8, 4))
    plt.bar(df["scenario"], df["GDP_2030"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("GDP 2030")
    plt.title("So sánh GDP 2030 theo kịch bản")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_budget_sensitivity_plot(df: pd.DataFrame, path: Path | None = None) -> Path:
    ensure_directories()
    path = path or OUTPUT_DIR / "fig_budget_sensitivity.png"
    plt.figure(figsize=(6, 4))
    plt.plot(df["budget"], df["objective"], marker="o")
    plt.xlabel("Ngân sách")
    plt.ylabel("Z*")
    plt.title("Độ nhạy ngân sách LP")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def save_all_figures(outputs: dict) -> list[Path]:
    """Save the standard five-plus figures used in the research report."""
    paths = []
    paths.append(save_tfp_plot(outputs["m1"]["fit"].data))
    paths.append(save_growth_accounting_plot(outputs["m1"]["accounting"]))
    paths.append(save_budget_sensitivity_plot(outputs["m3"]["budget_sensitivity"]))
    paths.append(save_priority_plot(outputs["m2"]["sector_default"]))
    paths.append(save_topsis_plot(outputs["m2"]["region_topsis_expert"]))
    paths.append(save_scenario_plot(outputs["scenarios"]))
    return paths
