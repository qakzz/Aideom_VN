"""End-to-end scenario pipeline for the dashboard and report."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import OUTPUT_DIR, SCENARIO_DESCRIPTIONS, SCENARIOS, ensure_directories
from .data_loader import load_macro
from .m1_cobb_douglas import scenario_forecast
from .m2_mcdm import run_m2
from .m3_optimization import run_m3
from .m4_labor import run_m4
from .m5_stochastic_pareto import run_m5


def simulate_2030_scenarios() -> pd.DataFrame:
    """Simulate the five policy scenarios in the problem statement."""
    macro = load_macro()
    rows = []
    for scenario, allocation in SCENARIOS.items():
        forecast = scenario_forecast(macro, allocation)
        cyber = 0.02 + 0.12 * allocation[2] - 0.03 * allocation[3]
        emission = 0.03 + 0.10 * allocation[0] + 0.08 * allocation[2] - 0.02 * allocation[3]
        netjob = 100000 * (0.7 * allocation[3] + 0.2 * allocation[1] - 0.3 * allocation[2])
        rows.append(
            {
                "scenario": scenario,
                "description": SCENARIO_DESCRIPTIONS[scenario],
                "GDP_2030": forecast["GDP_2030"],
                "D_2030": forecast["D_2030"],
                "AI_2030": forecast["AI_2030"],
                "H_2030": forecast["H_2030"],
                "CyberRisk": cyber,
                "EmissionIndex": emission,
                "NetJobIndex": netjob,
            }
        )
    return pd.DataFrame(rows)


def run_all_modules(include_rl: bool = False) -> Dict[str, object]:
    """Run the complete AIDEOM-VN computational pipeline."""
    from .m1_cobb_douglas import run_m1
    outputs = {
        "m1": run_m1(),
        "m2": run_m2(),
        "m3": run_m3(),
        "m4": run_m4(),
        "m5": run_m5(),
        "scenarios": simulate_2030_scenarios(),
    }
    if include_rl:
        from .m6_rl import run_m6
        outputs["m6"] = run_m6(episodes=1000)
    return outputs


def export_outputs(outputs: Dict[str, object] | None = None) -> None:
    """Export tables required for the report and dashboard."""
    ensure_directories()
    outputs = outputs or run_all_modules(include_rl=False)
    outputs["m1"]["fit"].data.to_csv(OUTPUT_DIR / "m1_macro_tfp.csv", index=False, encoding="utf-8-sig")
    outputs["m1"]["accounting"].to_csv(OUTPUT_DIR / "m1_growth_accounting.csv", index=False, encoding="utf-8-sig")
    outputs["m2"]["sector_default"].to_csv(OUTPUT_DIR / "m2_sector_priority.csv", index=False, encoding="utf-8-sig")
    outputs["m2"]["region_topsis_expert"].to_csv(OUTPUT_DIR / "m2_region_topsis.csv", index=False, encoding="utf-8-sig")
    outputs["m3"]["budget_sensitivity"].to_csv(OUTPUT_DIR / "m3_budget_sensitivity.csv", index=False, encoding="utf-8-sig")
    if outputs["m3"]["region_fair_relaxed"].success:
        outputs["m3"]["region_fair_relaxed"].variables.to_csv(OUTPUT_DIR / "m3_region_fair_lambda_068.csv", index=False, encoding="utf-8-sig")
    outputs["m3"]["project_scenarios"].to_csv(OUTPUT_DIR / "m3_project_scenarios.csv", index=False, encoding="utf-8-sig")
    outputs["m4"]["base"].allocation.to_csv(OUTPUT_DIR / "m4_labor_allocation.csv", index=False, encoding="utf-8-sig")
    outputs["m5"]["deterministic_comparison"].to_csv(OUTPUT_DIR / "m5_stochastic_compare.csv", index=False, encoding="utf-8-sig")
    outputs["scenarios"].to_csv(OUTPUT_DIR / "scenario_comparison_2030.csv", index=False, encoding="utf-8-sig")


def required_tables_manifest() -> pd.DataFrame:
    """Return the four core result tables required by the research report."""
    return pd.DataFrame(
        [
            {"table": "Bảng 1", "file": "m1_macro_tfp.csv", "content": "TFP, GDP thực tế và GDP dự báo"},
            {"table": "Bảng 2", "file": "m2_sector_priority.csv", "content": "Xếp hạng ưu tiên ngành"},
            {"table": "Bảng 3", "file": "m3_project_scenarios.csv", "content": "MIP lựa chọn dự án"},
            {"table": "Bảng 4", "file": "scenario_comparison_2030.csv", "content": "So sánh 5 kịch bản AIDEOM-VN"},
        ]
    )


def required_figures_manifest() -> pd.DataFrame:
    """Return the minimum five figures required by the research report."""
    return pd.DataFrame(
        [
            {"figure": "Hình 1", "file": "fig_tfp_trend.png", "content": "Xu hướng TFP"},
            {"figure": "Hình 2", "file": "fig_growth_accounting.png", "content": "Phân rã tăng trưởng"},
            {"figure": "Hình 3", "file": "fig_budget_sensitivity.png", "content": "Độ nhạy ngân sách"},
            {"figure": "Hình 4", "file": "fig_sector_priority.png", "content": "Ưu tiên ngành"},
            {"figure": "Hình 5", "file": "fig_topsis_regions.png", "content": "TOPSIS vùng"},
            {"figure": "Hình 6", "file": "fig_scenario_gdp2030.png", "content": "GDP 2030 theo kịch bản"},
        ]
    )
