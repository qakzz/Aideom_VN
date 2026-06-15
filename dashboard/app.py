"""AIDEOM-VN local dashboard.

Run from the repository root:
    streamlit run dashboard/app.py

The command will create a local browser link, usually:
    http://localhost:8501
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import data_catalog, load_macro, load_projects, load_regions, load_sectors
from src.m1_cobb_douglas import fit_average_tfp_model, forecast_2030_policy_target, growth_accounting
from src.m2_mcdm import compare_topsis_methods, sector_priority, topsis_regions
from src.m3_optimization import (
    budget_sensitivity,
    diagnose_region_fairness,
    project_scenario_table,
    solve_region_budget,
)
from src.m4_labor import solve_labor_budget, vulnerable_labor_flow_table
from src.m5_stochastic_pareto import deterministic_equivalent_table, solve_two_stage_sp
from src.scenario_pipeline import required_figures_manifest, required_tables_manifest, simulate_2030_scenarios

st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px;}
        h1 {font-size: 2.65rem !important; letter-spacing: 0.02em; font-weight: 800 !important;}
        h2 {font-size: 1.55rem !important; font-weight: 750 !important; margin-top: 1.4rem;}
        h3 {font-size: 1.15rem !important; font-weight: 700 !important;}
        [data-testid="stMetricValue"] {font-size: 2.05rem; font-weight: 700;}
        [data-testid="stMetricLabel"] {font-size: 0.96rem; color: #4f5b67;}
        div[data-testid="stHorizontalBlock"] > div {padding-right: 0.4rem;}
        .hero-subtitle {font-size: 1.02rem; color: #6b7280; margin-top: -0.8rem; margin-bottom: 1.2rem;}
        .section-note {font-size: 0.95rem; color: #5b6472; line-height: 1.55;}
        .policy-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 18px; background: #ffffff; min-height: 112px;}
        .policy-card b {font-size: 1.02rem;}
        .ok-pill {display: inline-block; padding: 3px 10px; border-radius: 999px; background: #e8f7ee; color: #166534; font-weight: 700;}
        .warn-pill {display: inline-block; padding: 3px 10px; border-radius: 999px; background: #fff4e5; color: #9a3412; font-weight: 700;}
        .small-caption {font-size: 0.85rem; color: #6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_all_data() -> dict:
    return {
        "macro": load_macro(),
        "sectors": load_sectors(),
        "regions": load_regions(),
        "projects": load_projects(),
    }


@st.cache_data(show_spinner=False)
def compute_dashboard(lam: float, budget_min: int, budget_max: int, budget_step: int) -> dict:
    data = load_all_data()
    fit = fit_average_tfp_model(data["macro"])
    forecast_2030 = forecast_2030_policy_target(data["macro"])
    acc = growth_accounting(fit.data)
    priority = sector_priority(data["sectors"])
    topsis = topsis_regions(data["regions"])
    topsis_compare = compare_topsis_methods()
    budgets = list(range(int(budget_min), int(budget_max) + 1, int(budget_step)))
    budget_df = budget_sensitivity(budgets)
    region_lp = solve_region_budget(lam=lam, fair=True)
    fairness_diag = diagnose_region_fairness(lam)
    project_df = project_scenario_table()
    labor = solve_labor_budget(sector_cap=9000)
    sp = solve_two_stage_sp()
    stochastic_compare = deterministic_equivalent_table()
    scenarios = simulate_2030_scenarios()
    return {
        "data": data,
        "fit": fit,
        "forecast_2030": forecast_2030,
        "accounting": acc,
        "priority": priority,
        "topsis": topsis,
        "topsis_compare": topsis_compare,
        "budget": budget_df,
        "region_lp": region_lp,
        "fairness_diag": fairness_diag,
        "projects": project_df,
        "labor": labor,
        "sp": sp,
        "stochastic_compare": stochastic_compare,
        "scenarios": scenarios,
    }


def format_number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta)


def allocation_heatmap(df: pd.DataFrame) -> go.Figure:
    matrix = df[["I", "D", "AI", "H"]].to_numpy(float)
    fig = px.imshow(
        matrix,
        labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
        x=["I", "D", "AI", "H"],
        y=df["region"],
        text_auto=".0f",
        aspect="auto",
        title="Heatmap phân bổ ngân sách theo vùng - hạng mục",
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def scenario_radar(df: pd.DataFrame) -> go.Figure:
    normalized = df.copy()
    metrics = ["GDP_2030", "D_2030", "AI_2030", "H_2030", "NetJobIndex"]
    for metric in metrics:
        s = normalized[metric]
        normalized[metric] = (s - s.min()) / (s.max() - s.min() + 1e-12)
    fig = go.Figure()
    for _, row in normalized.iterrows():
        values = [row[m] for m in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=row["scenario"],
            )
        )
    fig.update_layout(title="Hồ sơ tương đối của 5 kịch bản", polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=520)
    return fig


with st.sidebar:
    st.header("Thiết lập mô hình")
    lam = st.slider("Fairness λ cho LP vùng", min_value=0.60, max_value=0.75, value=0.68, step=0.01)
    st.caption("λ=0.70 trong đề có thể làm mô hình vùng bị infeasible; mặc định 0.68 để dashboard có nghiệm minh họa.")
    budget_min = st.number_input("Ngân sách LP từ", min_value=50, max_value=200, value=100, step=10)
    budget_max = st.number_input("Ngân sách LP đến", min_value=60, max_value=250, value=140, step=10)
    budget_step = st.number_input("Bước ngân sách", min_value=5, max_value=50, value=20, step=5)
    show_raw = st.toggle("Hiện dữ liệu gốc", value=False)

bundle = compute_dashboard(lam, budget_min, budget_max, budget_step)
macro = bundle["data"]["macro"]
fit = bundle["fit"]
scenarios = bundle["scenarios"]
region_lp = bundle["region_lp"]
labor = bundle["labor"]
sp = bundle["sp"]

st.title("AIDEOM-VN Vietnam AI-Era Economic Decision Support Dashboard")
st.markdown(
    "<div class='hero-subtitle'>Dashboard nguyên mẫu hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI, sử dụng dữ liệu và mô hình trong báo cáo AIDEOM-VN.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "1. Tổng quan",
        "2. Dự báo vĩ mô",
        "3. Ngành & vùng ưu tiên",
        "4. Tối ưu phân bổ",
        "5. Lao động & NetJob",
        "6. Rủi ro & Kịch bản",
        "7. Khuyến nghị chính sách",
    ]
)

with tabs[0]:
    st.subheader("Tổng quan kết quả mô hình")
    best_scenario = scenarios.sort_values("GDP_2030", ascending=False).iloc[0]
    metric_row(
        [
            ("GDP 2025 thực tế (nghìn tỷ VND)", format_number(macro["GDP_trillion_VND"].iloc[-1], 1), None),
            ("GDP 2030 dự báo mục tiêu", format_number(bundle["forecast_2030"], 1), None),
            ("MAPE mô hình", f"{fit.mape:.2f}%", None),
            ("TFP trung bình", f"{fit.average_tfp:.4f}", None),
        ]
    )
    st.divider()
    st.subheader("Kết quả tối ưu ngân sách M3")
    if region_lp.success:
        status = "Optimal"
        budget_used = region_lp.variables[["I", "D", "AI", "H"]].sum().sum()
        gdp_gain = region_lp.objective
    else:
        status = "Infeasible"
        budget_used = 0
        gdp_gain = 0
    metric_row(
        [
            ("Trạng thái tối ưu", status, None),
            ("Ngân sách sử dụng (tỷ VND)", format_number(budget_used, 0), None),
            ("GDP gain kỳ vọng", format_number(gdp_gain, 0), None),
            ("Fairness λ", f"{lam:.2f}", None),
        ]
    )
    if not region_lp.success:
        st.warning("Với λ hiện tại, LP vùng không khả thi. Xem chẩn đoán ở tab 4.")
    st.divider()
    st.subheader("Kết quả lao động M4")
    if labor.success:
        total_displaced = labor.allocation["DisplacedJob"].sum()
        total_retrain = labor.allocation["RetrainingCapacity"].sum()
        metric_row(
            [
                ("Trạng thái M4", "Optimal", None),
                ("Tổng NetJob", format_number(labor.objective, 0), None),
                ("Việc làm bị dịch chuyển", format_number(total_displaced, 0), None),
                ("Năng lực đào tạo lại", format_number(total_retrain, 0), None),
            ]
        )
    st.divider()
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(px.bar(scenarios, x="scenario", y="GDP_2030", title="GDP 2030 theo 5 kịch bản"), use_container_width=True)
    with c2:
        st.markdown("### Kịch bản nổi bật")
        st.markdown(
            f"""
            <div class='policy-card'>
            <b>{best_scenario['scenario']}</b><br>
            GDP 2030 cao nhất trong mô phỏng: <b>{best_scenario['GDP_2030']:,.1f}</b> nghìn tỷ VND.<br><br>
            <span class='ok-pill'>Ưu tiên tăng trưởng</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tabs[1]:
    st.subheader("Dự báo vĩ mô và TFP")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(px.line(fit.data, x="year", y="TFP_A", markers=True, title="TFP A_t 2020-2025"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(bundle["accounting"], x="factor", y="share_pct", title="Phân rã tăng trưởng bình quân 2020-2025"), use_container_width=True)
    st.dataframe(fit.data, use_container_width=True)

with tabs[2]:
    st.subheader("Ngành và vùng ưu tiên")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        priority = bundle["priority"]
        st.plotly_chart(px.bar(priority, x="sector_name_vi", y="Priority", title="Priority ngành"), use_container_width=True)
        st.dataframe(priority[["sector_name_vi", "Priority", "rank"]], use_container_width=True)
    with c2:
        topsis = bundle["topsis"]
        st.plotly_chart(px.bar(topsis, x="region_name_vi", y="TOPSIS_score", title="TOPSIS vùng ưu tiên AI"), use_container_width=True)
        st.dataframe(topsis[["region_name_vi", "TOPSIS_score", "rank"]], use_container_width=True)
    st.markdown("### So sánh TOPSIS trọng số chuyên gia và Entropy")
    st.dataframe(bundle["topsis_compare"], use_container_width=True)

with tabs[3]:
    st.subheader("Tối ưu phân bổ ngân sách")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(px.line(bundle["budget"], x="budget", y="objective", markers=True, title="Độ nhạy ngân sách LP 4 hạng mục"), use_container_width=True)
        st.dataframe(bundle["budget"], use_container_width=True)
    with c2:
        if region_lp.success:
            st.plotly_chart(allocation_heatmap(region_lp.variables), use_container_width=True)
        else:
            st.error(region_lp.message)
            st.dataframe(bundle["fairness_diag"], use_container_width=True)
    st.markdown("### MIP lựa chọn dự án chuyển đổi số")
    st.dataframe(bundle["projects"], use_container_width=True)

with tabs[4]:
    st.subheader("Lao động, NetJob và đào tạo lại")
    if labor.success:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(px.bar(labor.allocation, x="sector", y="NetJob", title="NetJob theo ngành"), use_container_width=True)
        with c2:
            long_df = labor.allocation.melt(id_vars="sector", value_vars=["x_AI", "x_H"], var_name="investment", value_name="value")
            st.plotly_chart(px.bar(long_df, x="sector", y="value", color="investment", barmode="group", title="Phân bổ x_AI và x_H"), use_container_width=True)
        st.dataframe(labor.allocation, use_container_width=True)
    else:
        st.error(labor.message)
    st.markdown("### Nhóm lao động dễ tổn thương")
    st.dataframe(vulnerable_labor_flow_table(), use_container_width=True)

with tabs[5]:
    st.subheader("Rủi ro, bất định và kịch bản")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(scenario_radar(scenarios), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(scenarios, x="CyberRisk", y="EmissionIndex", size="GDP_2030", color="scenario", title="Đánh đổi rủi ro - môi trường - tăng trưởng"), use_container_width=True)
    st.markdown("### Stochastic programming hai giai đoạn")
    if sp.success:
        m1, m2 = st.columns([1, 1])
        with m1:
            st.write("First-stage")
            st.dataframe(sp.first_stage, use_container_width=True)
        with m2:
            st.write("Second-stage recourse")
            st.dataframe(sp.second_stage, use_container_width=True)
    st.markdown("### So sánh chiến lược dưới bất định")
    st.dataframe(bundle["stochastic_compare"], use_container_width=True)

with tabs[6]:
    st.subheader("Khuyến nghị chính sách")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class='policy-card'>
            <b>1. Ưu tiên số hóa nhanh nhưng không bỏ quên bao trùm</b><br>
            S2 cho GDP cao, S4 tốt cho NetJob. Chính sách nên phối hợp hai hướng này thay vì chọn cực đoan.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class='policy-card'>
            <b>2. AI cần đi kèm nhân lực số</b><br>
            Đầu tư AI làm tăng rủi ro dịch chuyển việc làm; x_H cần được xem như điều kiện hấp thụ công nghệ.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class='policy-card'>
            <b>3. Kiểm tra tính khả thi của công bằng vùng</b><br>
            λ quá cao có thể khiến LP vô nghiệm. Cần gắn mục tiêu công bằng với ngân sách và xuất phát điểm vùng.
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("### Kiểm tra yêu cầu báo cáo")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(required_tables_manifest(), use_container_width=True)
    with c2:
        st.dataframe(required_figures_manifest(), use_container_width=True)

if show_raw:
    st.divider()
    st.subheader("Dữ liệu gốc")
    raw_tabs = st.tabs(["Macro", "Sectors", "Regions", "Projects", "Catalog"])
    with raw_tabs[0]:
        st.dataframe(load_macro(), use_container_width=True)
    with raw_tabs[1]:
        st.dataframe(load_sectors(), use_container_width=True)
    with raw_tabs[2]:
        st.dataframe(load_regions(), use_container_width=True)
    with raw_tabs[3]:
        st.dataframe(load_projects(), use_container_width=True)
    with raw_tabs[4]:
        st.dataframe(data_catalog(), use_container_width=True)
