from src.m1_cobb_douglas import fit_average_tfp_model, forecast_2030_policy_target
from src.m2_mcdm import sector_priority, topsis_regions
from src.m3_optimization import solve_simple_budget, solve_project_selection, solve_region_budget
from src.m4_labor import solve_labor_budget
from src.m5_stochastic_pareto import solve_two_stage_sp
from src.scenario_pipeline import simulate_2030_scenarios


def test_tfp_positive():
    result = fit_average_tfp_model()
    assert (result.data["TFP_A"] > 0).all()
    assert result.mape < 20


def test_forecast_2030_reasonable():
    assert forecast_2030_policy_target() > 12000


def test_sector_priority_top_exists():
    ranking = sector_priority()
    assert len(ranking) == 10
    assert ranking.iloc[0]["Priority"] >= ranking.iloc[-1]["Priority"]


def test_topsis_regions_count():
    ranking = topsis_regions()
    assert len(ranking) == 6


def test_simple_budget_objective():
    result = solve_simple_budget()
    assert result["success"]
    assert round(result["objective"], 2) == 112.25


def test_region_lp_relaxed_feasible():
    result = solve_region_budget(lam=0.68, fair=True)
    assert result.success


def test_project_selection_feasible():
    result = solve_project_selection()
    assert result["success"]
    assert len(result["selected"]) >= 7


def test_labor_model():
    result = solve_labor_budget(sector_cap=9000)
    assert result.success
    assert result.objective > 0


def test_stochastic_model():
    result = solve_two_stage_sp()
    assert result.success
    assert result.objective > 0


def test_scenarios():
    df = simulate_2030_scenarios()
    assert len(df) == 5
    assert (df["GDP_2030"] > 0).all()
