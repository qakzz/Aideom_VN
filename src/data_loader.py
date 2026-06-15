"""Data loading and validation helpers for AIDEOM-VN.

The final-exam problem provides a compact teaching data set.  This module wraps
those CSV files with explicit schema checks.  The point is not only to read
files, but also to make the computational pipeline reproducible: every model
starts from a checked dataframe with consistent names, numeric types, and units.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .config import DATA_DIR


@dataclass
class DataSchema:
    """Expected structure for a CSV file."""

    filename: str
    required_columns: List[str]

    def path(self) -> Path:
        return DATA_DIR / self.filename


MACRO_SCHEMA = DataSchema(
    "vietnam_macro_2020_2025.csv",
    [
        "year",
        "GDP_trillion_VND",
        "K_accum_trillion_VND",
        "L_million",
        "digital_GDP_pct",
        "AI_thousand_digital_firms",
        "trained_labor_pct",
    ],
)

SECTOR_SCHEMA = DataSchema(
    "vietnam_sectors_2024.csv",
    [
        "sector_name_vi",
        "growth_rate_2024_pct",
        "productivity_million_VND_per_worker",
        "spillover_coef_0_1",
        "export_billion_USD",
        "labor_million",
        "ai_readiness_0_100",
        "automation_risk_pct",
    ],
)

REGION_SCHEMA = DataSchema(
    "vietnam_regions_2024.csv",
    [
        "region_name_vi",
        "grdp_per_capita_million_VND",
        "fdi_registered_billion_USD",
        "digital_index_0_100",
        "ai_readiness_0_100",
        "trained_labor_pct",
        "rd_intensity_pct",
        "internet_penetration_pct",
        "gini_coef",
    ],
)

PROJECT_SCHEMA = DataSchema(
    "vietnam_projects_2026_2030.csv",
    [
        "project_id",
        "project_name_vi",
        "field",
        "cost_billion_VND",
        "benefit_NPV_billion_VND",
        "cost_year_1_2",
        "cost_year_3_5",
    ],
)


def read_csv(schema: DataSchema) -> pd.DataFrame:
    """Read a CSV and validate its required columns."""
    path = schema.path()
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    validate_columns(df, schema.required_columns, schema.filename)
    return df


def validate_columns(df: pd.DataFrame, required: Iterable[str], name: str = "dataframe") -> None:
    """Raise a clear error when a dataframe schema is not compatible."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric_columns(df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
    """Return numeric columns after excluding identifiers."""
    exclude_set = set(exclude)
    return [col for col in df.columns if col not in exclude_set and pd.api.types.is_numeric_dtype(df[col])]


def coerce_numeric(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Convert possible numeric columns to float while preserving labels."""
    out = df.copy()
    for col in out.columns:
        if col in set(exclude):
            continue
        try:
            out[col] = pd.to_numeric(out[col])
        except Exception:
            pass
    return out


def load_macro() -> pd.DataFrame:
    """Load Vietnam macro data for 2020-2025."""
    df = read_csv(MACRO_SCHEMA)
    df = coerce_numeric(df, exclude=[])
    return df.sort_values("year").reset_index(drop=True)


def load_sectors() -> pd.DataFrame:
    """Load sector-level data for 2024."""
    return coerce_numeric(read_csv(SECTOR_SCHEMA), exclude=["sector_name_vi"])


def load_regions() -> pd.DataFrame:
    """Load regional readiness data for 2024."""
    return coerce_numeric(read_csv(REGION_SCHEMA), exclude=["region_name_vi"])


def load_projects() -> pd.DataFrame:
    """Load project-selection data for the MIP module."""
    return coerce_numeric(read_csv(PROJECT_SCHEMA), exclude=["project_id", "project_name_vi", "field"])


def data_catalog() -> pd.DataFrame:
    """Return a compact catalog of all data files used by the prototype."""
    rows = []
    for schema in [MACRO_SCHEMA, SECTOR_SCHEMA, REGION_SCHEMA, PROJECT_SCHEMA]:
        path = schema.path()
        rows.append(
            {
                "file": schema.filename,
                "exists": path.exists(),
                "columns": len(schema.required_columns),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def describe_dataset(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Create a research-report-friendly numeric summary table."""
    if label_col and label_col in df.columns:
        data = df.drop(columns=[label_col])
    else:
        data = df
    summary = data.describe().T.reset_index().rename(columns={"index": "variable"})
    return summary


def assert_no_missing(df: pd.DataFrame, name: str) -> None:
    """Validate that the teaching data set has no missing values."""
    missing = int(df.isna().sum().sum())
    if missing:
        raise ValueError(f"{name} contains {missing} missing values")


def load_all() -> dict:
    """Load all project data in one dictionary."""
    bundle = {
        "macro": load_macro(),
        "sectors": load_sectors(),
        "regions": load_regions(),
        "projects": load_projects(),
    }
    for name, df in bundle.items():
        assert_no_missing(df, name)
    return bundle
