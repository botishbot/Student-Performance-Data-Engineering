"""Reusable cleaning and feature-engineering pipeline for student performance data.

The module deliberately keeps the original observations: exact duplicate rows are
removed, but missing values and invalid records raise clear errors rather than
being silently imputed or discarded.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import NamedTuple

import pandas as pd


REQUIRED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu",
    "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime",
    "studytime", "failures", "schoolsup", "famsup", "paid",
    "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3",
]

NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 3),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}
NON_NEGATIVE_COLUMNS = ["absences"]
NUMERIC_COLUMNS = [*NUMERIC_RANGES, *NON_NEGATIVE_COLUMNS]

CATEGORY_VALUES = {
    "school": {"GP", "MS"},
    "sex": {"F", "M"},
    "address": {"U", "R"},
    "famsize": {"LE3", "GT3"},
    "Pstatus": {"T", "A"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": {"yes", "no"},
    "famsup": {"yes", "no"},
    "paid": {"yes", "no"},
    "activities": {"yes", "no"},
    "nursery": {"yes", "no"},
    "higher": {"yes", "no"},
    "internet": {"yes", "no"},
    "romantic": {"yes", "no"},
}


class PipelineResult(NamedTuple):
    cleaned: pd.DataFrame
    featured: pd.DataFrame
    input_rows: int
    duplicates_removed: int


def _numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    """Convert one numeric field for validation and derived calculations."""
    try:
        values = pd.to_numeric(data[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Column '{column}' contains non-numeric values.") from exc

    if not values.map(lambda value: math.isfinite(float(value))).all():
        raise ValueError(f"Column '{column}' must contain finite numbers.")
    if values.mod(1).ne(0).any():
        raise ValueError(f"Column '{column}' must contain whole numbers.")
    return values


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the exact source schema and values without changing the data."""
    if data.empty:
        raise ValueError("The dataset is empty.")

    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names found: {duplicated}")

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    unexpected_columns = sorted(set(data.columns) - set(REQUIRED_COLUMNS))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if unexpected_columns:
        raise ValueError(f"Unexpected columns found: {unexpected_columns}")

    missing_values = data[REQUIRED_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        details = ", ".join(
            f"{column}: {count}" for column, count in missing_values.items()
        )
        raise ValueError(f"Missing values found; resolve them before continuing ({details}).")

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        values = _numeric_series(data, column)
        invalid = ~values.between(minimum, maximum)
        if invalid.any():
            examples = values[invalid].head(5).tolist()
            raise ValueError(
                f"Column '{column}' must be between {minimum} and {maximum}; "
                f"invalid examples: {examples}"
            )

    for column in NON_NEGATIVE_COLUMNS:
        values = _numeric_series(data, column)
        if (values < 0).any():
            raise ValueError(f"Column '{column}' must contain non-negative numbers.")

    for column, allowed in CATEGORY_VALUES.items():
        actual = set(data[column].astype(str).unique())
        unexpected = sorted(actual - allowed)
        if unexpected:
            raise ValueError(
                f"Unexpected values in '{column}': {unexpected}. "
                f"Expected values from: {sorted(allowed)}"
            )


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows, validate, and normalise numeric columns."""
    input_rows = len(data)
    cleaned = data.drop_duplicates().copy().reset_index(drop=True)
    duplicates_removed = input_rows - len(cleaned)
    validate_dataset(cleaned)

    # Store numeric-looking Excel/text values as numbers before later calculations.
    for column in NUMERIC_COLUMNS:
        cleaned[column] = _numeric_series(cleaned, column)

    return cleaned, duplicates_removed


def engineer_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create the project's derived columns using its documented thresholds."""
    validate_dataset(cleaned)
    featured = cleaned.copy()

    # Keep this function safe when called directly with numeric strings.
    for column in NUMERIC_COLUMNS:
        featured[column] = _numeric_series(featured, column)

    featured["Grade_Category"] = pd.cut(
        featured["G3"],
        bins=[-1, 9, 14, 20],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype("object")

    featured["Attendance_Status"] = featured["absences"].map(
        lambda absences: "Regular" if absences <= 10 else "Irregular"
    )
    featured["Study_Load"] = featured["studytime"] - featured["failures"]
    featured["Pass_Fail"] = featured["G3"].map(
        lambda grade: "Pass" if grade >= 10 else "Fail"
    )

    return featured


def run_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    """Read the Excel source, process it, and write both CSV outputs."""
    project_root = Path(__file__).resolve().parents[1]
    source = Path(input_path) if input_path else project_root / "data/raw/DTS_202_Dataset.xlsx"
    destination = Path(output_dir) if output_dir else project_root / "data/processed"

    if not source.exists():
        raise FileNotFoundError(f"Source workbook not found: {source}")

    raw = pd.read_excel(source)
    cleaned, duplicates_removed = clean_data(raw)
    featured = engineer_features(cleaned)

    destination.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination / "student_performance_cleaned.csv", index=False)
    featured.to_csv(destination / "student_performance_featured.csv", index=False)

    return PipelineResult(cleaned, featured, len(raw), duplicates_removed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, clean, and engineer the student performance dataset."
    )
    parser.add_argument("--input", dest="input_path", help="Path to the source Excel workbook.")
    parser.add_argument("--output-dir", help="Directory for the processed CSV files.")
    args = parser.parse_args()

    result = run_pipeline(args.input_path, args.output_dir)
    print("Pipeline completed successfully.")
    print(f"Input records: {result.input_rows}")
    print(f"Exact duplicates removed: {result.duplicates_removed}")
    print(f"Cleaned dataset shape: {result.cleaned.shape}")
    print(f"Featured dataset shape: {result.featured.shape}")


if __name__ == "__main__":
    main()
