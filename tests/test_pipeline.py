import pandas as pd
import pytest

from src.student_performance_pipeline import (
    REQUIRED_COLUMNS,
    clean_data,
    engineer_features,
    run_pipeline,
    validate_dataset,
)


def make_valid_data():
    rows = [
        {
            "school": "GP", "sex": "F", "age": 16, "address": "U",
            "famsize": "GT3", "Pstatus": "T", "Medu": 3, "Fedu": 2,
            "Mjob": "services", "Fjob": "other", "reason": "course",
            "guardian": "mother", "traveltime": 1, "studytime": 2,
            "failures": 0, "schoolsup": "no", "famsup": "yes",
            "paid": "no", "activities": "yes", "nursery": "yes",
            "higher": "yes", "internet": "yes", "romantic": "no",
            "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1,
            "Walc": 2, "health": 4, "absences": 10,
            "G1": 12, "G2": 13, "G3": 9,
        },
        {
            "school": "MS", "sex": "M", "age": 17, "address": "R",
            "famsize": "LE3", "Pstatus": "A", "Medu": 2, "Fedu": 3,
            "Mjob": "teacher", "Fjob": "health", "reason": "home",
            "guardian": "father", "traveltime": 2, "studytime": 3,
            "failures": 1, "schoolsup": "yes", "famsup": "no",
            "paid": "yes", "activities": "no", "nursery": "no",
            "higher": "yes", "internet": "no", "romantic": "yes",
            "famrel": 3, "freetime": 4, "goout": 5, "Dalc": 2,
            "Walc": 3, "health": 3, "absences": 11,
            "G1": 15, "G2": 16, "G3": 17,
        },
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_clean_data_removes_only_exact_duplicates():
    source = make_valid_data()
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    cleaned, removed = clean_data(source)

    assert len(cleaned) == 2
    assert removed == 1
    assert cleaned.duplicated().sum() == 0


def test_feature_engineering_uses_documented_boundaries():
    cleaned = make_valid_data()
    featured = engineer_features(cleaned)

    assert featured["Grade_Category"].tolist() == ["Low", "High"]
    assert featured["Attendance_Status"].tolist() == ["Regular", "Irregular"]
    assert featured["Study_Load"].tolist() == [2, 2]
    assert featured["Pass_Fail"].tolist() == ["Fail", "Pass"]


def test_missing_required_column_is_rejected():
    data = make_valid_data().drop(columns=["G3"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(data)


def test_unexpected_column_is_rejected():
    data = make_valid_data()
    data["extra_field"] = "unexpected"

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_dataset(data)


def test_missing_values_are_not_silently_imputed():
    data = make_valid_data()
    data.loc[0, "G3"] = None

    with pytest.raises(ValueError, match="Missing values found"):
        clean_data(data)


def test_out_of_range_grade_is_rejected():
    data = make_valid_data()
    data.loc[0, "G3"] = 21

    with pytest.raises(ValueError, match="must be between 0 and 20"):
        validate_dataset(data)


def test_fractional_grade_is_rejected():
    data = make_valid_data()
    data["G3"] = data["G3"].astype(float)
    data.loc[0, "G3"] = 9.5

    with pytest.raises(ValueError, match="whole numbers"):
        validate_dataset(data)


def test_non_finite_absence_value_is_rejected():
    data = make_valid_data()
    data["absences"] = data["absences"].astype(float)
    data.loc[0, "absences"] = float("inf")

    with pytest.raises(ValueError, match="finite numbers"):
        validate_dataset(data)


def test_numeric_text_is_normalised_before_feature_calculation():
    data = make_valid_data()
    data["studytime"] = data["studytime"].astype(str)
    data["failures"] = data["failures"].astype(str)

    featured = engineer_features(data)

    assert featured["Study_Load"].tolist() == [2, 2]
    assert pd.api.types.is_numeric_dtype(featured["studytime"])


def test_pipeline_reads_excel_and_writes_both_outputs(tmp_path):
    source = tmp_path / "source.xlsx"
    output = tmp_path / "processed"
    make_valid_data().to_excel(source, index=False)

    result = run_pipeline(source, output)

    cleaned_path = output / "student_performance_cleaned.csv"
    featured_path = output / "student_performance_featured.csv"
    assert cleaned_path.exists()
    assert featured_path.exists()
    assert result.input_rows == 2
    assert pd.read_csv(cleaned_path).shape == (2, 33)
    assert "Pass_Fail" in pd.read_csv(featured_path).columns


def test_committed_source_workbook_passes_pipeline_validation(tmp_path):
    from pathlib import Path

    repository_root = Path(__file__).resolve().parents[1]
    source = repository_root / "data" / "raw" / "DTS_202_Dataset.xlsx"
    assert source.exists(), "The committed source workbook is required for this integration test."

    result = run_pipeline(source, tmp_path / "actual-workbook-output")

    assert result.input_rows == 395
    assert result.duplicates_removed == 0
    assert result.cleaned.shape == (395, 33)
    assert result.featured.shape == (395, 37)
