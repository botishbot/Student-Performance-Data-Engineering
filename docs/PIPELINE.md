# Data processing pipeline

The implementation lives in `src/student_performance_pipeline.py`.

- `validate_dataset`: checks required columns, missing values, numeric ranges, and categorical domains.
- `clean_data`: removes exact duplicate rows, then validates the remaining data.
- `engineer_features`: creates `Grade_Category`, `Attendance_Status`, `Study_Load`, and `Pass_Fail`.
- `run_pipeline`: reads the Excel workbook and writes the cleaned and featured CSV files.

The pipeline intentionally does not impute missing data or remove statistical outliers. Invalid input stops processing with a descriptive exception, leaving the source workbook untouched.

Run it from the repository root with `python -m src.student_performance_pipeline`. Run checks with `python -m pytest`.
