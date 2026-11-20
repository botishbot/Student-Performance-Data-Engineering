# Student Performance Data Engineering

A student project exploring a reproducible data-preparation workflow and a machine-learning experiment using student academic-performance data. The work is organised as a sequence of Jupyter notebooks, from initial inspection through cleaning, feature engineering, transformation, and model evaluation.

> **Project status:** Educational / exploratory project. The models and findings are not validated for real-world student placement or intervention decisions.

## Project objectives

- Inspect the source dataset and understand its structure and quality.
- Clean and validate the data before downstream use.
- Engineer additional features and document the assumptions behind them.
- Prepare categorical and numerical data for machine learning.
- Train and compare classification models for a project-defined Pass/Fail outcome.
- Save processed datasets, evaluation metrics, and visual outputs.

## Dataset

The repository includes the source workbook at `data/raw/DTS_202_Dataset.xlsx`.

The initial exploration notebook reports **395 records and 33 source columns**. It found no missing values or exact duplicate rows in the inspected dataset. The analysis also flagged 15 records with unusually high absence counts using the IQR rule. These records were inspected and retained because an unusual value is not automatically an invalid value.

The project creates a binary outcome from the final-grade field (`G3`):

- **Pass:** `G3 >= 10`
- **Fail:** `G3 < 10`

The threshold of 10 is a project assumption, not a rule established by the dataset itself or by a documented institutional grading policy. The project also creates grade bands and an attendance label; their boundaries are documented in the feature-engineering notebook and should be treated as analytical choices.

## Workflow

Run the notebooks in numerical order:

| Notebook | Stage | Main work |
|---|---|---|
| `01_data_exploration.ipynb` | Exploration | Inspect schema, data types, missing values, duplicates, distributions, potential outliers, and relationships |
| `02_data_cleaning.ipynb` | Cleaning | Create a working copy, remove exact duplicates if present, verify missing values, and export the cleaned CSV |
| `03_feature_engineering.ipynb` | Feature engineering | Add grade category, attendance status, study-load score, and Pass/Fail target |
| `04_data_transformation.ipynb` | Transformation design | Separate target and predictors; define one-hot encoding and numerical scaling |
| `05_model_preparation_and_training.ipynb` | Modelling | Split data, fit preprocessing on training data, train classifiers, evaluate, compare, and save outputs |

The notebooks use paths relative to the `notebooks/` directory. Open the notebook files from that folder and execute their cells from top to bottom.

## Repository structure

```text
Student-Performance-Data-Engineering/
├── data/
│   ├── raw/
│   │   └── DTS_202_Dataset.xlsx
│   └── processed/
│       ├── student_performance_cleaned.csv
│       └── student_performance_featured.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_data_transformation.ipynb
│   └── 05_model_preparation_and_training.ipynb
├── outputs/
│   ├── figures/
│   └── metrics/
├── .gitignore
└── requirements.txt
```

## Setup

Python **3.10 or newer** is recommended. A virtual environment is local to your machine and should not be committed to Git.

### 1. Clone the repository

```bash
git clone https://github.com/botishbot/Student-Performance-Data-Engineering.git
cd Student-Performance-Data-Engineering
```

### 2. Create and activate a virtual environment

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter lab
```

Open the `notebooks/` directory and run the notebooks in the order listed above. The raw workbook must remain at `data/raw/DTS_202_Dataset.xlsx` for the existing relative file paths to work.

## Model experiment

The modelling notebook compares three classifiers:

- Logistic Regression
- Decision Tree
- Random Forest

The current experiment uses a **70/30 stratified train/test split** with `random_state=42`. The preprocessing transformer is fitted on the training data and then applied to the test data, reducing preprocessing leakage.

The committed evaluation output reports:

| Model | Accuracy | Fail recall | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 89.92% | 92.31% | 88.96% |
| Decision Tree | 89.92% | 97.44% | 89.18% |
| Random Forest | 89.08% | 89.74% | 87.98% |

The Decision Tree has the highest Fail-class recall and macro F1 in this particular split. Logistic Regression ties its accuracy. These figures are results from one hold-out split, not proof that the models will perform similarly on new schools or student populations.

### Important modelling limitations

- The dataset is relatively small (395 rows).
- The evaluation uses one train/test split; cross-validation and uncertainty estimates have not yet been added.
- The predictors include earlier-period grades (`G1` and `G2`). Therefore, the experiment estimates the final outcome using information available after those earlier assessments; it is not a prediction made before any grades are known.
- The Pass/Fail threshold and engineered feature boundaries are project-defined assumptions.
- No hyperparameter search or external validation has been documented.
- Model performance does not establish causation. The results should not be used to make consequential decisions about individual students.

## Outputs

- Cleaned and feature-engineered datasets: `data/processed/`
- Model metrics: `outputs/metrics/model_comparison.csv`
- Model comparison chart and confusion matrices: `outputs/figures/`

The notebooks are the primary record of the current implementation. Re-run them to regenerate the outputs.

## Current scope and future improvements

The repository currently demonstrates a notebook-based data workflow and a small classification experiment. It is not yet a production data pipeline. Sensible next steps include:

1. Add automated data-quality checks (schema, allowed values, ranges, and row-count checks).
2. Move repeatable cleaning and feature-building logic into reusable Python modules or scripts.
3. Add cross-validation and compare results across multiple folds.
4. Save the fitted preprocessing/model pipeline and document how to reproduce predictions.
5. Add tests and a lightweight continuous-integration workflow.
6. Document the dataset's original source and usage terms.

## Author

Created by [botishbot](https://github.com/botishbot) as a student learning project.
