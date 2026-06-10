# Stroke Prediction Model

A machine learning project that predicts stroke risk in patients using logistic regression. The model is designed to **prioritize recall** — minimizing missed stroke cases — making it suitable for clinical screening support.

---

## Project Structure

```
project/
├── data/
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   ├── model.pkl
│   └── threshold.pkl
├── images/
│   ├── stroke_vs_gender.png
│   ├── stroke_vs_residence_type.png
│   ├── stroke_vs_job_type.png
│   ├── stroke_vs_marriage_status.png
│   ├── stroke_vs_smoking_status.png
│   ├── stroke_vs_heart_disease.png
│   ├── stroke_vs_bmi.png
│   ├── distribution_of_bmi.png
│   ├── distribution_of_ages.png
│   └── tpr_vs_fpr.png
├── notebooks/
│   ├── preprocessing_and_eda.ipynb
│   └── model.ipynb
└── README.md
```

---

## Dataset

Source: [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle (`fedesoriano/stroke-prediction-dataset`), downloaded via `kagglehub`.

- **5,110 patients**, 12 columns (including target)
- **Class imbalance**: 4,861 non-stroke (95.1%) vs 249 stroke (4.9%)
- **Missing values**: `bmi` has 201 missing values; `smoking_status` has `"Unknown"` entries treated as NaN

---

## Notebooks

### 1. `preprocessing_and_eda.ipynb`

**Exploratory Data Analysis** — generates bar plots of stroke rate across various patient segments:

- Stroke vs Gender
- Stroke vs Residence Type
- Stroke vs Job Type
- Stroke vs Marriage Status
- Stroke vs Smoking Status
- Stroke vs Heart Disease
- Stroke vs BMI Group (Underweight / Normal / Overweight / Obese)
- Distribution of BMI values
- Distribution of Ages

**Preprocessing & Splitting:**
- Drops `id` and the temporary `bmi_group` columns
- Replaces `smoking_status = "Unknown"` with NaN
- Stratified 80/20 train/test split (`random_state=42`)
- Saves processed splits to `data/processed/`

---

### 2. `model.ipynb`

**Pipeline** (via `imblearn.Pipeline`):

| Step | Details |
|---|---|
| Categorical preprocessing | Mode imputation + One-Hot Encoding (`drop="if_binary"`) |
| Numerical preprocessing | Median imputation + Standard Scaling |
| Binary preprocessing | Mode imputation (pass-through) |
| Class imbalance | SMOTE oversampling (`random_state=42`) |
| Model | `LogisticRegression(max_iter=10000)` |

**Hyperparameter Tuning** via `RandomizedSearchCV` (21 iterations, 3-fold CV, scored on ROC-AUC):

```python
{
    "model__C"            : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "model__l1_ratio"     : [0.1, 0.5, 1],
    "model__solver"       : ["saga"],
    "model__class_weight" : ["balanced", None]
}
```

**Threshold Optimization** — the decision threshold is tuned to maximize precision while enforcing recall ≥ 0.94:

- Best threshold: `~0.164`
- Best recall: `0.96`
- Confusion matrix: `[[432, 540], [2, 48]]`

The model trades high false positives for near-complete capture of true stroke cases — appropriate for medical screening.

Serializes `model.pkl` and `threshold.pkl` to the `models/` directory.

---

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn kagglehub
```

---

## Usage

### Step 1 — Run EDA & Preprocessing

Open and run `notebooks/preprocessing_and_eda.ipynb`. This downloads the dataset, generates EDA plots, and saves the processed train/test splits.

### Step 2 — Train the Model

Open and run `notebooks/model.ipynb`. This trains the pipeline, tunes the threshold, and saves `model.pkl` and `threshold.pkl`.

**Output:**
```
Stroke Probability : 0.77
Risk Level         : High Risk
```

---

## Notes

- It is not a medical tool. It is just a student project a practice project.
- Feature names and values must match exactly (e.g., `'Residence_type'` with capital R, `'formerly smoked'` lowercase).
- Missing values in input are handled automatically by the pipeline's imputers.
