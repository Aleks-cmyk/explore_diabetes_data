# Predicting HbA1c from CGM and Clinical Features

A small, hands-on machine learning project on diabetes data. The goal was to learn how to work with glucose time series (continuous glucose monitoring, CGM) and clinical records, and to test how well the available features can predict HbA1c, the standard long-term blood glucose marker.

**Main finding:** the available features are not strongly predictive of HbA1c. The best held-out R² was about 0.15. The results are reported honestly below, including why I think this happened.

---

## Motivation

I am learning ML engineering and wanted a project that goes through the full workflow on real, messy medical data: exploration, data quality checks, feature engineering, model comparison with proper validation, and an honest assessment of results. Diabetes data is also a good way to get familiar with CGM readings, glucose metrics such as Time in Range, and the clinical meaning of HbA1c.

## Dataset

**Shanghai T1DM and T2DM dataset** (CGM time series with clinical summaries)

- **125 patients**: 16 with type 1 diabetes (T1DM), 109 with type 2 diabetes (T2DM)
- **CGM readings**: glucose in mg/dL, typically every 15 minutes, covering roughly 1 to 2 weeks per patient. Some patients have short gaps or irregular intervals.
- **Other time series**: sparse capillary glucose (CBG), insulin doses and medications, and dietary intake.
- **Summary record per patient**: HbA1c, age, gender, height, weight, BMI, smoking and alcohol history, diabetes duration, and laboratory values (lipids, creatinine, uric acid, and others).

Patients with a missing HbA1c (recorded as `/`) are excluded from modeling.

> Citation: `Zhu, Jinhao (2022). Diabetes Datasets-ShanghaiT1DM and ShanghaiT2DM. figshare. Dataset. https://doi.org/10.6084/m9.figshare.20444397.v3`

> Dataset source: `https://figshare.com/articles/dataset/Diabetes_Datasets-ShanghaiT1DM_and_ShanghaiT2DM/20444397?file=38259264`

## Approach

### 1. Exploratory data analysis (`notebooks/00_eda.ipynb`)
- Loaded and merged the T1DM and T2DM files. Column names differ slightly between them and had to be standardized.
- Checked data quality: missing dates, irregular CGM intervals, non-numeric placeholders in numeric columns, and very different recording lengths per patient.
- Computed the monitoring duration per patient and visualized it by diabetes type.
- Calculated Time in Range (TIR), Time Above Range (TAR) and Time Below Range (TBR) per patient, using full 24-hour days only so the percentages are comparable.

### 2. Feature engineering (`src/load_model_data.py`)
- **Glucose metrics**: TIR, TAR, TBR; CGM mean, standard deviation, coefficient of variation, quartiles, and range.
- **Variability and events**: counts of hypoglycemic (<70 mg/dL) and hyperglycemic (>180 mg/dL) readings, rapid glucose changes, and mean glucose excursion.
- **Clinical features**: age, gender (encoded 0/1), BMI, height, weight, smoking history, alcohol use (binary), diabetes type (binary), and diabetes duration.
- **CGM distribution**: every 5th percentile of each patient's CGM readings (21 values), which works for series of any length.

### 3. Dimensionality reduction (`notebooks/02_dim_reduction.ipynb`)
- Visualized patients and daily CGM profiles in 3D with PCA, UMAP and t-SNE, colored by HbA1c. A single script switches between the three methods.

### 4. Modeling (`notebooks/01_linear_model.ipynb`, `01_random_forest.ipynb`, `03_models_w_pca_features.ipynb`)
- Baseline: linear regression on TAR and TBR.
- Random Forest with hyperparameter search (GridSearchCV, 5-fold cross-validation on the training set only).
- Feature sets compared: TAR/TBR, extended CGM and clinical features, and PCA components of the CGM percentile profiles.
- **Leakage control**: the train/test split is made first, and PCA is fitted on the training data only.
- **Stability check**: the full pipeline is repeated over 5 random seeds, and results are reported as mean ± standard deviation instead of a single lucky split.

## Results and Conclusions

- **The available features are weak predictors of HbA1c.** Across linear regression, Random Forest, and PCA-based models, the best held-out R² was about 0.15, and the errors remain large in clinical terms.
- **Single correlations are modest.** TAR correlates with HbA1c at about r = 0.39 and TBR at about r = -0.19. No individual feature is strong.

**Likely reasons (hypotheses to verify):**
- **Timing mismatch.** HbA1c reflects average glucose over roughly the previous 2 to 3 months, while the CGM recordings cover only about 1 to 2 weeks. A short window cannot capture the full picture. HbA1c was also not necessarily measured after the CGM recording.
- **Small sample.** 125 patients, with only 16 T1DM, leaves little statistical power, and test-set estimates are noisy.
- **Limited clinical context.** Medication changes, adherence and other factors that drive HbA1c are only partly captured.

### Why I did not use deep learning on the CGM time series

The original plan included recurrent (RNN/LSTM) and transformer models trained directly on the CGM sequences. After exploring the data, I dropped this plan: with 125 patients, each contributing one series of varying length, there are far too few independent samples. A deep model would mostly memorize the training patients. Feature-based models were a better fit for this dataset size.

## Project Structure

```text
explore_diabetes_data/
├── data/
│   ├── raw/data/                 # Original T1DM/T2DM files as received
│   └── processed/
│       ├── data/                 # Files read by the data loaders
│       └── splits/               # Train/test splits written by prepare_splits.py
├── notebooks/                    # Analysis in order of the project story
│   ├── 00_eda.ipynb
│   ├── 01_linear_model.ipynb
│   ├── 01_random_forest.ipynb
│   ├── 02_dim_reduction.ipynb
│   └── 03_models_w_pca_features.ipynb
├── src/
│   ├── load_model_data.py        # Loads time series and summaries, builds features
│   ├── prepare_splits.py         # Train/test split, PCA fitted on train only, saving
│   └── data_utils.py             # Helpers: polynomial features, X/y extraction
├── docker/, docs/, tests/        # Placeholders for future deployment, docs and tests
├── pyproject.toml, uv.lock       # Dependencies managed with uv
├── CLAUDE.md                     # Project plan and working conventions
└── README.md
```

## Getting Started

**Requirements:** Python and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Aleks-cmyk/explore_diabetes_data
cd explore_diabetes_data

# Create the virtual environment and install all dependencies from uv.lock
uv sync

# Launch Jupyter and open the notebooks in order
uv run jupyter lab
```

To add a package, use `uv add <package>` (not `pip install`).

**Generate the train/test splits** (run once; this also writes the PCA features):

```bash
uv run python src/prepare_splits.py
```

**Load the modeling data in your own code:**

```python
import sys
sys.path.append('src')  # if running from the project root

from load_model_data import load_model_data
from prepare_splits import create_and_save_splits, load_splits

model_data = load_model_data()   # one row per patient, features + HbA1c
splits = load_splits()           # {'train': DataFrame, 'test': DataFrame}
```

If you run notebooks from the `notebooks/` folder, add `sys.path.append('../src')` before importing from `src`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Plotly, UMAP-learn, openpyxl and xlrd (Excel loading), Jupyter, uv.

## Limitations and Next Steps

- Collect longer CGM windows aligned with HbA1c measurement dates, which is the most likely way to improve predictions.
- Try a classification version: predict glucose control categories (for example, below or above 7%) instead of the exact value.


## Author

**Oleksandra Menzatiuk**

## AI Assistance

Claude (Anthropic) was used as a coding and teaching assistant during this project, for example to help write and debug code, check approaches, and draft parts of this README. All analysis decisions, code and conclusions were reviewed and understood by the author.
