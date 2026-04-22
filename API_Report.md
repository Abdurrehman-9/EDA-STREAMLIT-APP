# DataWrangler API — Technical Report

**Authors:** AbdurRehman   
**Dataset Used for Demonstration:** Car Features (11,914 rows × 16 columns)  
**Version:** 1.0.0

---

## 1. Overview

The **DataWrangler API** is a modular, dataset-agnostic Python library that packages every stage of a standard EDA workflow into seven self-contained, chainable classes. Each class follows a single-responsibility principle: it accepts a DataFrame, applies a focused set of transformations or analyses, and returns results in a reproducible, documented form. The API was developed against the Car Features dataset but is designed to work with any tabular CSV, Excel, JSON, or Parquet file.

---

## 2. Architecture — Seven Modules

### Module 1 · `DataLoader`
Handles file ingestion and the first structural inspection of a dataset.

**Key methods:**

| Method | Purpose |
|---|---|
| `load()` | Reads CSV / Excel / JSON / Parquet; auto-detects format from extension |
| `overview()` | Prints shape, memory usage, dtype, missing counts per column, and a 5-row sample |
| `column_types()` | Returns `{"numeric": [...], "categorical": [...]}` lists for downstream use |

**Usability note:** Callers can pass any `pandas` reader kwargs (e.g. `sep=';'`, `sheet_name=1`) via `**read_kwargs`, so the loader handles non-standard delimiters or multi-sheet workbooks without modification.

---

### Module 2 · `DataCleaner`
Addresses **inconsistencies and data-entry errors** — the most common source of downstream bias.

**Key methods:**

| Method | Strategy |
|---|---|
| `remove_duplicates()` | Drops exact-duplicate rows; configurable `subset` and `keep` policy |
| `normalise_strings()` | Strips whitespace and standardises case across all object columns |
| `fix_dtypes()` | Applies a user-supplied `{column: dtype}` map (e.g. treating `Year` as a label, not an integer) |
| `replace_values()` | Corrects specific known errors via a replacement dictionary |
| `flag_invalid_range()` | Returns a boolean mask of rows violating a domain constraint (e.g. `Engine HP < 0`) — surfaces errors before deciding on remediation |
| `drop_columns()` | Drops columns explicitly, or automatically when missing % exceeds a threshold |
| `print_log()` | Prints a numbered audit trail of every cleaning step |

**Applied to Car Features:**  
* 6 exact duplicate rows removed.  
* `Year` recast from `int64` to `object` (it is a nominal label, not a quantity).  
* `Market Category` dropped (≈ 30 % missing, no domain recovery feasible).  
* String normalisation applied to all object columns to resolve inconsistent capitalisation.

---

### Module 3 · `MissingValueHandler`
Provides **strategy-aware imputation** — the strategy is chosen based on the missingness mechanism (MCAR / MAR / MNAR) rather than applying a single blanket rule.

**Key methods:**

| Method | Strategy |
|---|---|
| `summary()` | Tabulates missing counts and % per column, sorted descending |
| `fill_constant()` | Fills with a fixed value (suitable for MCAR when the constant is domain-known) |
| `fill_statistic()` | Mean / median / mode imputation |
| `fill_by_group()` | Computes per-group statistics and fills within each group (MAR) |
| `fill_ffill_bfill()` | Forward / backward fill for ordered / time-series data |
| `fill_ml()` | Trains a **Random Forest** on complete rows and predicts missing values (MNAR) |
| `print_log()` | Audit trail for every imputation step |

**Applied to Car Features:**

| Column | Mechanism | Strategy |
|---|---|---|
| Engine Fuel Type | MCAR (confirmed by domain lookup: Suzuki Verona → "regular unleaded") | Group-mode by Make + Model, then constant fallback |
| Number of Doors | MNAR (model-specific configuration) | Domain-driven lookup (Tesla Model S → 4, Ferrari FF → 4), then mode-by-vehicle-style |
| Engine Cylinders | MAR (depends on Fuel Type and Vehicle Style) | Conditional fill: electric → 0, others → median within Fuel Type × Vehicle Style group |
| Engine HP | MNAR (but recoverable via other features) | Random Forest Regressor using MSRP, Cylinders, Make, Model, Size, Style, Transmission |

---

### Module 4 · `OutlierAnalyzer`
Detects and treats outliers while distinguishing **domain-valid extremes** from genuine errors.

**Key methods:**

| Method | Description |
|---|---|
| `detect_iqr()` | Flags values outside Q1 − k·IQR / Q3 + k·IQR (default k = 1.5) |
| `detect_zscore()` | Flags values where \|z\| > threshold (default 3.0) |
| `report()` | Summary table: outlier count and % per column |
| `winsorize()` | Caps extremes at user-defined percentile limits (default 5 % / 95 %) — **preserves row count** |
| `remove_outliers()` | Drops flagged rows (appropriate only when extremes are confirmed errors) |
| `plot()` | Boxplot + histogram side-by-side for each numeric column |

**Applied to Car Features:**  
Outliers were detected in all numeric columns. Analysis revealed that most extremes are domain-valid (Ferrari/Lamborghini HP, EV highway MPG, luxury MSRP). Winsorisation at 5 % limits was applied to `Engine HP`, `Engine Cylinders`, `highway MPG`, `city mpg`, `MSRP`, and `Popularity`, capping extremes without discarding data.

---

### Module 5 · `UnivariateAnalyzer`
Full **univariate distribution analysis** for both data types.

**Key methods:**

| Method | Output |
|---|---|
| `numeric_summary()` | Extended `describe()` augmented with skewness, kurtosis, and missing % |
| `plot_numeric()` | Histogram + KDE grid for all numeric columns |
| `categorical_summary()` | Top-N value-count table with percentages |
| `plot_categorical()` | Bar-chart grid for all categorical columns |
| `normality_test()` | Shapiro-Wilk test per numeric column (sampled to 5,000 rows for large datasets) |

**Key findings on Car Features:**

* MSRP is heavily right-skewed (skewness > 3); log transformation recommended before modelling.  
* Engine HP and Cylinders show bimodal distributions reflecting performance vs. economy segments.  
* Chevrolet, Ford, and Volkswagen dominate the `Make` column — notable class imbalance.  
* Regular unleaded and automatic transmission account for the majority of records.

---

### Module 6 · `BivariateAnalyzer`
**Bivariate and multivariate statistical analysis** linking features to each other and to a target variable.

**Key methods:**

| Method | Test / Chart |
|---|---|
| `correlation_heatmap()` | Pearson / Spearman / Kendall heatmap (lower triangle only) |
| `scatter_plot()` | Scatter with optional OLS regression line and hue grouping |
| `anova()` | One-way ANOVA for numeric target vs. categorical groups (statsmodels) |
| `tukey_hsd()` | Pairwise Tukey HSD post-hoc test |
| `chi_square_matrix()` | Pairwise Chi-Square p-value heatmap for categorical columns |
| `grouped_boxplot()` | Box plot of numeric target across top-N groups |

**Statistical findings on Car Features:**

**Pearson Correlations**
* Engine HP ↔ Cylinders: r = 0.77 (strong positive) — larger engines have more cylinders.  
* Engine HP / Cylinders ↔ MPG: r ≈ −0.60 (strong negative) — power reduces efficiency.  
* City MPG ↔ Highway MPG: r = 0.89 (very strong positive).  
* MSRP ↔ Engine HP: r ≈ 0.54 — powerful cars are priced higher.

**ANOVA (target = MSRP)**  
All tested categorical variables show statistically significant group differences (p < 0.001). `Make` and `Model` have the highest F-statistics, confirming that brand identity is the primary price driver.

**Tukey HSD**  
* Vehicle Size: all three size tiers (Compact, Midsize, Large) differ significantly from each other.  
* Transmission Type: Automated Manual is significantly different from all other types; Automatic vs. Direct Drive and Direct Drive vs. Manual are not significantly different.

**Chi-Square**  
All categorical pairs show near-zero p-values, indicating that categorical features in this dataset are statistically interdependent (e.g. Driven Wheels and Vehicle Size are not independent).

---

### Module 7 · `EDAReport`
A **convenience wrapper** that chains all six modules into a single pipeline call.

```python
report = EDAReport(df)
report.run_full_pipeline(
    target         = "MSRP",
    anova_groups   = ["Vehicle Size", "Transmission Type"],
    chi_sq_cols    = ["Driven_Wheels", "Vehicle Size"],
    winsorise_cols = ["Engine HP", "MSRP"],
    tukey_col      = "Vehicle Size",
)
```

This produces the complete set of summaries, plots, and statistical tests in sequence, suitable for a first-pass exploration of any new tabular dataset.

---

## 3. Usability with Other Datasets

The API is fully dataset-agnostic. Every method that needs a column list defaults to auto-detecting numeric or categorical columns from the DataFrame's `dtypes`, so no configuration is required for a new dataset. When domain knowledge is available, column lists can be passed explicitly to override the defaults.

**Minimum usage for a new dataset:**

```python
from data_wrangler_api import DataLoader, EDAReport

loader = DataLoader("my_dataset.csv")
df     = loader.load()
loader.overview()

EDAReport(df).run_full_pipeline(target="TargetColumn")
```

The ML-based imputation (`fill_ml`) automatically selects a `RandomForestRegressor` for numeric targets and a `RandomForestClassifier` for categorical ones, so it adapts to the target column's type without any change to the calling code.

---

## 4. File Structure

```
eda_api/
├── data_wrangler_api.py      ← Main API (7 modules, ~550 lines)
├── demo_car_features.py      ← Full end-to-end demo on Car Features dataset
└── API_Report.md             ← This report
```

---

## 5. Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Core data structures |
| `matplotlib`, `seaborn` | Static visualisations |
| `scipy` | IQR/z-score outlier detection, Shapiro-Wilk, Chi-Square, linregress |
| `statsmodels` | ANOVA (`ols`), Tukey HSD |
| `scikit-learn` | Random Forest imputation pipeline |
| `scipy.stats.mstats` | Winsorisation |

All packages are available in a standard Anaconda environment. Install any missing ones with `pip install statsmodels scikit-learn`.

---

## 6. Grading Rubric Alignment

| Criterion | Where Addressed |
|---|---|
| **Inconsistency / data-entry error strategy** | `DataCleaner`: `remove_duplicates`, `normalise_strings`, `fix_dtypes`, `flag_invalid_range`, `replace_values`, `drop_columns` |
| **Missing value strategy & execution** | `MissingValueHandler`: four strategies (constant, statistic, group-based, ML); applied column-by-column based on missingness mechanism |
| **Univariate analysis incl. outlier analysis** | `UnivariateAnalyzer` (distributions, normality tests) + `OutlierAnalyzer` (IQR detection, winsorisation, visual inspection) |
| **Bivariate / multivariate analysis & statistical testing** | `BivariateAnalyzer`: Pearson correlation heatmap, ANOVA, Tukey HSD, Chi-Square p-value matrix, scatter plots, grouped box plots |
