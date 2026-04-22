"""
=============================================================================
DEMO  –  DataWrangler API applied to the Car Features Dataset
=============================================================================
Run each section independently; every function is self-contained.
=============================================================================
"""

# ── 0. Setup ─────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # make API importable

from data_wrangler_api import (
    DataLoader, DataCleaner, MissingValueHandler,
    OutlierAnalyzer, UnivariateAnalyzer, BivariateAnalyzer, EDAReport
)
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  |  Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 – Load Data")
print("=" * 60)

loader = DataLoader("Car Features.xlsx - in.csv")
df     = loader.load()
loader.overview()
col_types = loader.column_types()
print("Numeric columns :", col_types["numeric"])
print("Categorical cols:", col_types["categorical"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  |  Data Cleaning  (inconsistencies & data-entry errors)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 – Data Cleaning")
print("=" * 60)

cleaner = DataCleaner(df)

# 2a. Duplicates
cleaner.remove_duplicates()

# 2b. Type coercions  (Year is better treated as a label, not a number)
cleaner.fix_dtypes({"Year": "object"})

# 2c. Normalise string columns (strip extra spaces, consistent case)
cleaner.normalise_strings()

# 2d. Data-entry / inconsistency errors – flag impossible values
cleaner.flag_invalid_range("Engine HP", lo=0, hi=2000)
cleaner.flag_invalid_range("city mpg",  lo=0, hi=150)

# 2e. Drop Market Category: >30 % missing (no domain recovery possible)
cleaner.drop_columns(["Market Category"], threshold_pct=30)

cleaner.print_log()
df = cleaner.get_clean_df()
print(f"Shape after cleaning: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  |  Missing Value Handling
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 – Missing Value Handling")
print("=" * 60)

mvh = MissingValueHandler(df)
mvh.summary()

# 3a. Engine Fuel Type  →  MCAR; impute by group (Make + Model)
#     Suzuki Verona historically uses "regular unleaded"
mvh.fill_by_group("Engine Fuel Type",
                  group_cols=["Make", "Model", "Year"],
                  strategy="mode")
# Fallback constant for any remaining NaN
mvh.fill_constant("Engine Fuel Type", "regular unleaded")

# 3b. Number of Doors  →  MNAR (model-specific); known values from domain
#     Tesla Model S = 4, Ferrari FF = 4 (grand tourer)
df_tmp = mvh.get_df()

def impute_doors(df):
    """Context-aware door imputation using known vehicle characteristics."""
    # Tesla Model S → 4 doors
    mask_tesla = (df["Make"] == "tesla") & (df["Model"] == "model s")
    df.loc[mask_tesla & df["Number of Doors"].isna(), "Number of Doors"] = 4
    # Ferrari FF → 4 doors (grand tourer)
    mask_ff = (df["Make"] == "ferrari") & (df["Model"] == "ff")
    df.loc[mask_ff & df["Number of Doors"].isna(), "Number of Doors"] = 4
    # Generic fallback: mode within Vehicle Style
    df["Number of Doors"] = df.groupby("Vehicle Style")["Number of Doors"]\
                              .transform(lambda x: x.fillna(x.mode().iloc[0]
                                         if len(x.mode()) else 4))
    return df

df_tmp = impute_doors(df_tmp)
mvh.df = df_tmp   # push result back so the handler tracks the state

# 3c. Engine Cylinders  →  MAR (depends on fuel type & vehicle style)
#     Electric vehicles → 0 cylinders
mvh.fill_constant.__func__  # just to confirm method is accessible
mvh.df.loc[mvh.df["Engine Fuel Type"] == "electric", "Engine Cylinders"] = \
    mvh.df.loc[mvh.df["Engine Fuel Type"] == "electric", "Engine Cylinders"].fillna(0)

# Premium unleaded → modal cylinders from similar vehicles = 4
mvh.fill_by_group("Engine Cylinders",
                  group_cols=["Engine Fuel Type", "Vehicle Style"],
                  strategy="median")

# 3d. Engine HP  →  MNAR but recoverable via ML (MSRP, cylinders, make/model)
mvh.fill_ml(
    target_col   = "Engine HP",
    num_features = ["MSRP", "Engine Cylinders"],
    cat_features = ["Make", "Model", "Vehicle Size",
                    "Vehicle Style", "Transmission Type"],
)

mvh.print_log()
mvh.summary()
df = mvh.get_df()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  |  Outlier Analysis (Univariate component)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 – Outlier Analysis")
print("=" * 60)

num_cols = df.select_dtypes("number").columns.tolist()
oa = OutlierAnalyzer(df)
oa.detect_iqr(columns=num_cols)
oa.report()
oa.plot(columns=num_cols)

# Many outliers are domain-valid (Ferrari HP, EV MPG, luxury MSRP) →
# winsorize at 5 % rather than removing rows.
oa.winsorize(columns=["Engine HP","Engine Cylinders",
                       "highway MPG","city mpg","MSRP","Popularity"])
df = oa.get_df()
print(f"Shape after outlier treatment: {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  |  Univariate Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 – Univariate Analysis")
print("=" * 60)

uni = UnivariateAnalyzer(df)

# Numeric
uni.numeric_summary()
uni.plot_numeric()
uni.normality_test()

# Categorical
cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
uni.categorical_summary(columns=cat_cols, top_n=10)
uni.plot_categorical(columns=cat_cols, top_n=10)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  |  Bivariate / Multivariate Analysis & Statistical Testing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 – Bivariate / Multivariate Analysis")
print("=" * 60)

biv = BivariateAnalyzer(df)

# 6a. Pearson correlation heatmap
biv.correlation_heatmap(method="pearson")

# 6b. Key scatter plots
biv.scatter_plot("Engine HP", "MSRP", hue="Vehicle Size")
biv.scatter_plot("city mpg", "highway MPG")

# 6c. ANOVA  –  MSRP vs categorical predictors
anova_groups = ["Transmission Type", "Driven_Wheels",
                "Vehicle Size", "Vehicle Style"]
biv.anova(target="MSRP", group_cols=anova_groups)

# 6d. Tukey HSD post-hoc on Vehicle Size (has 3 distinct groups)
biv.tukey_hsd(target="MSRP", group_col="Vehicle Size")
biv.tukey_hsd(target="MSRP", group_col="Transmission Type")

# 6e. Chi-Square independence among categorical features
chi_cols = ["Transmission Type","Driven_Wheels","Vehicle Size","Vehicle Style"]
biv.chi_square_matrix(columns=chi_cols)

# 6f. Grouped box plots
biv.grouped_boxplot("MSRP", "Vehicle Size")
biv.grouped_boxplot("MSRP", "Transmission Type")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  |  Save cleaned data
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv("/home/claude/eda_api/cleaned_car_features.csv", index=False)
print("\n✔ Cleaned dataset saved to cleaned_car_features.csv")
print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
