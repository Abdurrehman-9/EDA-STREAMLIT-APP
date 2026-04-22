"""
=============================================================================
Streamlit EDA App  –  DataWrangler
=============================================================================
Authors : AbdurRehman 

CLEANING LOGIC:
  • Car Features dataset  → exact domain-aware imputation from the notebook
  • Any other dataset     → smart generic cleaning (auto median/mode/ML)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math, sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data_wrangler_api import (
    DataCleaner, MissingValueHandler, OutlierAnalyzer,
    UnivariateAnalyzer, BivariateAnalyzer,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataWrangler EDA",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 DataWrangler EDA App")
st.caption("Built on Car Features dataset · Works with any CSV too")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("---")
    win_limit = st.slider("Winsorise limit %", 1, 15, 5)
    st.markdown("---")
    st.markdown("**Modules to run**")
    run_clean   = st.checkbox("Data Cleaning",       value=True)
    run_missing = st.checkbox("Missing Values",      value=True)
    run_outlier = st.checkbox("Outlier Analysis",    value=True)
    run_univar  = st.checkbox("Univariate Analysis", value=True)
    run_bivar   = st.checkbox("Bivariate Analysis",  value=True)

if uploaded is None:
    st.info("👈  Upload a CSV file from the sidebar to begin.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def is_car_features(df: pd.DataFrame) -> bool:
    """Return True if the uploaded file looks like the Car Features dataset."""
    required = {
        "Make", "Model", "Year", "Engine Fuel Type",
        "Engine HP", "Engine Cylinders", "Transmission Type",
        "Driven_Wheels", "Number of Doors", "Market Category",
        "Vehicle Size", "Vehicle Style", "highway MPG",
        "city mpg", "Popularity", "MSRP",
    }
    return required.issubset(set(df.columns))


def make_fig(nrows, ncols, w_per=5, h_per=3.5):
    return plt.subplots(nrows, ncols, figsize=(w_per * ncols, h_per * nrows))


# ═════════════════════════════════════════════════════════════════════════════
# LOAD
# ═════════════════════════════════════════════════════════════════════════════

df_raw = pd.read_csv(uploaded)
car_dataset = is_car_features(df_raw)

st.success(
    f"Loaded **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns**"
    + ("  ·  Car Features dataset detected — using domain-aware cleaning" if car_dataset else "")
)

with st.expander("📋 Raw data preview"):
    st.dataframe(df_raw.head(30), use_container_width=True)

df = df_raw.copy()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 ▸ DATA CLEANING
# ═════════════════════════════════════════════════════════════════════════════

if run_clean:
    st.header("Data Cleaning")

    cleaner = DataCleaner(df)

    # ── 1a. Duplicates (same for every dataset) ───────────────────────────
    cleaner.remove_duplicates()

    # ── 1b. Year: treat as label, not number (Car Features) ──────────────
    if car_dataset and "Year" in df.columns:
        cleaner.fix_dtypes({"Year": "object"})

    # ── 1c. Drop Market Category if present and >30% missing ─────────────
    if "Market Category" in cleaner.df.columns:
        pct = cleaner.df["Market Category"].isna().mean() * 100
        if pct >= 30:
            cleaner.drop_columns(["Market Category"])

    # ── 1d. Drop any other column with ≥40% missing ───────────────────────
    high_miss = [
        c for c in cleaner.df.columns
        if cleaner.df[c].isna().mean() * 100 >= 40
        and c != "Market Category"   # already handled above
    ]
    if high_miss:
        cleaner.drop_columns(high_miss)

    df = cleaner.get_clean_df()

    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    dups_removed = df_raw.duplicated().sum()
    col1.metric("Duplicate rows removed", dups_removed)
    col2.metric("Rows remaining",         f"{df.shape[0]:,}")
    col3.metric("Columns remaining",      df.shape[1])

    with st.expander("Cleaning log"):
        for entry in cleaner._log:
            st.write(f"• {entry}")

    # Missing value bar chart before cleaning
    miss_before = df_raw.isna().sum()
    miss_before = miss_before[miss_before > 0]
    if len(miss_before):
        fig, ax = plt.subplots(figsize=(10, 3))
        miss_before.sort_values(ascending=False).plot(
            kind="bar", ax=ax, color="#7F77DD", edgecolor="white")
        ax.set_title("Missing values per column (before cleaning)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 ▸ MISSING VALUES
# ═════════════════════════════════════════════════════════════════════════════

if run_missing:
    st.header("Missing Value Handling")

    mvh = MissingValueHandler(df)

    miss_now = df.isna().sum()
    miss_now = miss_now[miss_now > 0]

    if len(miss_now) == 0:
        st.success("No missing values after cleaning!")
    else:
        miss_df = pd.DataFrame({
            "missing":   miss_now,
            "missing_%": (miss_now / len(df) * 100).round(2),
        }).sort_values("missing_%", ascending=False)
        st.dataframe(miss_df, use_container_width=True)

        # ── CAR FEATURES: exact domain-aware imputation ─────────────────
        if car_dataset:
            st.info("Using domain-aware imputation from your notebook.")

            # Engine Fuel Type → MCAR → fill with 'regular unleaded'
            if "Engine Fuel Type" in df.columns and df["Engine Fuel Type"].isna().any():
                mvh.fill_constant("Engine Fuel Type", "regular unleaded")

            # Number of Doors → domain knowledge
            if "Number of Doors" in df.columns and df["Number of Doors"].isna().any():
                # Tesla Model S → 4 doors
                mask_ts = (
                    (mvh.df["Make"] == "Tesla") &
                    (mvh.df["Model"] == "Model S") &
                    mvh.df["Number of Doors"].isna()
                )
                mvh.df.loc[mask_ts, "Number of Doors"] = 4

                # Ferrari FF → 2 doors
                mask_ff = (
                    (mvh.df["Make"] == "Ferrari") &
                    (mvh.df["Model"] == "FF") &
                    mvh.df["Number of Doors"].isna()
                )
                mvh.df.loc[mask_ff, "Number of Doors"] = 2

                # Remaining → mode within Vehicle Style
                mvh.fill_by_group("Number of Doors", ["Vehicle Style"], strategy="mode")

            # Engine Cylinders → electric = 0, then by fuel type
            if "Engine Cylinders" in df.columns and df["Engine Cylinders"].isna().any():
                mvh.df.loc[
                    (mvh.df["Engine Fuel Type"] == "electric") &
                    mvh.df["Engine Cylinders"].isna(),
                    "Engine Cylinders"
                ] = 0

                mvh.df.loc[
                    (mvh.df["Engine Fuel Type"] == "premium unleaded (required)") &
                    mvh.df["Engine Cylinders"].isna(),
                    "Engine Cylinders"
                ] = 4

                mvh.df.loc[
                    (mvh.df["Engine Fuel Type"] == "regular unleaded") &
                    mvh.df["Engine Cylinders"].isna(),
                    "Engine Cylinders"
                ] = 6

                # Any still missing → group median
                if mvh.df["Engine Cylinders"].isna().any():
                    mvh.fill_by_group(
                        "Engine Cylinders",
                        ["Engine Fuel Type", "Vehicle Style"],
                        strategy="median",
                    )

            # Safety: ensure Engine Cylinders has no NaN before ML uses it as a feature
            if "Engine Cylinders" in mvh.df.columns and mvh.df["Engine Cylinders"].isna().any():
                mvh.fill_statistic("Engine Cylinders", strategy="median")

            # Engine HP → Random Forest (MNAR but recoverable)
            if "Engine HP" in df.columns and df["Engine HP"].isna().any():
                with st.spinner("Training Random Forest to impute Engine HP…"):
                    mvh.fill_ml(
                        target_col   = "Engine HP",
                        num_features = ["MSRP", "Engine Cylinders"],
                        cat_features = [
                            "Make", "Model", "Vehicle Size",
                            "Vehicle Style", "Transmission Type",
                        ],
                    )

        # ── GENERIC: smart automatic imputation ──────────────────────────
        else:
            st.info("Generic imputation: numeric → median, categorical → mode.")
            num_cols = df.select_dtypes("number").columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            for col in num_cols:
                if mvh.df[col].isna().any():
                    mvh.fill_statistic(col, strategy="median")

            for col in cat_cols:
                if mvh.df[col].isna().any():
                    mvh.fill_statistic(col, strategy="mode")

        df = mvh.get_df()

        # Show remaining missing
        still_missing = df.isna().sum().sum()
        if still_missing == 0:
            st.success("All missing values handled!")
        else:
            st.warning(f"{still_missing} values still missing after imputation.")

        with st.expander("Imputation log"):
            for entry in mvh._log:
                st.write(f"• {entry}")


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 ▸ OUTLIER ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

if run_outlier:
    st.header("Outlier Analysis")

    # For Car Features use the same columns as the notebook
    if car_dataset:
        outlier_cols = [
            c for c in
            ["Engine HP", "Engine Cylinders", "highway MPG",
             "city mpg", "Popularity", "MSRP"]
            if c in df.columns
        ]
    else:
        outlier_cols = df.select_dtypes("number").columns.tolist()

    oa = OutlierAnalyzer(df)
    oa.detect_iqr(columns=outlier_cols)
    report_df = oa.report()
    st.dataframe(report_df, use_container_width=True)

    st.info(
        "Most outliers in this dataset are domain-valid (luxury cars, EVs). "
        "Winsorisation is applied instead of deletion to preserve all rows."
    )

    lim = win_limit / 100
    oa.winsorize(columns=outlier_cols, limits=(lim, lim))
    df = oa.get_df()
    st.success(f"Winsorised {len(outlier_cols)} columns at {win_limit}% limits. "
               f"Shape unchanged: {df.shape[0]:,} × {df.shape[1]}")

    # Boxplot grid
    show_cols = outlier_cols[:6]
    if show_cols:
        n = len(show_cols)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(show_cols):
            data = pd.Series(np.asarray(df[col]), dtype="float64").dropna()
            axes[i].boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor="#7F77DD", alpha=0.6))
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Boxplots after Winsorisation", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 ▸ UNIVARIATE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

if run_univar:
    st.header("📊 Univariate Analysis")

    uni      = UnivariateAnalyzer(df)
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab_num, tab_cat, tab_norm = st.tabs(
        ["Numeric distributions", "Categorical distributions", "Normality test"]
    )

    # ── Numeric ──────────────────────────────────────────────────────────
    with tab_num:
        summary = uni.numeric_summary()
        st.dataframe(summary.round(4), use_container_width=True)

        cols_show = num_cols[:8]
        n    = len(cols_show)
        nc   = min(3, n)
        nr   = math.ceil(n / nc)
        fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.5 * nr))
        axes = np.array(axes).flatten()
        palette = sns.color_palette("deep", n)
        for i, col in enumerate(cols_show):
            data = pd.Series(np.asarray(df[col]), dtype="float64").dropna()
            axes[i].hist(data, bins=25, color=palette[i],
                         alpha=0.75, density=True, edgecolor="white")
            try:
                data.plot.kde(ax=axes[i], color="black", lw=1.5)
            except Exception:
                pass
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(labelsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Numeric Feature Distributions (Histogram + KDE)", fontsize=11)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Categorical ───────────────────────────────────────────────────────
    with tab_cat:
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            cat_summary = uni.categorical_summary(top_n=10)
            st.dataframe(cat_summary, use_container_width=True)

            cols_show = cat_cols[:9]
            n  = len(cols_show)
            nc = min(3, n)
            nr = math.ceil(n / nc)
            fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3.5 * nr))
            axes = np.array(axes).flatten()
            palette = sns.color_palette("deep", 10)
            for i, col in enumerate(cols_show):
                vc = df[col].value_counts().head(10)
                axes[i].bar(range(len(vc)), vc.values,
                            color=[palette[j % 10] for j in range(len(vc))],
                            edgecolor="white")
                axes[i].set_xticks(range(len(vc)))
                axes[i].set_xticklabels(vc.index.astype(str),
                                        rotation=40, ha="right", fontsize=7)
                axes[i].set_title(col, fontsize=9)
                axes[i].set_ylabel("Count", fontsize=7)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Categorical Feature Distributions (top-10 values)",
                         fontsize=11)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Normality test ────────────────────────────────────────────────────
    with tab_norm:
        norm_df = uni.normality_test()
        st.dataframe(norm_df, use_container_width=True)
        st.caption(
            "Shapiro-Wilk test — p > 0.05 means the column is likely normally "
            "distributed. For large datasets a sample of 5,000 rows is used."
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 ▸ BIVARIATE / MULTIVARIATE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

if run_bivar:
    st.header("📊 Bivariate & Statistical Analysis")

    biv      = BivariateAnalyzer(df)
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab_corr, tab_scatter, tab_anova, tab_tukey, tab_chi = st.tabs([
        "Correlation heatmap",
        "Scatter plot",
        "ANOVA",
        "Tukey HSD",
        "Chi-Square",
    ])

    # ── Correlation heatmap ───────────────────────────────────────────────
    with tab_corr:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            plain_df = pd.DataFrame({
                c: pd.Series(np.asarray(df[c]), dtype="float64") for c in num_cols
            })
            corr = plain_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(11, 7))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap="coolwarm", center=0, ax=ax,
                        linewidths=0.5, annot_kws={"size": 8})
            ax.set_title("Pearson Correlation Heatmap", fontsize=12)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Key findings:**")
            if car_dataset:
                st.write(
                    "- Engine HP ↔ Cylinders: strong positive (r ≈ 0.77)\n"
                    "- HP / Cylinders ↔ MPG: strong negative (r ≈ −0.60)\n"
                    "- City MPG ↔ Highway MPG: very strong positive (r ≈ 0.89)\n"
                    "- MSRP ↔ Engine HP: moderate positive (r ≈ 0.54)"
                )

    # ── Scatter plot ──────────────────────────────────────────────────────
    with tab_scatter:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            c1, c2, c3 = st.columns(3)
            x_col   = c1.selectbox("X axis", num_cols,
                                   index=num_cols.index("Engine HP")
                                   if "Engine HP" in num_cols else 0)
            y_col   = c2.selectbox("Y axis", num_cols,
                                   index=num_cols.index("MSRP")
                                   if "MSRP" in num_cols else min(1, len(num_cols)-1))
            hue_opt = ["None"] + cat_cols
            hue_col = c3.selectbox("Colour by", hue_opt)

            fig, ax = plt.subplots(figsize=(8, 5))
            if hue_col != "None":
                top_cats = df[hue_col].value_counts().head(8).index
                palette  = dict(zip(top_cats, sns.color_palette("deep", 8)))
                for label in top_cats:
                    grp = df[df[hue_col] == label]
                    ax.scatter(grp[x_col], grp[y_col],
                               alpha=0.4, s=15, label=label,
                               color=palette[label])
                ax.legend(title=hue_col, fontsize=7, markerscale=2)
            else:
                ax.scatter(df[x_col], df[y_col],
                           alpha=0.4, s=15, color="#534AB7")

            # OLS line
            try:
                mask = df[[x_col, y_col]].notna().all(axis=1)
                m, b, r, p, _ = stats.linregress(
                    df.loc[mask, x_col], df.loc[mask, y_col])
                xs = np.linspace(df[x_col].min(), df[x_col].max(), 200)
                ax.plot(xs, m * xs + b, "r--", lw=1.5,
                        label=f"OLS  r={r:.2f}  p={p:.3g}")
                ax.legend(fontsize=8)
            except Exception:
                pass

            ax.set_xlabel(x_col); ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── ANOVA ─────────────────────────────────────────────────────────────
    with tab_anova:
        if not cat_cols or not num_cols:
            st.info("Need numeric and categorical columns.")
        else:
            default_target = "MSRP" if "MSRP" in num_cols else num_cols[0]
            default_group  = (
                "Vehicle Size" if "Vehicle Size" in cat_cols
                else cat_cols[0]
            )
            target   = st.selectbox("Numeric target (dependent)", num_cols,
                                    index=num_cols.index(default_target))
            groupvar = st.selectbox("Categorical group (independent)", cat_cols,
                                    index=cat_cols.index(default_group))
            if st.button("Run ANOVA"):
                with st.spinner("Running ANOVA…"):
                    results = biv.anova(target=target, group_cols=[groupvar])
                if groupvar in results:
                    st.dataframe(results[groupvar], use_container_width=True)
                    p = results[groupvar]["PR(>F)"].iloc[0]
                    if p < 0.05:
                        st.success(
                            f"p = {p:.4f} → Groups differ significantly. "
                            f"Run Tukey HSD to see which pairs differ."
                        )
                    else:
                        st.info(f"p = {p:.4f} → No significant difference detected.")

    # ── Tukey HSD ─────────────────────────────────────────────────────────
    with tab_tukey:
        if not cat_cols or not num_cols:
            st.info("Need numeric and categorical columns.")
        else:
            default_target = "MSRP" if "MSRP" in num_cols else num_cols[0]
            tukey_groups = [
                c for c in cat_cols
                if df[c].nunique() <= 20       # avoid huge category columns
            ]
            default_grp = (
                "Vehicle Size" if "Vehicle Size" in tukey_groups
                else (tukey_groups[0] if tukey_groups else cat_cols[0])
            )
            t_target = st.selectbox("Target variable", num_cols,
                                    index=num_cols.index(default_target),
                                    key="tukey_target")
            t_group  = st.selectbox("Group variable (≤20 categories)", tukey_groups,
                                    index=tukey_groups.index(default_grp)
                                    if default_grp in tukey_groups else 0,
                                    key="tukey_group")
            if st.button("Run Tukey HSD"):
                with st.spinner("Running Tukey HSD…"):
                    try:
                        result = biv.tukey_hsd(t_target, t_group)
                        tukey_df = pd.DataFrame(
                            data    = result._results_table.data[1:],
                            columns = result._results_table.data[0],
                        )
                        st.dataframe(tukey_df, use_container_width=True)
                        st.caption(
                            "reject=True means those two groups differ significantly (α=0.05)."
                        )
                    except Exception as e:
                        st.error(f"Tukey HSD failed: {e}")

    # ── Chi-Square ────────────────────────────────────────────────────────
    with tab_chi:
        if len(cat_cols) < 2:
            st.info("Need at least 2 categorical columns.")
        else:
            # Pre-select sensible defaults for Car Features
            if car_dataset:
                defaults = [
                    c for c in
                    ["Transmission Type", "Driven_Wheels",
                     "Vehicle Size", "Vehicle Style"]
                    if c in cat_cols
                ]
            else:
                defaults = cat_cols[:4]

            chi_selected = st.multiselect(
                "Select categorical columns to test",
                options=cat_cols,
                default=defaults,
            )
            if len(chi_selected) < 2:
                st.info("Select at least 2 columns.")
            elif st.button("Run Chi-Square"):
                with st.spinner("Computing Chi-Square matrix…"):
                    chi_mat = biv.chi_square_matrix(chi_selected)
                    st.dataframe(chi_mat.round(4), use_container_width=True)
                    st.caption(
                        "Values ≈ 0 → the two categorical features are "
                        "statistically dependent (not independent)."
                    )
                    # Plot
                    fig, ax = plt.subplots(
                        figsize=(max(6, len(chi_selected)),
                                 max(5, len(chi_selected)))
                    )
                    sns.heatmap(chi_mat.astype(float), annot=True, fmt=".3f",
                                cmap="coolwarm_r", ax=ax,
                                linewidths=0.5, annot_kws={"size": 9})
                    ax.set_title("Chi-Square P-Value Heatmap (≈0 → dependent)",
                                 fontsize=11)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.caption("DataWrangler EDA API · by · AbdurRehman")
