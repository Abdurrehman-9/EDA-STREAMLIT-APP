"""
=============================================================================
DataWrangler API  –  Data Cleaning & Exploratory Data Analysis Toolkit
=============================================================================
Authors : Shayan (27027) & AbdurRehman (27041)
Version : 2.0.0
Fix     : fill_statistic() now handles categorical columns without crashing
=============================================================================
"""

import warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set_theme(style="whitegrid", palette="deep")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 ▸ DataLoader
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    LOADERS = {
        ".csv":     pd.read_csv,
        ".tsv":     lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".json":    pd.read_json,
        ".parquet": pd.read_parquet,
    }

    def __init__(self, filepath: str, **read_kwargs):
        self.filepath    = filepath
        self.read_kwargs = read_kwargs
        self.df          = None

    def load(self):
        import pathlib
        ext    = pathlib.Path(self.filepath).suffix.lower()
        loader = self.LOADERS.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file type '{ext}'.")
        self.df = loader(self.filepath, **self.read_kwargs)
        print(f"Loaded {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self.df

    def overview(self):
        if self.df is None:
            raise RuntimeError("Call .load() first.")
        df = self.df
        summary = pd.DataFrame({
            "dtype":     df.dtypes,
            "non-null":  df.notna().sum(),
            "missing":   df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "unique":    df.nunique(),
        })
        print(summary.to_string())

    def column_types(self):
        if self.df is None:
            raise RuntimeError("Call .load() first.")
        num = self.df.select_dtypes(include="number").columns.tolist()
        cat = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        return {"numeric": num, "categorical": cat}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 ▸ DataCleaner
# ══════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df   = df.copy()
        self._log = []

    def remove_duplicates(self, subset=None, keep="first"):
        before = len(self.df)
        self.df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed = before - len(self.df)
        self._log.append(f"[Duplicates] Removed {removed:,} rows.")
        print(f"  ✔ {self._log[-1]}")
        return self

    def normalise_strings(self, columns=None, case="lower"):
        cols = columns or self.df.select_dtypes("object").columns.tolist()
        for col in cols:
            orig_nulls = self.df[col].isna()
            s = self.df[col].astype(str).str.strip()
            if case == "lower":   s = s.str.lower()
            elif case == "upper": s = s.str.upper()
            elif case == "title": s = s.str.title()
            self.df[col] = s.where(~orig_nulls, other=np.nan)
        self._log.append(f"[Strings] Normalised {len(cols)} columns.")
        print(f"  ✔ {self._log[-1]}")
        return self

    def fix_dtypes(self, type_map: dict):
        for col, dtype in type_map.items():
            if col not in self.df.columns:
                continue
            try:
                self.df[col] = self.df[col].astype(dtype)
                self._log.append(f"[dtypes] '{col}' -> {dtype}")
            except Exception as exc:
                print(f"  Warning: Could not cast '{col}' to {dtype}: {exc}")
        print(f"  ✔ [dtypes] Applied {len(type_map)} type conversions.")
        return self

    def flag_invalid_range(self, column: str, lo=None, hi=None):
        mask = pd.Series(False, index=self.df.index)
        if lo is not None:
            mask |= self.df[column] < lo
        if hi is not None:
            mask |= self.df[column] > hi
        print(f"  Info '{column}': {mask.sum():,} values outside [{lo}, {hi}].")
        return mask

    def drop_columns(self, columns: list, threshold_pct=None):
        to_drop = []
        for col in columns:
            if col not in self.df.columns:
                continue
            if threshold_pct is not None:
                pct = self.df[col].isna().mean() * 100
                if pct >= threshold_pct:
                    to_drop.append(col)
                    self._log.append(f"[Drop] '{col}' ({pct:.1f}% missing)")
            else:
                to_drop.append(col)
                self._log.append(f"[Drop] '{col}' (explicit)")
        self.df.drop(columns=to_drop, inplace=True, errors="ignore")
        print(f"  ✔ Dropped {len(to_drop)} column(s): {to_drop}")
        return self

    def get_clean_df(self):
        return self.df.copy()

    def print_log(self):
        print("\n── DataCleaner Log ─────────────────────────────────────")
        for i, e in enumerate(self._log, 1):
            print(f"  {i:>2}. {e}")
        print("─" * 57)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 ▸ MissingValueHandler
# ══════════════════════════════════════════════════════════════════════════════

class MissingValueHandler:
    def __init__(self, df: pd.DataFrame):
        self.df   = df.copy()
        self._log = []

    def summary(self):
        miss = self.df.isna().sum()
        miss = miss[miss > 0]
        out  = pd.DataFrame({
            "missing":   miss,
            "missing_%": (miss / len(self.df) * 100).round(2),
        }).sort_values("missing_%", ascending=False)
        print("\n── Missing Value Summary ────────────────────────────────")
        print(out.to_string() if len(out) else "  No missing values found.")
        print("─" * 57)
        return out

    def fill_constant(self, column: str, value):
        n = int(self.df[column].isna().sum())
        self.df[column].fillna(value, inplace=True)
        self._log.append(f"[Constant] '{column}': filled {n} NaNs with {repr(value)}")
        print(f"  ✔ {self._log[-1]}")
        return self

    def fill_statistic(self, column: str, strategy: str = "median"):
        """
        Fill NaNs with mean, median, or mode.
        Handles both numeric and categorical columns safely.
        The log message always converts the fill value to str before
        formatting — this fixes the 'g format code on str' crash.
        """
        n = int(self.df[column].isna().sum())
        if n == 0:
            return self

        if strategy == "mean":
            val     = self.df[column].mean()
            val_str = f"{val:.4f}"
        elif strategy == "median":
            val     = self.df[column].median()
            val_str = f"{val:.4f}"
        elif strategy == "mode":
            mode_series = self.df[column].mode()
            if len(mode_series) == 0:
                print(f"  Warning: No mode found for '{column}', skipping.")
                return self
            val     = mode_series.iloc[0]
            val_str = str(val)          # ← key fix: str() works for any dtype
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                             f"Choose 'mean', 'median', or 'mode'.")

        self.df[column].fillna(val, inplace=True)
        self._log.append(
            f"[{strategy.title()}] '{column}': filled {n} NaNs with {val_str}"
        )
        print(f"  ✔ {self._log[-1]}")
        return self

    def fill_by_group(self, column: str, group_cols: list, strategy: str = "median"):
        n_before = int(self.df[column].isna().sum())
        if strategy in ("mean", "median"):
            self.df[column] = (
                self.df.groupby(group_cols)[column]
                       .transform(lambda x: x.fillna(getattr(x, strategy)()))
            )
        elif strategy == "mode":
            self.df[column] = (
                self.df.groupby(group_cols)[column]
                       .transform(lambda x: x.fillna(
                           x.mode().iloc[0] if len(x.mode()) else np.nan))
            )
        n_after = int(self.df[column].isna().sum())
        filled  = n_before - n_after
        self._log.append(
            f"[GroupFill] '{column}' by {group_cols} ({strategy}): "
            f"filled {filled}, {n_after} still missing"
        )
        print(f"  ✔ {self._log[-1]}")
        return self

    def fill_ffill_bfill(self, column: str, method: str = "ffill"):
        n = int(self.df[column].isna().sum())
        self.df[column] = (
            self.df[column].ffill() if method == "ffill"
            else self.df[column].bfill()
        )
        self._log.append(f"[{method}] '{column}': filled {n} NaNs")
        print(f"  ✔ {self._log[-1]}")
        return self

    def fill_ml(self, target_col: str, num_features: list,
                cat_features: list, n_estimators: int = 200,
                random_state: int = 42):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline

        train = self.df[self.df[target_col].notna()]
        pred  = self.df[self.df[target_col].isna()]

        if len(pred) == 0:
            print(f"  Info: '{target_col}' has no missing values, skipping.")
            return self

        all_feats  = num_features + cat_features
        is_numeric = self.df[target_col].dtype.kind in ("f", "i")

        preprocessor = ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ])
        Estimator = RandomForestRegressor if is_numeric else RandomForestClassifier
        pipeline  = Pipeline([
            ("prep",  preprocessor),
            ("model", Estimator(n_estimators=n_estimators,
                                random_state=random_state)),
        ])
        pipeline.fit(train[all_feats], train[target_col])
        predicted = pipeline.predict(pred[all_feats])
        if is_numeric:
            predicted = np.round(predicted, 2)
        self.df.loc[self.df[target_col].isna(), target_col] = predicted
        self._log.append(
            f"[ML-RandomForest] '{target_col}': imputed {len(pred)} values"
        )
        print(f"  ✔ {self._log[-1]}")
        return self

    def get_df(self):
        return self.df.copy()

    def print_log(self):
        print("\n── MissingValueHandler Log ──────────────────────────────")
        for i, e in enumerate(self._log, 1):
            print(f"  {i:>2}. {e}")
        print("─" * 57)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 ▸ OutlierAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class OutlierAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df     = df.copy()
        self._flags = {}

    def _to_plain_series(self, col):
        """Convert winsorized masked arrays to plain float64 Series."""
        return pd.Series(np.asarray(self.df[col]), dtype="float64", name=col)

    def detect_iqr(self, columns=None, k: float = 1.5):
        cols = columns or self.df.select_dtypes("number").columns.tolist()
        for col in cols:
            s      = self._to_plain_series(col).dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - k * iqr, q3 + k * iqr
            idx    = s[(s < lo) | (s > hi)].index
            self._flags[col] = idx
            print(f"  [{col}]  IQR [{lo:.2f}, {hi:.2f}]  -> {len(idx):,} outliers")
        return self

    def detect_zscore(self, columns=None, threshold: float = 3.0):
        cols = columns or self.df.select_dtypes("number").columns.tolist()
        for col in cols:
            s   = self._to_plain_series(col).dropna()
            z   = np.abs(stats.zscore(s))
            idx = s.index[z > threshold]
            self._flags[col] = idx
            print(f"  [{col}]  z-score > {threshold}  -> {len(idx):,} outliers")
        return self

    def report(self):
        rows = []
        for col, idx in self._flags.items():
            rows.append({
                "column":       col,
                "n_outliers":   len(idx),
                "pct_outliers": round(len(idx) / len(self.df) * 100, 2),
            })
        out = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
        print("\n── Outlier Report ───────────────────────────────────────")
        print(out.to_string(index=False))
        print("─" * 57)
        return out

    def winsorize(self, columns=None, limits=(0.05, 0.05)):
        cols = columns or list(self._flags.keys()) or \
               self.df.select_dtypes("number").columns.tolist()
        for col in cols:
            self.df[col] = np.array(winsorize(self.df[col], limits=list(limits)))
            print(f"  ✔ Winsorised '{col}' limits={limits}")
        return self

    def remove_outliers(self, columns=None):
        cols    = columns or list(self._flags.keys())
        all_idx = pd.Index([])
        for col in cols:
            all_idx = all_idx.union(self._flags.get(col, []))
        before = len(self.df)
        self.df.drop(index=all_idx, inplace=True, errors="ignore")
        print(f"  ✔ Removed {before - len(self.df):,} outlier rows; {len(self.df):,} remain.")
        return self

    def plot(self, columns=None):
        cols = columns or self.df.select_dtypes("number").columns.tolist()
        for col in cols:
            data = self._to_plain_series(col).dropna()
            fig, axes = plt.subplots(1, 2, figsize=(12, 3))
            axes[0].hist(data, bins=30, color=sns.color_palette("deep")[0],
                         alpha=0.7, edgecolor="white")
            axes[0].set_title(f"{col} – Distribution")
            axes[1].boxplot(data, vert=False, patch_artist=True,
                            boxprops=dict(facecolor=sns.color_palette("deep")[1],
                                         alpha=0.7))
            axes[1].set_title(f"{col} – Boxplot")
            fig.tight_layout()
            plt.show()

    def get_df(self):
        return self.df.copy()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 ▸ UnivariateAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class UnivariateAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _plain(self, col):
        return pd.Series(np.asarray(self.df[col]), dtype="float64", name=col)

    def numeric_summary(self, columns=None):
        cols     = columns or self.df.select_dtypes("number").columns.tolist()
        plain_df = pd.DataFrame({c: self._plain(c) for c in cols})
        desc     = plain_df.describe().T
        desc["skewness"]  = plain_df.skew()
        desc["kurtosis"]  = plain_df.kurtosis()
        desc["missing_%"] = (self.df[cols].isna().mean() * 100).round(2)
        print("\n── Numeric Summary ──────────────────────────────────────")
        print(desc.to_string())
        print("─" * 57)
        return desc

    def plot_numeric(self, columns=None, n_cols: int = 2):
        cols   = columns or self.df.select_dtypes("number").columns.tolist()
        n_rows = math.ceil(len(cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
        axes    = np.array(axes).flatten()
        palette = sns.color_palette("deep", len(cols))
        for i, col in enumerate(cols):
            data = self._plain(col).dropna()
            axes[i].hist(data, bins=30, color=palette[i], alpha=0.7,
                         density=True, edgecolor="white")
            try:
                data.plot.kde(ax=axes[i], color="black", lw=1.5)
            except Exception:
                pass
            axes[i].set_title(col)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Numeric Feature Distributions", fontsize=14, y=1.01)
        fig.tight_layout()
        plt.show()

    def categorical_summary(self, columns=None, top_n: int = 10):
        cols   = columns or self.df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        frames = []
        for col in cols:
            vc  = self.df[col].value_counts(normalize=False).head(top_n)
            pct = self.df[col].value_counts(normalize=True).head(top_n) * 100
            frames.append(pd.DataFrame({
                "column": col, "value": vc.index,
                "count":  vc.values,
                "pct":    pct.values.round(2),
            }))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        print("\n── Categorical Summary ──────────────────────────────────")
        print(out.to_string(index=False))
        print("─" * 57)
        return out

    def plot_categorical(self, columns=None, top_n: int = 10, n_cols: int = 3):
        cols   = columns or self.df.select_dtypes(
            include=["object", "category"]).columns.tolist()
        n_rows = math.ceil(len(cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes    = np.array(axes).flatten()
        palette = sns.color_palette("deep", top_n)
        for i, col in enumerate(cols):
            vc = self.df[col].value_counts().head(top_n)
            axes[i].bar(range(len(vc)), vc.values,
                        color=[palette[j % len(palette)] for j in range(len(vc))],
                        edgecolor="white")
            axes[i].set_xticks(range(len(vc)))
            axes[i].set_xticklabels(vc.index.astype(str),
                                    rotation=40, ha="right", fontsize=8)
            axes[i].set_title(col, fontsize=10)
            axes[i].set_ylabel("Count")
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Categorical Feature Distributions", fontsize=14, y=1.01)
        fig.tight_layout()
        plt.show()

    def normality_test(self, columns=None):
        cols = columns or self.df.select_dtypes("number").columns.tolist()
        rows = []
        for col in cols:
            data = self._plain(col).dropna()
            if len(data) > 5000:
                data = data.sample(5000, random_state=42)
            try:
                w, p = stats.shapiro(data)
                rows.append({
                    "column":         col,
                    "W-stat":         round(w, 4),
                    "p-value":        round(p, 6),
                    "normal (a=0.05)": p > 0.05,
                })
            except Exception:
                pass
        out = pd.DataFrame(rows)
        print("\n── Normality Test (Shapiro-Wilk) ────────────────────────")
        print(out.to_string(index=False))
        print("─" * 57)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 ▸ BivariateAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class BivariateAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def correlation_heatmap(self, columns=None, method: str = "pearson",
                            figsize=(12, 8)):
        cols     = columns or self.df.select_dtypes("number").columns.tolist()
        plain_df = pd.DataFrame({
            c: pd.Series(np.asarray(self.df[c]), dtype="float64") for c in cols
        })
        corr = plain_df.corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title(f"{method.title()} Correlation Heatmap", fontsize=13)
        plt.tight_layout()
        plt.show()
        return corr

    def scatter_plot(self, x: str, y: str, hue=None, regression: bool = True):
        fig, ax = plt.subplots(figsize=(8, 5))
        if hue:
            for label, grp in self.df.groupby(hue):
                ax.scatter(grp[x], grp[y], alpha=0.5, s=20, label=label)
            ax.legend(title=hue, fontsize=7)
        else:
            ax.scatter(self.df[x], self.df[y], alpha=0.4, s=20,
                       color=sns.color_palette("deep")[0])
        if regression:
            mask = self.df[[x, y]].notna().all(axis=1)
            m, b, r, p, _ = stats.linregress(
                self.df.loc[mask, x], self.df.loc[mask, y])
            xs = np.linspace(self.df[x].min(), self.df[x].max(), 200)
            ax.plot(xs, m * xs + b, "r--", lw=1.5,
                    label=f"OLS r={r:.2f} p={p:.3g}")
            ax.legend()
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{x}  vs  {y}")
        plt.tight_layout()
        plt.show()

    def anova(self, target: str, group_cols: list):
        results = {}
        print("\n── ANOVA Results ────────────────────────────────────────")
        for col in group_cols:
            try:
                formula = f'Q("{target}") ~ C(Q("{col}"))'
                model   = ols(formula, data=self.df).fit()
                table   = sm.stats.anova_lm(model, typ=2)
                results[col] = table
                p   = table["PR(>F)"].iloc[0]
                sig = "significant" if p < 0.05 else "not significant"
                print(f"\n  {target} ~ {col}  [p={p:.4g}, {sig}]")
                print(table.to_string())
            except Exception as exc:
                print(f"  Warning: Skipping '{col}': {exc}")
        print("─" * 57)
        return results

    def tukey_hsd(self, target: str, group_col: str, alpha: float = 0.05):
        result = pairwise_tukeyhsd(
            endog=self.df[target].dropna(),
            groups=self.df.loc[self.df[target].notna(), group_col],
            alpha=alpha,
        )
        print(f"\n── Tukey HSD: {target} ~ {group_col} ───────────────────")
        print(result)
        print("─" * 57)
        return result

    def chi_square_matrix(self, columns: list):
        mat = pd.DataFrame(index=columns, columns=columns, dtype=float)
        for c1 in columns:
            for c2 in columns:
                if c1 == c2:
                    mat.loc[c1, c2] = 0.0
                else:
                    try:
                        tbl = pd.crosstab(self.df[c1], self.df[c2])
                        _, p, _, _ = chi2_contingency(tbl)
                        mat.loc[c1, c2] = round(p, 6)
                    except Exception:
                        mat.loc[c1, c2] = np.nan
        fig, ax = plt.subplots(figsize=(max(6, len(columns)),
                                         max(5, len(columns))))
        sns.heatmap(mat.astype(float), annot=True, fmt=".3f",
                    cmap="coolwarm_r", ax=ax, linewidths=0.5,
                    annot_kws={"size": 8})
        ax.set_title("Chi-Square P-Value Heatmap (≈0 → dependent)", fontsize=11)
        plt.tight_layout()
        plt.show()
        return mat

    def grouped_boxplot(self, target: str, group_col: str, top_n: int = 15):
        top    = self.df[group_col].value_counts().head(top_n).index
        subset = self.df[self.df[group_col].isin(top)]
        order  = (subset.groupby(group_col)[target]
                        .median().sort_values(ascending=False).index)
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=subset, x=group_col, y=target, order=order,
                    palette="deep", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        ax.set_title(f"{target} by {group_col} (top {top_n} groups)")
        plt.tight_layout()
        plt.show()
