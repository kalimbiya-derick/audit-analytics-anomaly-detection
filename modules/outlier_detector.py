"""
outlier_detector.py
----------------------
Identifies statistically unusual transaction amounts using robust
outlier-detection methods, applied PER ACCOUNT TYPE.

WHY PER ACCOUNT, NOT GLOBALLY:
A 5,000,000 transaction is unremarkable among Loan Disbursements but would
be alarming among Savings Deposits or Processing Fee Income. Pooling all
accounts together and computing one global outlier threshold would either
miss real anomalies (drowned out by naturally large disbursements) or
flag entirely normal small-account transactions. Every method below
groups by account_col before computing thresholds.

WHY NOT JUST CLASSIC Z-SCORE:
Z-score = (x - mean) / std_dev, then flag |z| > threshold (commonly 3).
This is the method most tutorials teach, but it has a serious weakness for
audit work: BOTH the mean and standard deviation are themselves distorted
by the outliers you're trying to detect (a "masking effect"). One genuinely
huge fraudulent transaction inflates the mean and especially the std dev,
which can make the very transaction you're hunting for look statistically
unremarkable — exactly when it matters most.

We therefore default to two more robust alternatives, both standard in
forensic/statistical literature:

- IQR (Tukey's method): flags values outside
  [Q1 - k*IQR, Q3 + k*IQR], where IQR = Q3 - Q1. Quartiles are far less
  sensitive to extreme values than the mean.

- Modified Z-score (Iglewicz & Hoaglin, 1993): replaces mean/std with
  MEDIAN and MAD (median absolute deviation) — both robust statistics
  that resist distortion from the outliers themselves.
        modified_z = 0.6745 * (x - median) / MAD
  Recommended flagging threshold: |modified_z| > 3.5

Classic z-score is still included for completeness and comparison, but
modified_z_score is the DEFAULT method for this module's main entry point.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.chart_style import apply_style, add_footer, COLORS


def _iqr_bounds(series: pd.Series, multiplier: float = 1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def detect_outliers_iqr(df: pd.DataFrame, amount_col: str = "amount",
                          account_col: str = "account", multiplier: float = 1.5) -> pd.DataFrame:
    """Flags transactions outside Tukey's IQR fences, computed per account type."""
    flagged_parts = []
    for account, group in df.groupby(account_col):
        if len(group) < 4:  # IQR is unstable on tiny groups
            continue
        lower, upper = _iqr_bounds(group[amount_col], multiplier)
        outliers = group[(group[amount_col] < lower) | (group[amount_col] > upper)].copy()
        if not outliers.empty:
            outliers["account_q1"] = round(group[amount_col].quantile(0.25), 2)
            outliers["account_q3"] = round(group[amount_col].quantile(0.75), 2)
            outliers["iqr_lower_bound"] = round(lower, 2)
            outliers["iqr_upper_bound"] = round(upper, 2)
            outliers["direction"] = np.where(outliers[amount_col] > upper, "high", "low")
            outliers["flag_reason"] = (
                "Statistical outlier (IQR method) within '" + str(account) +
                "' — " + outliers["direction"] + " extreme relative to peer transactions"
            )
            flagged_parts.append(outliers)
    if not flagged_parts:
        return df.iloc[0:0].copy()
    return pd.concat(flagged_parts).sort_values(amount_col, ascending=False).reset_index(drop=True)


def detect_outliers_modified_zscore(df: pd.DataFrame, amount_col: str = "amount",
                                      account_col: str = "account", threshold: float = 3.5) -> pd.DataFrame:
    """
    Flags transactions using the modified z-score (median/MAD based),
    computed per account type. This is the recommended default method —
    robust to the skewed distributions typical of financial data and not
    self-distorted by the outliers it's trying to detect.
    """
    flagged_parts = []
    for account, group in df.groupby(account_col):
        if len(group) < 4:
            continue
        median = group[amount_col].median()
        mad = (group[amount_col] - median).abs().median()
        if mad == 0:
            # All values identical or MAD degenerate — fall back to a tiny
            # epsilon to avoid division by zero rather than skipping the group
            mad = 1e-9
        modified_z = 0.6745 * (group[amount_col] - median) / mad
        outliers = group[modified_z.abs() > threshold].copy()
        if not outliers.empty:
            outliers["modified_z_score"] = round(modified_z[modified_z.abs() > threshold], 2)
            outliers["account_median"] = round(median, 2)
            outliers["direction"] = np.where(outliers[amount_col] > median, "high", "low")
            outliers["flag_reason"] = (
                "Statistical outlier (modified z-score) within '" + str(account) +
                "' — " + outliers["direction"] + " extreme relative to peer transactions"
            )
            flagged_parts.append(outliers)
    if not flagged_parts:
        return df.iloc[0:0].copy()
    return pd.concat(flagged_parts).sort_values(
        "modified_z_score", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def detect_outliers_zscore(df: pd.DataFrame, amount_col: str = "amount",
                             account_col: str = "account", threshold: float = 3.0) -> pd.DataFrame:
    """
    Classic z-score method (mean/std based). Included for completeness and
    comparison against the more robust methods above — NOT recommended as
    the primary method for skewed financial data (see module docstring).
    """
    flagged_parts = []
    for account, group in df.groupby(account_col):
        if len(group) < 4:
            continue
        mean = group[amount_col].mean()
        std = group[amount_col].std()
        if std == 0:
            continue
        z = (group[amount_col] - mean) / std
        outliers = group[z.abs() > threshold].copy()
        if not outliers.empty:
            outliers["z_score"] = round(z[z.abs() > threshold], 2)
            outliers["account_mean"] = round(mean, 2)
            outliers["direction"] = np.where(outliers[amount_col] > mean, "high", "low")
            outliers["flag_reason"] = (
                "Statistical outlier (classic z-score) within '" + str(account) +
                "' — " + outliers["direction"] + " extreme relative to peer transactions"
            )
            flagged_parts.append(outliers)
    if not flagged_parts:
        return df.iloc[0:0].copy()
    return pd.concat(flagged_parts).sort_values(
        "z_score", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def detect_outliers(df: pd.DataFrame, amount_col: str = "amount", account_col: str = "account",
                     method: str = "modified_zscore", **kwargs) -> pd.DataFrame:
    """
    Main entry point. method options: 'modified_zscore' (default, recommended),
    'iqr', or 'zscore'.
    """
    if method == "modified_zscore":
        return detect_outliers_modified_zscore(df, amount_col, account_col,
                                                 threshold=kwargs.get("threshold", 3.5))
    elif method == "iqr":
        return detect_outliers_iqr(df, amount_col, account_col,
                                    multiplier=kwargs.get("multiplier", 1.5))
    elif method == "zscore":
        return detect_outliers_zscore(df, amount_col, account_col,
                                       threshold=kwargs.get("threshold", 3.0))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'modified_zscore', 'iqr', or 'zscore'.")


def plot_outliers_by_account(df: pd.DataFrame, flagged: pd.DataFrame, output_path: str,
                               amount_col: str = "amount", account_col: str = "account",
                               title: str = "Transaction Amount Distribution by Account (Outliers Highlighted)"):
    """
    Boxplot of transaction amounts per account type, with flagged outliers
    overlaid as distinct points — visually communicates both the spread of
    normal activity and exactly which transactions broke from it.
    """
    accounts = sorted(df[account_col].unique())
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    data_by_account = [df[df[account_col] == acc][amount_col].values for acc in accounts]
    bp = ax.boxplot(data_by_account, tick_labels=accounts, orientation="vertical", patch_artist=True,
                     showfliers=False)  # we'll plot flagged points ourselves for clarity
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["low"])
        patch.set_alpha(0.25)

    if not flagged.empty:
        for i, acc in enumerate(accounts, start=1):
            acc_flagged = flagged[flagged[account_col] == acc]
            if not acc_flagged.empty:
                jitter = np.random.uniform(-0.08, 0.08, size=len(acc_flagged))
                ax.scatter([i + j for j in jitter], acc_flagged[amount_col],
                           color=COLORS["high"], zorder=5, s=35, label="Flagged outlier" if i == 1 else "")

    ax.set_ylabel("Transaction Amount (TZS)")
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1], loc="upper right")

    plt.tight_layout()
    add_footer(fig)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def print_outlier_summary(flagged: pd.DataFrame, amount_col: str = "amount", method_label: str = ""):
    print("=" * 55)
    print(f"OUTLIER DETECTION{' — ' + method_label if method_label else ''}")
    print("=" * 55)
    if flagged.empty:
        print("No outliers flagged.")
        print("=" * 55)
        return
    print(f"Total flagged transactions: {len(flagged)}")
    if "direction" in flagged.columns:
        print(f"  High-side outliers: {(flagged['direction'] == 'high').sum()}")
        print(f"  Low-side outliers: {(flagged['direction'] == 'low').sum()}")
    print(f"Total value of flagged transactions: {flagged[amount_col].sum():,.2f}")
    print("=" * 55)


if __name__ == "__main__":
    pass
