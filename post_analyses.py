import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================
use_calibration = True  # True/False
DATA_PATH = "data/disease_label_processed.csv"

# NOTE: keep the same filenames as your pipeline
F_TRAIN_IDS = "data/F_na_data_train.csv"
F_TEST_IDS  = "data/F_na_data_test.csv"
M_TRAIN_IDS = "data/M_na_data_train.csv"
M_TEST_IDS  = "data/M_na_data_test.csv"

OUT_DIR = "results"
ANALYSIS_DIR = os.path.join(OUT_DIR, "analysis")


# =========================
# Calibration 
# =========================
def perform_calibration(train_df: pd.DataFrame, all_df: pd.DataFrame):
    """
    Notebook-aligned idea:
    Fit CA -> (BA-CA) on TRAIN, then for any subject:
        C_BA = BA - predicted_gap(CA)
        C_Age_Gap = C_BA - CA
  """
    # Fit CA -> (BA-CA)
    train_ca = train_df["CA"].values.reshape(-1, 1)
    train_gap = (train_df["BA"].values - train_df["CA"].values)  # BA-CA
    calibrator = LinearRegression().fit(train_ca, train_gap)

    all_ca = all_df["CA"].values.reshape(-1, 1)
    pred_gap = calibrator.predict(all_ca)

    all_df = all_df.copy()
    all_df["C_BA"] = all_df["BA"].values - pred_gap
    all_df["C_Age_Gap"] = all_df["C_BA"] - all_df["CA"]

    return all_df, calibrator


# =========================
# Notebook functions 
# =========================
def bootstrap_confidence_interval(y_true, y_pred, func, alpha=0.95):
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    loss = []
    for i in range(n):
        l = func([y_true[i]], [y_pred[i]])
        loss.append(l)
    a, b = st.norm.interval(alpha=alpha, loc=np.mean(loss), scale=st.sem(loss))
    if pd.isna(a) or pd.isna(n):
        return a, b
    else:
        return round(a, 3), round(b, 3)


def metric_generate(label_f: pd.DataFrame, hc_test_f):
    normal_f = label_f[label_f["Participant ID"].isin(hc_test_f)]
    disease_b = label_f[label_f["BC_label"].isin(["Pre-existing Disease", "Post-DXA and Pre-Disease"])]
    disease_a = label_f[label_f["BC_label"].isin(["Post-DXA Disease", "Post-DXA and Pre-Disease"])]

    vat_l = label_f[label_f["VAT_label"] == "Hypernormal"]
    vat_u = label_f[label_f["VAT_label"] == "Suboptimal"]

    # age strata
    strata = ["Overall", "40-49", "50-59", "60-69", "70-79"]

    def _slice(df, age):
        if age == "Overall":
            return df
        return df[df["Age_Label"] == age]

    rows = []
    for age in strata:
        t = _slice(normal_f, age)
        db = _slice(disease_b, age)
        da = _slice(disease_a, age)
        l = _slice(vat_l, age)
        u = _slice(vat_u, age)

        # helper safe metric
        def safe_metrics(d):
            if len(d) == 0:
                return (np.nan, (np.nan, np.nan), np.nan, np.nan)
            mae = mean_absolute_error(d["CA"], d["BA"])
            mae_ci = bootstrap_confidence_interval(d["CA"], d["BA"], mean_absolute_error)
            mse = mean_squared_error(d["CA"], d["BA"])
            r2 = r2_score(d["CA"], d["BA"])
            return (mae, mae_ci, mse, r2)

        t_mae, t_ci, t_mse, t_r2 = safe_metrics(t)
        db_mae, db_ci, db_mse, db_r2 = safe_metrics(db)
        da_mae, da_ci, da_mse, da_r2 = safe_metrics(da)
        l_mae, l_ci, l_mse, l_r2 = safe_metrics(l)
        u_mae, u_ci, u_mse, u_r2 = safe_metrics(u)

        rows.append({
            "Age Group": age,

            "Normal Reference Test MAE": f"{t_mae:.3f} {t_ci}" if pd.notna(t_mae) else np.nan,
            "Normal Reference Test MSE": round(t_mse, 3) if pd.notna(t_mse) else np.nan,
            "Normal Reference Test R2": round(t_r2, 3) if pd.notna(t_r2) else np.nan,

            "Pre-existing MAE": f"{db_mae:.3f} {db_ci}" if pd.notna(db_mae) else np.nan,
            "Pre-existing MSE": round(db_mse, 3) if pd.notna(db_mse) else np.nan,
            "Pre-existing R2": round(db_r2, 3) if pd.notna(db_r2) else np.nan,

            "Post-DXA MAE": f"{da_mae:.3f} {da_ci}" if pd.notna(da_mae) else np.nan,
            "Post-DXA MSE": round(da_mse, 3) if pd.notna(da_mse) else np.nan,
            "Post-DXA R2": round(da_r2, 3) if pd.notna(da_r2) else np.nan,

            "Hypernormal MAE": f"{l_mae:.3f} {l_ci}" if pd.notna(l_mae) else np.nan,
            "Hypernormal MSE": round(l_mse, 3) if pd.notna(l_mse) else np.nan,
            "Hypernormal R2": round(l_r2, 3) if pd.notna(l_r2) else np.nan,

            "Suboptimal MAE": f"{u_mae:.3f} {u_ci}" if pd.notna(u_mae) else np.nan,
            "Suboptimal MSE": round(u_mse, 3) if pd.notna(u_mse) else np.nan,
            "Suboptimal R2": round(u_r2, 3) if pd.notna(u_r2) else np.nan,
        })

    return pd.DataFrame(rows)


def map_label(df: pd.DataFrame, target: str):
    # notebook: Before -> Before, else -> HC (includes After)
    new_label = []
    label = df[target + " " + "Label"].to_list()
    for l in label:
        if l == "Before":
            new_label.append("Before")
        else:
            new_label.append("HC")
    return new_label


def odd_ratio(t2df: pd.DataFrame):
    # notebook: no constant, x = [age_diff, CA]
    x = t2df[["age_diff", "CA"]]
    y = t2df["new_label"]
    est = sm.Logit(y, x).fit(disp=0)
    odds_ratio = est.params.apply(lambda z: round(np.exp(z), 2))
    conf = est.conf_int()
    print(est.summary())
    print("Odds ratio:", odds_ratio.to_dict())
    print("CI:", np.exp(conf))



def hz_analysis(t2df, target, thred, spath):
    # notebook layout
    x = t2df[["BA", "CA", "age_diff", target + " " + "Duration", "new_label"]].copy()
    x["durations"] = x[target + " " + "Duration"] / 365  # years
    T = x["durations"]
    E = x["new_label"]
    groups = t2df["age_diff"]

    km_2 = KaplanMeierFitter()
    i1 = (groups <= thred)
    i2 = (groups > thred)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)

    km_2.fit(T[i1], E[i1], label="BA smaller than CA Goup")
    a1 = km_2.plot(ax=ax1)
    km_2.fit(T[i2], E[i2], label="BA larger than CA Goup")
    km_2.plot(ax=a1)

    stage_results = logrank_test(T[i1], T[i2], event_observed_A=E[i1], event_observed_B=E[i2])
    print("log_rank-test:")
    stage_results.print_summary()

    if spath and len(spath) > 0:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        fig.savefig(os.path.join(ANALYSIS_DIR, spath), bbox_inches="tight", dpi=350)
    plt.close(fig)

    cph2 = CoxPHFitter()
    cph2.fit(x[["age_diff", "new_label", "durations"]], "durations", event_col="new_label")
    cph2.print_summary()
