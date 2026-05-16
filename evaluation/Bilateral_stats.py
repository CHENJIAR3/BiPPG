import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import glob
# ── 配置 ─────────────────────────────────────────────────────────────────────
PKL_DIR   = "/home/cjr/PPGBen/predictions/"
SAVE_PATH = "../results/bilateral_stats.xlsx"
ALPHA     = 0.05

METRICS      = ["SBP", "DBP", "HR", "Age"]
UNITS        = ["mmHg", "mmHg", "bpm", "years"]
TRUE_COLS    = ["sbp_fix",        "dbp_fix",        "pr_ref",      "age"]
IPSI_COLS    = ["pred_sbp",       "pred_dbp",        "pred_hr",     "pred_age"]
CONTRA_COLS  = ["pred_sbp_right", "pred_dbp_right",  "pred_hr_right","pred_age_right"]

ORDER = [
    "CRNN","ACNN","ResNet1D","Net1D","LSTM","Efficient1D",
    "AutoFormer","PatchTST","Informer","iTransformer",
    "ResNet1DMoE","CSFM","PaAno",
]

COMPARE = "ipsi"   # "ipsi" | "contra" | "both"


# ── 核心统计函数 ──────────────────────────────────────────────────────────────
def wilcoxon_test(a, b):
    if np.all(a == b):
        return np.nan, np.nan
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return stat, p

from scipy.stats import t as t_dist

def paired_cohens_d(a, b):
    """
    Paired Cohen's d for |error_a| vs |error_b|.
    d > 0 means bilateral (b) has smaller errors than baseline (a).
    """
    diff = (a - b)  # positive = bilateral is better
    n = len(diff)
    mean_d = np.mean(diff)
    std_d  = np.std(diff, ddof=1)
    return mean_d / std_d if std_d > 0 else np.nan

def ci_mae_difference(errors_baseline, errors_bilateral, alpha=0.05):
    """
    95% CI on ΔMAE = (MAE_base - MAE_bi)/MAE_base  (positive = bilateral is better).
    Uses paired t-distribution via bootstrap or normal approx.
    """
    diff = (errors_baseline - errors_bilateral) # ΔMAE per sample
    n = len(diff)
    mean_d = np.mean(diff)
    se     = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    return mean_d, mean_d - t_crit * se, mean_d + t_crit * se

def rank_biserial_r(errors_baseline, errors_bilateral):
    """r > 0  ⟺  baseline 误差更大  ⟺  bilateral 更优"""
    diff  = errors_baseline - errors_bilateral
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    denom = n_pos + n_neg
    return (n_pos - n_neg) / denom if denom else np.nan


def better_ratio(errors_baseline, errors_bilateral):
    """bilateral 误差严格小于 baseline 的样本比例 (%)"""
    return 100.0 * np.mean(errors_bilateral < errors_baseline)


# ── 单模型 → 逐指标行 ────────────────────────────────────────────────────────
def analyze_one_model(df: pd.DataFrame, model_name: str):
    y_true   = df[TRUE_COLS].values
    y_ipsi   = df[IPSI_COLS].values
    y_contra = df[CONTRA_COLS].values
    y_bi     = 0.5 * (y_ipsi + y_contra)

    e_ipsi   = np.abs(y_ipsi   - y_true)
    e_contra = np.abs(y_contra - y_true)
    e_bi     = np.abs(y_bi     - y_true)

    rows = []
    for k, (metric, unit) in enumerate(zip(METRICS, UNITS)):
        bi  = e_bi[:, k]
        ips = e_ipsi[:, k]
        con = e_contra[:, k]

        for baseline_label, baseline_err in [("ipsi", ips), ("contra", con)]:
            _, p_raw = wilcoxon_test(bi, baseline_err)
            r        = rank_biserial_r(baseline_err, bi)
            br       = better_ratio(baseline_err, bi)

            mae_bi   = bi.mean()
            mae_base = baseline_err.mean()

            delta_pct = 100.0 * (mae_base - mae_bi) / mae_base   # positive = improvement

            delta_mae, ci_lo, ci_hi = ci_mae_difference(baseline_err, bi)
            d = paired_cohens_d(baseline_err, bi)

            rows.append({
                "Model":    model_name,
                "Metric":   metric,
                # "Unit":     unit,
                "Baseline": baseline_label,
                # "N":        len(bi),
                "MAE_bi":   mae_bi,
                "MAE_base": mae_base,
                # "delta_pct":delta_pct,
                "delta_mae": delta_mae,  # 绝对改善量 (mmHg/bpm/years)
                "delta_pct": 100*delta_mae/mae_base,
                "ci_lo": 100*ci_lo/mae_base,  # 95% CI 下界
                "ci_hi": 100*ci_hi/mae_base,  # 95% CI 上界
                "cohens_d": d,  # 效应量
                "p_raw":    p_raw,
                "r":        r,
                "better_ratio": br,
            })
    return rows


# ── 多重比较校正 ──────────────────────────────────────────────────────────────
def apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    mask  = df["p_raw"].notna()
    pvals = df.loc[mask, "p_raw"].values
    _, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    df["p_adj"] = np.nan
    df.loc[mask, "p_adj"] = p_adj
    return df


# ── 格式化为 LaTeX 表格 ───────────────────────────────────────────────────────
def fmt_p(p: float) -> str:
    """格式化 p 值为论文风格"""
    if np.isnan(p):
        return "--"
    if p < 1e-4:
        return r"$<10^{-4}$"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def build_latex_table(summary: pd.DataFrame, baseline_label: str, caption_extra: str = "") -> str:
    """
    summary: 已聚合到 Metric 级别的 DataFrame，含
        delta_pct, p_adj, r, better_ratio
    """
    caption = (
        r"Statistical comparison between bilateral and unilateral ("
        + baseline_label
        + r") configurations. "
        r"$\Delta$(\%) indicates relative MAE reduction compared with the baseline."
        + caption_extra
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Metric & $\Delta$MAE (\%) & $p$-value & Effect size & Better ratio (\%) \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Metric']} & "
            f"{row['delta_pct']:+.2f} & "
            f"{fmt_p(row['p_adj'])} & "
            f"{row['r']:.2f} & "
            f"{row['better_ratio']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # pkl_files = sorted(
    #     [f for f in os.listdir(PKL_DIR) if f.endswith(".pkl")],
    #     key=lambda x: ORDER.index(x.split("_")[0]) if x.split("_")[0] in ORDER else 999,
    # )
    modelnames = ["CRNN", "ACNN", "ResNet1D", "Net1D", "LSTM", "Efficient1D", "AutoFormer",
                  "PatchTST", "Informer", "iTransformer", "ResNet1DMoE", "CSFM", ]

    all_rows = []
    # for pkl_file in pkl_files:
    #     model_name = pkl_file.split("_")[0]
    for model_name in modelnames:
        try:
            pkl_files = glob.glob(os.path.join(PKL_DIR, model_name + "*_prediction.pkl"))
            df   = pd.read_pickle(pkl_files[0])
            rows = analyze_one_model(df, model_name)
            all_rows.extend(rows)
            print(f"[OK] {model_name}  N={rows[0]['N']}")
        except Exception as e:
            pass

            # print(f"[SKIP] {pkl_file}: {e}")

    if not all_rows:
        raise RuntimeError("未读取到任何预测文件，请检查 PKL_DIR 和列名。")

    full_df = pd.DataFrame(all_rows)
    full_df  = apply_fdr(full_df)   # 全局 BH-FDR 校正

    # ── 生成表格（对每种对比方向各出一张）────────────────────────────────────
    for bl in ["ipsi", "contra"]:
        sub = full_df[full_df["Baseline"] == bl].copy()
        # 跨模型聚合：对同一 Metric 取均值
        # p 值用 Fisher's method 或直接取中位数均可；
        # 这里取中位数（保守），effect size / better_ratio 取均值。

        summary = (
            sub.groupby("Metric", sort=False)
            .agg(
                delta_pct   = ("delta_pct",    "mean"),
                p_adj       = ("p_adj",         "median"),   # 跨模型中位 p
                r           = ("r",             "mean"),
                better_ratio= ("better_ratio",  "mean"),
            )
            .loc[METRICS]   # 保持 SBP / DBP / HR / Age 顺序
            .reset_index()
        )

        latex = build_latex_table(summary, bl)
        print(f"\n{'='*60}")
        print(f"LaTeX table — bilateral vs {bl}:")
        print(f"{'='*60}")
        print(latex)

    # ── 保存完整结果到 Excel ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with pd.ExcelWriter(SAVE_PATH, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="Per-Model Details")
        for bl in ["ipsi", "contra"]:
            sub = full_df[full_df["Baseline"] == bl]
            sub.groupby("Metric", sort=False).agg(
                delta_pct    = ("delta_pct",   "mean"),
                p_adj        = ("p_adj",        "median"),
                r            = ("r",            "mean"),
                better_ratio = ("better_ratio", "mean"),
            ).loc[METRICS].reset_index().to_excel(
                writer, index=False, sheet_name=f"Summary_vs_{bl}"
            )
    print(f"\n完整结果已保存至 {SAVE_PATH}")
