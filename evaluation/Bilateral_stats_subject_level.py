import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, t as t_dist
from statsmodels.stats.multitest import multipletests

# ── 配置 ─────────────────────────────────────────────────────────────────────
PKL_DIR   = "/home/cjr/PPGBen/predictions/"
SAVE_PATH = "../results/bilateral_stats_subject_level.xlsx"
ALPHA     = 0.05

METRICS     = ["SBP", "DBP", "HR", "Age"]
UNITS       = ["mmHg", "mmHg", "bpm", "years"]
TRUE_COLS   = ["sbp_fix",        "dbp_fix",        "pr_ref",       "age"]
IPSI_COLS   = ["pred_sbp",       "pred_dbp",        "pred_hr",      "pred_age"]
CONTRA_COLS = ["pred_sbp_right", "pred_dbp_right",  "pred_hr_right","pred_age_right"]
BI_COLS     = ["pred_sbp_bi",    "pred_dbp_bi",     "pred_hr_bi",   "pred_age_bi"]



# ── Step 1：segment → subject 聚合（正确顺序）────────────────────────────────
def aggregate_to_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    先在 segment level 算绝对误差，再按 subject_id 聚合取均值 MAE。
    返回 subject-level DataFrame，每列是该受试者的平均绝对误差，每行一个受试者。
    列名：subject_id + {e_ipsi_sbp, e_contra_sbp, e_bi_sbp, ...} × 4 指标
    """
    df = df.copy()

    # 1a. segment level：计算 bilateral 预测 & 各方向绝对误差
    err_cols = {}
    for ipsi_col, contra_col, bi_col, true_col, metric in zip(
            IPSI_COLS, CONTRA_COLS, BI_COLS, TRUE_COLS, METRICS):
        df[bi_col] = 0.5 * (df[ipsi_col] + df[contra_col])
        m = metric.lower()
        df[f"e_ipsi_{m}"]   = (df[ipsi_col]   - df[true_col]).abs()
        df[f"e_contra_{m}"] = (df[contra_col] - df[true_col]).abs()
        df[f"e_bi_{m}"]     = (df[bi_col]     - df[true_col]).abs()
        err_cols[metric] = (f"e_ipsi_{m}", f"e_contra_{m}", f"e_bi_{m}")

    # 1b. 按 subject_id 聚合：对绝对误差列取均值 → subject-level MAE
    all_err_cols = [c for trio in err_cols.values() for c in trio]
    subject_df = df.groupby("subject_id")[all_err_cols].mean().reset_index()
    subject_df.attrs["err_cols"] = err_cols   # 传递列名映射供下游使用
    return subject_df


# ── Step 2：统计检验函数 ──────────────────────────────────────────────────────
def wilcoxon_test(a: np.ndarray, b: np.ndarray):
    if np.all(a == b):
        return np.nan, np.nan
    _, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return _, p


def paired_cohens_d(errors_base: np.ndarray, errors_bi: np.ndarray) -> float:
    """
    Paired Cohen's d.  d > 0 → bilateral 误差更小（有改善）。
    """
    diff  = errors_base - errors_bi
    std_d = np.std(diff, ddof=1)
    return np.mean(diff) / std_d if std_d > 0 else np.nan


def ci_delta_mae(errors_base: np.ndarray, errors_bi: np.ndarray,
                 alpha: float = 0.05):
    """
    95% CI on ΔMAE (absolute) = mean(|e_base| - |e_bi|).
    正值 → bilateral 更优。返回 (point_est, lower, upper)。
    """
    diff   = errors_base - errors_bi
    n      = len(diff)
    mean_d = np.mean(diff)
    se     = np.std(diff, ddof=1) / np.sqrt(n)
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    return mean_d, mean_d - t_crit * se, mean_d + t_crit * se


def rank_biserial_r(errors_base: np.ndarray, errors_bi: np.ndarray) -> float:
    diff  = errors_base - errors_bi
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    denom = n_pos + n_neg
    return (n_pos - n_neg) / denom if denom else np.nan


def better_ratio(errors_base: np.ndarray, errors_bi: np.ndarray) -> float:
    """bilateral 误差严格小于 baseline 的受试者比例 (%)"""
    return 100.0 * np.mean(errors_bi < errors_base)


# ── Step 3：单模型分析（subject level） ───────────────────────────────────────
def analyze_one_model(df_raw: pd.DataFrame, model_name: str) :
    """
    输入原始 segment-level DataFrame。
    aggregate_to_subject() 已在 segment level 算好绝对误差，
    再 groupby subject 取均值，得到每人的 MAE。
    这里直接读取这些误差列做统计，不再重新计算误差。
    """
    subj     = aggregate_to_subject(df_raw)
    err_cols = subj.attrs["err_cols"]   # {metric: (e_ipsi_col, e_contra_col, e_bi_col)}
    N        = len(subj)

    rows = []
    for metric, unit in zip(METRICS, UNITS):
        e_ipsi_col, e_contra_col, e_bi_col = err_cols[metric]
        ips = subj[e_ipsi_col].values
        con = subj[e_contra_col].values
        bi  = subj[e_bi_col].values

        for baseline_label, baseline_err in [("ipsi", ips), ("contra", con)]:
            _, p_raw        = wilcoxon_test(bi, baseline_err)
            r               = rank_biserial_r(baseline_err, bi)
            br              = better_ratio(baseline_err, bi)
            d               = paired_cohens_d(baseline_err, bi)
            delta, lo, hi   = ci_delta_mae(baseline_err, bi)

            mae_bi   = bi.mean()
            mae_base = baseline_err.mean()

            rows.append({
                "Model":        model_name,
                "Metric":       metric,
                "Unit":         unit,
                "Baseline":     baseline_label,
                "N_subjects":   N,
                "MAE_bi":       mae_bi,
                "MAE_base":     mae_base,
                # 绝对改善量及 95% CI（单位与指标一致）
                "delta_mae":    delta,
                "ci_lo":        100.0 * lo / mae_base,
                "ci_hi":        100.0 * hi / mae_base,
                # 相对改善量（%）
                "delta_pct":    100.0 * delta / mae_base if mae_base > 0 else np.nan,
                # 效应量
                "cohens_d":     d,
                "r":            r,
                # 受试者级别的改善比例
                "better_ratio": br,
                "p_raw":        p_raw,
            })
    return rows


# ── Step 4：多重比较校正 ──────────────────────────────────────────────────────
def apply_fdr(df: pd.DataFrame) -> pd.DataFrame:
    mask  = df["p_raw"].notna()
    pvals = df.loc[mask, "p_raw"].values
    _, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="fdr_bh")
    df["p_adj"] = np.nan
    df.loc[mask, "p_adj"] = p_adj
    return df


# ── Step 5：LaTeX 表格 ────────────────────────────────────────────────────────
def fmt_p(p: float) -> str:
    if pd.isna(p):     return "--"
    if p < 1e-4:       return r"$<10^{-4}$"
    if p < 0.001:      return f"{p:.4f}"
    return f"{p:.3f}"


def fmt_ci(lo: float, hi: float, unit: str) -> str:
    return f"[{lo:+.2f}, {hi:+.2f}] {unit}"


def build_latex_table(summary: pd.DataFrame, baseline_label: str) -> str:
    """
    summary 列：Metric, Unit, delta_mae, ci_lo, ci_hi, cohens_d, p_adj,
                 delta_pct, r, better_ratio
    主列顺序（审稿人要求）：ΔMAE (95% CI) | Cohen's d | p-value
    辅助列保留在表尾备查：Δ% | better_ratio
    """
    caption = (
        r"Paired comparison: bilateral vs.\ unilateral ("
        + baseline_label
        + r") at the \emph{subject} level (Wilcoxon signed-rank, BH-FDR corrected). "
        r"$\Delta$MAE = MAE$_{\text{base}}$ $-$ MAE$_{\text{bi}}$ (positive = bilateral is better). "
        r"Cohen's $d$: $|d|<0.2$ negligible, $0.2$--$0.5$ small, $0.5$--$0.8$ medium, "
        r"$>0.8$ large \citep{cohen1988}. "
        r"Reference device precision: $\pm3$--$5$~mmHg (oscillometric cuff)."
    )
    col_spec = r"lrrrrrrr"
    header   = (
        r"Metric & MAE$_{\text{base}}$ & MAE$_{\text{bi}}$ & "
        r"$\Delta$MAE & 95\%~CI & Cohen's $d$ & $p_{\text{adj}}$ & "
        r"Better (\%) \\"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:bilateral_vs_" + baseline_label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        unit = row["Unit"]
        lines.append(
            f"{row['Metric']} ({unit}) & "
            f"{row['MAE_base']:.2f} & "
            f"{row['MAE_bi']:.2f} & "
            f"{row['delta_mae']:+.2f} & "
            f"[{row['ci_lo']:+.2f},\\;{row['ci_hi']:+.2f}] & "
            f"{row['cohens_d']:.2f} & "
            f"{fmt_p(row['p_adj'])} & "
            f"{row['better_ratio']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_rows = []
    MODELNAMES = ["CRNN", "ACNN", "ResNet1D", "Net1D", "LSTM", "Efficient1D", "AutoFormer",
                  "PatchTST", "Informer", "iTransformer", "ResNet1DMoE", "CSFM", ]

    for model_name in MODELNAMES:
        pkl_files = glob.glob(os.path.join(PKL_DIR, model_name + "*_prediction.pkl"))
        if not pkl_files:
            print(f"[SKIP] {model_name}: no file found")
            continue
        try:
            df_raw = pd.read_pickle(pkl_files[0])
            rows   = analyze_one_model(df_raw, model_name)
            all_rows.extend(rows)
            print(f"[OK] {model_name}  N_subjects={rows[0]['N_subjects']}")
        except Exception as e:
            print(f"[SKIP] {model_name}: {e}")

    if not all_rows:
        raise RuntimeError("未读取到任何预测文件，请检查 PKL_DIR 和列名。")

    full_df = pd.DataFrame(all_rows)
    full_df = apply_fdr(full_df)   # 全局 BH-FDR 校正（跨模型×指标×对比方向）

    # ── 跨模型聚合 + 输出 LaTeX ──────────────────────────────────────────────
    # 聚合策略：
    #   MAE_bi / MAE_base / delta_mae / ci_lo / ci_hi / cohens_d → 跨模型均值
    #   p_adj → 中位数（保守，避免 Fisher 合并的假设）
    #   better_ratio → 均值
    AGG = {
        "Unit":         ("Unit",         "first"),
        "MAE_base":     ("MAE_base",     "mean"),
        "MAE_bi":       ("MAE_bi",       "mean"),
        "delta_mae":    ("delta_mae",    "mean"),
        "ci_lo":        ("ci_lo",        "mean"),
        "ci_hi":        ("ci_hi",        "mean"),
        "cohens_d":     ("cohens_d",     "mean"),
        "p_adj":        ("p_adj",        "median"),
        "delta_pct":    ("delta_pct",    "mean"),
        "r":            ("r",            "mean"),
        "better_ratio": ("better_ratio", "mean"),
    }

    for bl in ["ipsi", "contra"]:
        sub = full_df[full_df["Baseline"] == bl].copy()

        summary = (
            sub.groupby("Metric", sort=False)
            .agg(**AGG)
            .loc[METRICS]
            .reset_index()
        )

        latex = build_latex_table(summary, bl)
        print(f"\n{'='*65}")
        print(f"LaTeX table — bilateral vs {bl}:")
        print(f"{'='*65}")
        print(latex)

    # ── 保存 Excel ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with pd.ExcelWriter(SAVE_PATH, engine="openpyxl") as writer:
        # Sheet 1：每模型每指标的完整结果
        full_df.to_excel(writer, index=False, sheet_name="Per-Model Details")

        # Sheet 2/3：跨模型汇总（分对比方向）
        for bl in ["ipsi", "contra"]:
            sub = full_df[full_df["Baseline"] == bl]
            (
                sub.groupby("Metric", sort=False)
                .agg(**AGG)
                .loc[METRICS]
                .reset_index()
                .to_excel(writer, index=False, sheet_name=f"Summary_vs_{bl}")
            )

    print(f"\n完整结果已保存至 {SAVE_PATH}")