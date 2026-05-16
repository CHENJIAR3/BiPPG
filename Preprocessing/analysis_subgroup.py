import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp

# =========================
# 1. 分组定义
# =========================
AGE_BINS = [0, 18, 45, 60, 75, 120]
AGE_LABELS = ["0–18", "18–44", "45–59", "60–74", "≥75"]

BMI_BINS = [0, 18.5, 23.9, 27.9, 100]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]

POSTURE_LABELS = ["lay", "sit", "stand"]
GENDER_MAP = {1: "Male", 2: "Female"}


# =========================
# 2. 工具函数
# =========================
def compute_bmi(weight, height_cm):
    if height_cm <= 0 or weight <= 0:
        return np.nan
    return weight / (height_cm / 100) ** 2


def assign_bin(value, bins, labels):
    if np.isnan(value):
        return None
    idx = np.digitize(value, bins) - 1
    if idx < 0 or idx >= len(labels):
        return None
    return labels[idx]


# =========================
# 3. 读取所有数据
# =========================
def load_all_records(data_dir):
    records = []

    for path in glob.glob(os.path.join(data_dir, "*.pkl")):
        df = pd.read_pickle(path)
        if df.empty:
            continue

        meta = df.iloc[0]
        age = meta["age"]
        gender = GENDER_MAP.get(meta["gender"], None)
        bmi = compute_bmi(meta["weight"], meta["height"])

        age_grp = assign_bin(age, AGE_BINS, AGE_LABELS)
        bmi_grp = assign_bin(bmi, BMI_BINS, BMI_LABELS)

        for _, row in df.iterrows():
            if pd.isna(row["sbp_fix"]) or pd.isna(row["dbp_fix"]):
                continue

            records.append({
                "SBP": row["sbp_fix"],
                "DBP": row["dbp_fix"],
                "HR": row["pr_ref"],
                "AgeGroup": age_grp,
                "Gender": gender,
                "BMIGroup": bmi_grp,
                "Posture": row["position"]
            })

    return pd.DataFrame(records)


# =========================
# 4. 均值 ± std 表
# =========================
def summary_table(df, group_col, value_col):
    res = (
        df
        .groupby(group_col)[value_col]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    res["mean±std"] = res["mean"].round(2).astype(str) + " ± " + res["std"].round(2).astype(str)
    return res[[group_col, "count", "mean±std"]]


# =========================
# 5. 统计检验（整体 + 两两）
# =========================
def statistical_tests(df, group_col, value_col):
    # 【新增：关键修复】剔除分组列或数值列中包含 None/NaN 的行
    # 这能防止 posthoc_dunn 内部由于 KeyError: None 崩溃
    temp_df = df.dropna(subset=[group_col, value_col]).copy()

    # 确保分组标签中没有 None 对象（有时候 dropna 后仍残留 None 类型标签）
    temp_df = temp_df[temp_df[group_col].notnull()]

    # 重新获取分组数据用于 Kruskal-Wallis 检验
    groups = [
        g[value_col].values
        for _, g in temp_df.groupby(group_col)
        if len(g) > 0
    ]

    # 如果分组少于 2 个，无法统计
    if len(groups) < 2:
        return None, None

    # 1. 整体检验
    _, p_kw = kruskal(*groups)

    # 2. 两两比较 (传入过滤后的 temp_df)
    try:
        dunn = sp.posthoc_dunn(
            temp_df,
            val_col=value_col,
            group_col=group_col,
            p_adjust="bonferroni"
        )
    except Exception as e:
        print(f"⚠️ Dunn test failed for {group_col}-{value_col}: {e}")
        return round(p_kw, 4), None

    return round(p_kw, 4), dunn.round(4)


# =========================
# 6. 主流程
# =========================
def run_analysis(data_dir, save_path):
    df = load_all_records(data_dir)

    writer = pd.ExcelWriter(save_path, engine="openpyxl")

    subgroup_defs = {
        "AGE": "AgeGroup",
        "Gender": "Gender",
        "BMI": "BMIGroup",
        "Posture": "Posture"
    }

    for name, col in subgroup_defs.items():
        for metric in ["SBP", "DBP", "HR"]:
            summary = summary_table(df, col, metric)
            p_kw, dunn = statistical_tests(df, col, metric)

            summary["Metric"] = metric
            summary["Kruskal_p"] = p_kw

            sheet_summary = f"{name}_summary"
            sheet_stats = f"{name}_stats"

            summary.to_excel(writer, sheet_name=sheet_summary, index=False, startrow=writer.sheets[sheet_summary].max_row + 1 if sheet_summary in writer.sheets else 0)

            if dunn is not None:
                dunn.to_excel(writer, sheet_name=sheet_stats)

    writer.close()
    print(f"✅ Subgroup statistics saved to {save_path}")


# =========================
# 7. 入口
# =========================
if __name__ == "__main__":
    # data_dir = "/home/cjr/datasets/Ring2Health/10_second/"
    data_dir = "D:\研究课题\datasets/Ring2Health/10_second/"

    save_path = "../results/Subgroup_BP_Statistics.xlsx"
    run_analysis(data_dir, save_path)
