import os

import pandas as pd
from evaluation.model_metric import get_performance

# ==============================
# 1 扁平化结果 + pr_ref -> HR
# ==============================

def flatten_result(result_dict):
    new_dict = {}
    # BP
    new_dict.update(result_dict["regression"]["SBP"])
    # HR (原 pr_ref)
    dbp_dict = result_dict["regression"]["DBP"]
    for k, v in dbp_dict.items():
        if k == "count":
            continue
        new_dict[k] = v
    hr_dict = result_dict["regression"]["HR"]
    for k, v in hr_dict.items():
        if k == "count":
            continue
        new_key = k.replace("pr_ref", "HR")
        new_dict[new_key] = v
    # AGE
    age_dict = result_dict["regression"]["Age"]
    for k, v in age_dict.items():
        if k == "count":
            continue
        new_dict[k] = v
    return new_dict

# ==============================
# 2 百分号转 float
# ==============================

def percent_to_float(v):

    if isinstance(v, str) and "%" in v:
        return float(v.replace("%", ""))

    return v


# ==============================
# 3 构建 dataframe
# ==============================

def build_dataframe(all_results):
    rows = []
    for model_name, model_results in all_results.items():
        for side, res in model_results.items():
            if res is None:
                continue
            flat = flatten_result(res)
            for k in flat:
                flat[k] = percent_to_float(flat[k])
            flat["model"] = model_name
            flat["type"] = side
            rows.append(flat)
    df = pd.DataFrame(rows)
    cols = ["model", "type"] + [c for c in df.columns if c not in ["model", "type"]]
    df = df[cols]
    return df


# ==============================
# 4 找最优指标
# ==============================

def find_best_values(df):
    metric_cols = [c for c in df.columns if c not in ["model", "type"]]
    best = {}
    for col in metric_cols:
        if "MAE" in col or "SDE" in col:
            best[col] = df[col].min()
        elif "ME" in col:
            best[col] = df[col].abs().min()
        elif "CPE" in col:
            best[col] = df[col].max()
    return best


# ==============================
# 5 输出 Excel
# ==============================

def export_excel(df, save_path):

    best = find_best_values(df)

    writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    # df.to_excel(writer, sheet_name="Original", index=False)

    df.to_excel(writer, sheet_name="Results", index=False)

    workbook = writer.book
    worksheet = writer.sheets["Results"]

    red_format = workbook.add_format({"font_color": "red"})

    for col_idx, col_name in enumerate(df.columns):

        if col_name not in best:
            continue

        best_value = best[col_name]

        for row in range(len(df)):

            value = df.iloc[row][col_name]

            if "ME" in col_name:

                if abs(value) == best_value:
                    worksheet.write(row + 1, col_idx, value, red_format)

            else:

                if value == best_value:
                    worksheet.write(row + 1, col_idx, value, red_format)

    writer.close()


# ==============================
# 6 主流程
# ==============================
# 自定义顺序
order = ["CRNN","ACNN","ResNet1D","Net1D","LSTM","Efficient1D","AutoFormer",
         "PatchTST","Informer","iTransformer","ResNet1DMoE","CSFM",]

# 排序


if __name__ == "__main__":
    pkl_dir = "/home/cjr/PPGBen/predictions/"
    save_path = "../results/all_models_results.xlsx"
    pkl_files = os.listdir(pkl_dir)
    pkl_files.sort(key=lambda x: order.index(x.split("_")[0]))

    all_results = {}
    for pkl_file in pkl_files:
        result_df = pd.read_pickle(os.path.join(pkl_dir, pkl_file))
        model_type = pkl_file.split("_")[0]
        all_results[model_type] = {}
        targets = result_df[["sbp_fix", "dbp_fix", "pr_ref", "age"]].values
        pred_left = result_df[["pred_sbp", "pred_dbp", "pred_hr", "pred_age"]].values
        pred_right = result_df[["pred_sbp_right", "pred_dbp_right", "pred_hr_right", "pred_age_right"]].values

        result_left = (get_performance(pred_left, targets, ))
        result_right = (get_performance(pred_right, targets, ))
        result = (get_performance(0.5 * (pred_left + pred_right), targets, ))
        all_results[model_type]["left"] = result_left
        all_results[model_type]["right"] = result_right
        all_results[model_type]["avg"] = result
    df = build_dataframe(all_results)
    # 按照性能排序
    export_excel(df, save_path)