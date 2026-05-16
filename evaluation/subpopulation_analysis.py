# 根据人口统计学获取
import os
import pandas as pd
from evaluation.model_metric import get_performance,results_bygrouped
from predictions_to_excel import build_dataframe,flatten_result
import datetime
formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")


groups = {
            # "age": [[0, 18], [18, 45], [45, 60], [60, 75], [75, 1000]],
            "age": [[0, 60], [60, 75], [75, 1000]],
            "bmi":[[0,23.9],[23.9,27.9],[27.9,100]],
            "gender":[[1],[2]],
            # "position":[[1],[2],[3]],
            "bp":[["normotensive"],["stage-1 HBP"],["stage-2 HBP"]]
            }

if __name__ == '__main__':
    pkl_dir = "/home/cjr/PPGBen/predictions/"
    save_path = f"../results/subpopulation_results_{formatted_date}.xlsx"
    TARGET_COLS = ["sbp_fix", "dbp_fix", "pr_ref", "age"]
    PRED_COLS = [
        "pred_sbp", "pred_dbp", "pred_hr", "pred_age",
        "pred_sbp_right", "pred_dbp_right", "pred_hr_right", "pred_age_right"
    ]
    order = ["CRNN", "ACNN", "ResNet1D", "Net1D", "LSTM", "Efficient1D", "AutoFormer",
             "PatchTST", "Informer", "iTransformer", "ResNet1DMoE", "CSFM", "PaAno"]

    pkl_files = os.listdir(pkl_dir)
    pkl_files.sort(key=lambda x: order.index(x.split("_")[0]))
    index_all = []
    with pd.ExcelWriter(save_path) as writer:
        for pkl_file in pkl_files:
            all_results = {}; left_results = {}; right_results = {}
            result_all = pd.read_pickle(os.path.join(pkl_dir, pkl_file))
            model_type = pkl_file.split("_")[0]
            preds = result_all[["pred_sbp", "pred_dbp", "pred_hr", "pred_age"]].values
            preds_right = result_all[["pred_sbp_right", "pred_dbp_right", "pred_hr_right", "pred_age_right"]].values
            targets = result_all[TARGET_COLS].values
            key = "All"
            df_left = flatten_result(get_performance(preds, targets))
            left_results[key] = df_left
            df_right = flatten_result(get_performance(preds_right, targets))
            right_results[key] = df_right
            df_mean = flatten_result(get_performance(0.5 * (preds + preds_right), targets))
            all_results[key] = df_mean
            for group_key in groups.keys():
                for group_value in groups[group_key]:
                    if group_key=="bp":
                        if group_value[0]=="normotensive":
                            result_df = result_all[
                                (result_all["sbp_fix"] <= 129) & (result_all["dbp_fix"] <= 79)]
                            val = group_value[0]
                        elif "1" in group_value[0]:
                            result_df = result_all[
                                ((result_all["sbp_fix"] > 129) |  (result_all["dbp_fix"] > 79))&
                                (result_all["sbp_fix"] <= 139)&
                                (result_all["dbp_fix"] <= 89) ]
                            val = group_value[0]
                        else:
                            result_df = result_all[(result_all["sbp_fix"] >139)|(result_all["dbp_fix"] > 89)]
                            val = group_value[0]
                    else:
                        if len(group_value) == 2:
                            result_df = result_all[(result_all[group_key] >= group_value[0]) & (result_all[group_key] < group_value[1])]
                            val = str(group_value[0]) + "--" + str(group_value[1])
                        else:
                            result_df = result_all[result_all[group_key] == group_value[0]]
                            val = str(group_value[0])

                    print(val,len(result_df))
                    # index_all.append()
                    key = group_key + "_" + val
                    preds = result_df[["pred_sbp", "pred_dbp", "pred_hr", "pred_age"]].values
                    preds_right = result_df[[ "pred_sbp_right", "pred_dbp_right", "pred_hr_right", "pred_age_right"]].values
                    targets = result_df[TARGET_COLS].values
                    df_left = flatten_result(get_performance(preds,targets))
                    left_results[key] = df_left
                    df_right = flatten_result(get_performance(preds_right,targets))
                    right_results[key] = df_right
                    df_mean = flatten_result(get_performance(0.5*(preds+preds_right),targets))
                    all_results[key] = df_mean

            df_result = pd.DataFrame(all_results).transpose()
            df_result.to_excel(writer, sheet_name=model_type+"_all")
            df_result = pd.DataFrame(left_results).transpose()
            df_result.to_excel(writer, sheet_name=model_type+"_left",)
            df_result = pd.DataFrame(right_results).transpose()
            df_result.to_excel(writer, sheet_name=model_type+"_right",)
        #
        # targets = result_df[["sbp_fix", "dbp_fix", "pr_ref", "age"]].values
        # preds = result_df[["pred_sbp", "pred_dbp", "pred_hr", "pred_age","pred_sbp_right", "pred_dbp_right", "pred_hr_right", "pred_age_right"]].values
        # # pred_right = result_df[[]].values
        #
        # original_result = results_bygrouped(preds,targets)
        # df = build_dataframe({model_type:original_result})
        # df.to_excel(writer, sheet_name=model_type,index=False)
