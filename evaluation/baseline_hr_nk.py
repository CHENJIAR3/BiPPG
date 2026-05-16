import numpy as np
import torch
import datetime
from model_metric import get_performance, save_all_metrics_to_xlsx,reg_evalution
from tqdm import tqdm
import pickle
import pandas as pd

formatted_date = datetime.datetime.now().strftime("%Y-%m-%d:%H-%M-%S")

if __name__ == '__main__':
    pkl_file = "../Preprocessing/data_splits_paths.pkl"

    data = pd.read_pickle(pkl_file)
    train_paths,test_paths,val_paths = data['train'],data['test'],data['val']
    pr_est_cols = ['pr_est(ppg_g_1)', 'pr_est(ppg_g_2)', 'pr_est(ppg_ga_1)', 'pr_est(ppg_ga_2)', 'pr_est(ppg_r_1)', 'pr_est(ppg_r_2)',
     'pr_est(ppg_ir_1)', 'pr_est(ppg_ir_2)']
    targets = np.empty((0,))
    predictions = np.empty((0,len(pr_est_cols)))
    for test_path in tqdm(test_paths):
        data = pd.read_pickle(test_path)
        pr_ref = data["pr_ref"].values
        pr_est = np.concatenate(
            [np.mean(np.stack(data[key].values),axis=1)[:,None] for key in pr_est_cols],
            axis=1
        ).astype(np.float32)
        targets =  np.concatenate((targets, pr_ref),axis=0)
        predictions = np.concatenate((predictions, pr_est),axis=0)
        # break
    #
    data_saved = {"targets":targets, "predictions":predictions}
    with open("../results/nk_predictions.pkl", "wb") as f:
        pickle.dump(data_saved, f)
    all_result = {}

    #
    # # ---------- evaluate ----------
    for i,ppg_name  in enumerate(pr_est_cols):
        ppg_ch  = ppg_name.replace("pr_est(ppg_","").replace(")","")
        result = reg_evalution(predictions[:,i], targets,type="HR")
        all_result[f"{ppg_ch}"] = result
    #
    # # ---------- save ----------
    result_file = f"../results/hr_nk_baseline_{formatted_date}.xlsx"
    df = pd.DataFrame(all_result)
    df.to_excel(result_file)
    # save_all_metrics_to_xlsx(all_result, result_file)
    #
    # print("Baseline evaluation finished.")
    # print("Saved to:", result_file)
