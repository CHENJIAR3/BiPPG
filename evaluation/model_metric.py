import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from itertools import combinations
from scipy import stats


def BP_evalution(p_sbps,r_sbps,p_dbps,r_dbps,only_num=False):
    count = p_sbps.shape[0]

    assert p_sbps.shape[0]==r_sbps.shape[0]
    BP_result = {}
    BP_result ["count"] = count
    # 计算皮尔逊相关系数并打印
    sbp_errors = p_sbps - r_sbps
    dbp_errors = p_dbps - r_dbps
    sbp_corr, sbp_p_value = stats.pearsonr(r_sbps, p_sbps)
    dbp_corr, dbp_p_value = stats.pearsonr(r_dbps, p_dbps)

    # 计算并打印统计数据
    BP_result["SBP_MAE"] = np.round(np.mean(abs(r_sbps - p_sbps)), 2)
    BP_result["SBP_ME"] = np.round(np.mean((r_sbps - p_sbps)), 2)
    BP_result["SBP_SDE"] = np.round(np.std((r_sbps - p_sbps)), 2)


    if only_num is False:
        if abs(np.mean((sbp_errors))) <= 5 and np.std((sbp_errors)) <= 8:
            BP_result["SBP_AAMI"] = "Pass"
        else:
            BP_result["SBP_AAMI"] = "Fail"
        #
        # if BP_result["SBP_MAE"]<=5:
        #     BP_result["SBP_IEEE1708"]="A"
        # elif BP_result["SBP_MAE"]<=6:
        #     BP_result["SBP_IEEE1708"]="B"
        # elif BP_result["SBP_MAE"]<=7:
        #     BP_result["SBP_IEEE1708"]="C"
        # else:
        #     BP_result["SBP_IEEE1708"]="D"

    # BP_result["SBP_Corr"] = np.round(sbp_corr, 2)
    # BP_result["SBP_Corr_p"] = np.round(sbp_p_value, 4)

    SBP_CPE_5, SBP_CPE_10, SBP_CPE_15 = np.sum(abs(sbp_errors) <= 5) / (count), np.sum(abs(sbp_errors) <= 10) / (
        count), np.sum(abs(sbp_errors) <= 15) / (count)

    if only_num is False:
        BP_result["SBP_CPE_5"] = f"{np.round(SBP_CPE_5 * 100, 2)}%"
        BP_result["SBP_CPE_10"] = f"{np.round(SBP_CPE_10 * 100, 2)}%"
        BP_result["SBP_CPE_15"] = f"{np.round(SBP_CPE_15 * 100, 2)}%"
        if SBP_CPE_5 >= 0.6 and SBP_CPE_10 >= 0.85 and SBP_CPE_15 >= 0.95:
            BP_result["SBP_BHS"] = "A"
        elif SBP_CPE_5 >= 0.5 and SBP_CPE_10 >= 0.75 and SBP_CPE_15 >= 0.90:
            BP_result["SBP_BHS"] = "B"
        elif SBP_CPE_5 >= 0.4 and SBP_CPE_10 >= 0.65 and SBP_CPE_15 >= 0.85:
            BP_result["SBP_BHS"] = "C"
        else:
            BP_result["SBP_BHS"] = "D"
    else:
        BP_result["SBP_CPE_5"] = np.round(SBP_CPE_5 * 100, 2)
        BP_result["SBP_CPE_10"] = np.round(SBP_CPE_10 * 100, 2)
        BP_result["SBP_CPE_15"] = np.round(SBP_CPE_15 * 100, 2)
    BP_result["DBP_MAE"] = np.round(np.mean(abs(r_dbps - p_dbps)), 2)

    BP_result["DBP_ME"] = np.round(np.mean((r_dbps - p_dbps)), 2)
    BP_result["DBP_SDE"] = np.round(np.std((r_dbps - p_dbps)), 2)

    if only_num is False:
        if abs(np.mean((dbp_errors))) <= 5 and np.std((dbp_errors)) <= 8:
            BP_result["DBP_AAMI"] = "Pass"
        else:
            BP_result["DBP_AAMI"] = "Fail"

        # if BP_result["DBP_MAE"]<=5:
        #     BP_result["DBP_IEEE1708"]="A"
        # elif BP_result["DBP_MAE"]<=6:
        #     BP_result["DBP_IEEE1708"]="B"
        # elif BP_result["DBP_MAE"]<=7:
        #     BP_result["DBP_IEEE1708"]="C"
        # else:
        #     BP_result["DBP_IEEE1708"]="D"

    # BP_result["DBP_Corr"] = np.round(dbp_corr, 2)
    # BP_result["DBP_Corr_p"] = np.round(dbp_p_value, 4)


    DBP_CPE_5, DBP_CPE_10, DBP_CPE_15 = np.sum(abs(dbp_errors) <= 5) / (count), np.sum(abs(dbp_errors) <= 10) / (
        count), np.sum(abs(dbp_errors) <= 15) / (count)

    if only_num is False:
        BP_result["DBP_CPE_5"] = f"{np.round(DBP_CPE_5 * 100, 2)}%"
        BP_result["DBP_CPE_10"] = f"{np.round(DBP_CPE_10 * 100, 2)}%"
        BP_result["DBP_CPE_15"] = f"{np.round(DBP_CPE_15 * 100, 2)}%"
        if DBP_CPE_5 >= 0.6 and DBP_CPE_10 >= 0.85 and DBP_CPE_15 >= 0.95:
            BP_result["DBP_BHS"] = "A"
        elif DBP_CPE_5 >= 0.5 and DBP_CPE_10 >= 0.75 and DBP_CPE_15 >= 0.90:
            BP_result["DBP_BHS"] = "B"
        elif DBP_CPE_5 >= 0.4 and DBP_CPE_10 >= 0.65 and DBP_CPE_15 >= 0.85:
            BP_result["DBP_BHS"] = "C"
        else:
            BP_result["DBP_BHS"] = "D"

    else:
        BP_result["DBP_CPE_5"] = np.round(DBP_CPE_5 * 100, 2)
        BP_result["DBP_CPE_10"] = np.round(DBP_CPE_10 * 100, 2)
        BP_result["DBP_CPE_15"] = np.round(DBP_CPE_15 * 100, 2)
    return BP_result


def reg_evalution(preds, reals, type="SBP",detailed_flag=True):
    from scipy import stats
    count = preds.shape[0]

    assert preds.shape[0] == reals.shape[0]
    df_result = {}
    df_result["count"] = count

    errors = preds - reals

    corr, p_value = stats.pearsonr(reals, preds)

    df_result[f"{type}_ME"] = np.round(np.mean((preds - reals)), 2)
    df_result[f"{type}_MAE"] = np.round(np.mean(abs(preds - reals)), 2)
    df_result[f"{type}_SDE"] = np.round(np.std((preds - reals)), 2)

    # # ✅ 新增 MAD（中位数绝对偏差，比 MAE 更鲁棒）
    # df_result[f"{type}_MAD"] = np.round(np.median(np.abs(errors - np.median(errors))), 2)

    # ✅ 新增 MAPE（注意避免除以零）
    mask = reals != 0
    mape = np.mean(np.abs(errors[mask] / reals[mask])) * 100
    df_result[f"{type}_R"] = np.round(corr, 4)
    df_result[f"{type}_MAPE"] = f"{np.round(mape, 2)}%"
    if detailed_flag:

        # ✅ 新增 RMSE
        df_result[f"{type}_RMSE"] = np.round(np.sqrt(np.mean(errors ** 2)), 2)
        df_result[f"{type}_P"] = np.round(p_value, 4)

        CPE_5  = np.sum(abs(errors) <= 5)  / count
        CPE_10 = np.sum(abs(errors) <= 10) / count
        CPE_15 = np.sum(abs(errors) <= 15) / count

        df_result[f"{type}_CPE_5"]  = f"{np.round(CPE_5  * 100, 2)}%"
        df_result[f"{type}_CPE_10"] = f"{np.round(CPE_10 * 100, 2)}%"
        df_result[f"{type}_CPE_15"] = f"{np.round(CPE_15 * 100, 2)}%"

        if "BP" in type:
            if abs(np.mean(errors)) <= 5 and np.std(errors) <= 8:
                df_result[f"{type}_AAMI"] = "Pass"
            else:
                df_result[f"{type}_AAMI"] = "Fail"

            if CPE_5 >= 0.6 and CPE_10 >= 0.85 and CPE_15 >= 0.95:
                df_result[f"{type}_BHS"] = "A"
            elif CPE_5 >= 0.5 and CPE_10 >= 0.75 and CPE_15 >= 0.90:
                df_result[f"{type}_BHS"] = "B"
            elif CPE_5 >= 0.4 and CPE_10 >= 0.65 and CPE_15 >= 0.85:
                df_result[f"{type}_BHS"] = "C"
            else:
                df_result[f"{type}_BHS"] = "D"

    return df_result
def _to_label(x):
    """
    将预测或标签统一转成 1D label
    """
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    if x.ndim == 2:  # logits or probs
        x = np.argmax(x, axis=1)

    return x.astype(int)

def cls_evalution(predictions, targets, type="gender"):
    """
    分类性能评估函数

    Parameters
    ----------
    predictions : array-like
        预测结果（label 或 logits）
    targets : array-like
        真实标签
    type : str
        任务类型：'gender' or 'pos'

    Returns
    -------
    result : dict
        包含 overall + per-class + macro average
    """

    y_pred = _to_label(predictions)
    y_true = _to_label(targets)

    # -------- 类别定义 --------
    if type == "gender":
        labels = [1, 2]
    elif type == "pos":
        labels = [1, 2, 3]
    else:
        labels = np.unique(y_true).tolist()

    # -------- Overall Accuracy --------
    acc = accuracy_score(y_true, y_pred)

    # -------- Precision / Recall / F1 --------
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    # -------- 组织结果 --------
    result = {
        "type": type,
        "accuracy": acc,
        "macro_avg": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro
        },
        "per_class": {}
    }

    for i, cls in enumerate(labels):
        result["per_class"][cls] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i]
        }

    return result
def get_performance(predictions,targets):
    final_results = {
        "bp": {},
        # "classification": {},
        "regression": {}
    }
    try:
        # -------- BP --------
        # bp_result = reg_evalution(
        #     predictions[:, 0], targets[:, 0],
        #     "SBP"
        #     # predictions[:, 1], targets[:, 1]
        # )
        # final_results["bp"]["SBP"] = bp_result
        # bp_result = reg_evalution(
        #     predictions[:, 1], targets[:, 1],
        #     "DBP"
        #     # predictions[:, 1], targets[:, 1]
        # )
        # final_results["bp"]["DBP"] = bp_result
        label_keys = ["SBP","DBP","HR","Age",]
        for i in range(targets.shape[1]):
            key = label_keys[i]
            reg_result = reg_evalution(
                predictions[:, i],
                targets[:, i],
                type=key
            )
            final_results["regression"][key] = reg_result
    except:
        final_results = None
    return final_results
def get_simple_performance(predictions,targets,metrics=["SBP","DBP","HR","Age"]):
    """
    pred:   (N, 4)
    target: (N, 4)
    返回 DataFrame，一行
    """
    num = predictions.shape[1]
    result = {}
    for i in range(num):
        # reg_evalution 返回 dict，直接 update 合并
        result.update(reg_evalution(predictions[:, i], targets[:, i], metrics[i],detailed_flag=False))
    return pd.DataFrame([result])   # 套列表，标量dict → 一行DataFrame

def save_all_metrics_to_xlsx(all_results, save_path="all_metrics.xlsx"):
    """
    将所有实验结果（BP + 分类 + 回归）写入一个结构化的 Excel 文件

    Parameters
    ----------
    all_results : dict
        结构示例：
        {
            exp_name: {
                "bp": {...},
                "classification": {...},
                "regression": {...}
            }
        }

    save_path : str
        保存的 xlsx 文件路径
    """

    # ===================== BP metrics =====================
    bp_rows = []
    for exp, res in all_results.items():
        if res is None:
            continue
        bp = res.get("bp", {})
        if not bp:
            continue

        bp_rows.append({
            "exp": exp,
            "target": "BP",
            "count": bp["count"],
            "SBP_MAE": bp["SBP_MAE"],
            "SBP_ME": bp["SBP_ME"],
            "SBP_SDE": bp["SBP_SDE"],
            "SBP_AAMI": bp["SBP_AAMI"],
            "SBP_CPE_5": bp["SBP_CPE_5"],
            "SBP_CPE_10": bp["SBP_CPE_10"],
            "SBP_CPE_15": bp["SBP_CPE_15"],
            "SBP_BHS": bp["SBP_BHS"],
            "DBP_MAE": bp["DBP_MAE"],
            "DBP_ME": bp["DBP_ME"],
            "DBP_SDE": bp["DBP_SDE"],
            "DBP_AAMI": bp["DBP_AAMI"],
            "DBP_CPE_5": bp["DBP_CPE_5"],
            "DBP_CPE_10": bp["DBP_CPE_10"],
            "DBP_CPE_15": bp["DBP_CPE_15"],
            "DBP_BHS": bp["DBP_BHS"],
        })



    df_bp = pd.DataFrame(bp_rows)

    # ===================== Regression metrics =====================
    reg_rows = []
    for exp, res in all_results.items():
        if res is None:
            continue
        reg_res = res.get("regression", {})
        row = {"exp": exp}
        for key, r in reg_res.items():
            row[f"{key}_MAE"] = r[f"{key}_MAE"]
            row[f"{key}_ME"] = r[f"{key}_ME"]
            row[f"{key}_SDE"] = r[f"{key}_SDE"]
            # row[f"{key}_CPE_5"] = r[f"{key}_CPE_5"]
            # row[f"{key}_CPE_10"] = r[f"{key}_CPE_10"]
            # row[f"{key}_CPE_15"] = r[f"{key}_CPE_15"]
        reg_rows.append(row)

    df_reg = pd.DataFrame(reg_rows)

    # ===================== Write to Excel =====================
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df_bp.to_excel(writer, sheet_name="BP_metrics", index=False)
        df_reg.to_excel(writer, sheet_name="Regression_metrics", index=False)

    print(f"[OK] All metrics saved to {save_path}")


def build_group_labels(edges):

    labels = []

    for i in range(len(edges) - 1):

        left = edges[i]
        right = edges[i + 1]

        if np.isinf(left):
            labels.append(f"minus_{int(right)}")

        elif np.isinf(right):
            labels.append(f"{int(left)}_plus")

        else:
            labels.append(f"{int(left)}_{int(right)}")

    return labels

def results_bygrouped(predictions, targets,
                      pred_num=4, group_edges=[0,5,np.inf],
                      label_based = False #是否基于标签统计
                      ):

    all_results = {}

    hands = {"Left":0, "Right":pred_num}

    group_labels = build_group_labels(group_edges)

    conditions = [
        {"name":"sbp","idx":0},
        {"name":"dbp","idx":1},
        {"name":"hr","idx":2},
        {"name":"age","idx":3},
    ]

    cond_names = [c["name"] for c in conditions]
    cond_indices = [c["idx"] for c in conditions]

    # ---------- 工具函数 ----------
    def compute_group_perf(pred, delta, name):

        for i in range(len(group_edges)-1):
            mask = (delta >= group_edges[i]) & (delta < group_edges[i+1])

            perf = None if mask.sum()==0 else get_performance(
                pred[mask], targets[mask]
            )
            all_results[f"{name}_{group_labels[i]}"] = perf


    # ---------- 左右手差异矩阵 ----------
    left_vals  = predictions[:, cond_indices]
    right_vals = predictions[:, pred_num + np.array(cond_indices)]

    lr_deltas = np.abs(left_vals - right_vals)  # (N , C)


    # =============================
    # 1. Left / Right
    # =============================


    for hand, hand_idx in hands.items():

        pred_hand = predictions[:, hand_idx:hand_idx+pred_num]

        all_results[hand] = get_performance(pred_hand, targets)
        if label_based:
            # ----- prediction vs target -----
            for c in conditions:

                delta = np.abs(
                    predictions[:, hand_idx + c["idx"]] - targets[:, c["idx"]]
                )

                compute_group_perf(
                    pred_hand,
                    delta,
                    f"{hand}_{c['name']}"
                )

        # ----- left vs right single condition -----
        for i,c in enumerate(conditions):

            compute_group_perf(
                pred_hand,
                lr_deltas[:,i],
                f"{hand}_lr_{c['name']}"
            )

        # ----- 任意条件组合 -----
        for k in range(2, len(conditions)+1):

            for combo in combinations(range(len(conditions)), k):

                combo_name = "+".join([cond_names[i] for i in combo])

                delta = np.max(lr_deltas[:,combo], axis=1)

                compute_group_perf(
                    pred_hand,
                    delta,
                    f"{hand}_lr_{combo_name}"
                )


    # =============================
    # 2. Global Average
    # =============================

    pred_left  = predictions[:,:pred_num]
    pred_right = predictions[:,pred_num:pred_num*2]

    pred_avg = (pred_left + pred_right)/2

    all_results["Global_Average"] = get_performance(pred_avg, targets)


    # ----- global prediction error -----
    if label_based:
        for c in conditions:

            delta = np.abs(pred_avg[:,c["idx"]] - targets[:,c["idx"]])

            compute_group_perf(
                pred_avg,
                delta,
                f"Global_{c['name']}"
            )


    # ----- global left vs right -----
    for i,c in enumerate(conditions):

        compute_group_perf(
            pred_avg,
            lr_deltas[:,i],
            f"Global_lr_{c['name']}"
        )


    # ----- global 任意组合 -----
    for k in range(2, len(conditions)+1):

        for combo in combinations(range(len(conditions)), k):

            combo_name = "+".join([cond_names[i] for i in combo])

            delta = np.max(lr_deltas[:,combo], axis=1)

            compute_group_perf(
                pred_avg,
                delta,
                f"Global_lr_{combo_name}"
            )
    all_results["Left_as_reference"] = get_performance(pred_right, pred_left)
    all_results["Right_as_reference"] = get_performance(pred_left, pred_right)

    return all_results

def results_byths(predictions, targets,
                      pred_num=4, ths=[5,10,15,np.inf],
                      label_based = False #是否基于标签统计
                      ):

    all_results = {}

    hands = {"Left":0, "Right":pred_num}


    conditions = [
        {"name":"sbp","idx":0},
        {"name":"dbp","idx":1},
        {"name":"hr","idx":2},
        {"name":"age","idx":3},
    ]

    cond_names = [c["name"] for c in conditions]
    cond_indices = [c["idx"] for c in conditions]

    # ---------- 工具函数 ----------
    def compute_group_th(pred, delta, name):

        for i in range(len(ths)):
            mask = (delta < ths[i])

            perf = None if mask.sum()==0 else get_performance(
                pred[mask], targets[mask]
            )
            all_results[f"{name}_{ths[i]}"] = perf


    # ---------- 左右手差异矩阵 ----------
    left_vals  = predictions[:, cond_indices]
    right_vals = predictions[:, pred_num + np.array(cond_indices)]

    lr_deltas = np.abs(left_vals - right_vals)  # (N , C)



    # =============================
    # 2. Global Average
    # =============================

    pred_left  = predictions[:,:pred_num]
    pred_right = predictions[:,pred_num:pred_num*2]

    pred_avg = (pred_left + pred_right)/2

    all_results["Global_Average"] = get_performance(pred_avg, targets)

    delta = np.max(lr_deltas, axis=1)

    compute_group_th(
        pred_avg,
        delta,
        f"Global_lr_"
    )

    return all_results