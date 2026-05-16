from collections import defaultdict, OrderedDict
import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from utils import args
import pickle
import datetime
from joblib import Parallel, delayed
from Preprocessing.preprocessing_functions import calibrate_height
from scipy.stats import kruskal, chi2_contingency


formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")
# 定义全局分布区间（可根据需求调整）
AGE_BINS = [0, 18, 45, 60, 75, 100]
# 医学视角 # 常见医学分组是：# 儿童 / 青少年（<18）# 青年（18–44）# 中年（45–59）# 老年（≥60 或 ≥65）
AGE_LABELS = ["0-18岁", "18-45岁", "45-60岁", "60-75岁", "75-100岁"]

BMI_BINS = [0, 18.5, 23.9, 27.9, 100]  # BMI区间（中国标准）
BMI_LABELS = ["偏瘦(<18.5)", "正常(18.5-23.9)", "超重(24.0-27.9)", "肥胖(≥28.0)"]
# 血压分组区间（中国高血压防治指南2023版）
SBP_BINS = [0, 90, 120, 140, 160, 180, np.inf]  # SBP分级
SBP_LABELS = ["低血压(<90)","正常(90-119)", "正常高值(120-139)", "1级高血压(140-159)", "2级高血压(160-179)", "3级高血压(≥180)"]
DBP_BINS = [0, 60, 80, 90, 100, 110, np.inf]  # DBP分级
DBP_LABELS = ["低血压（<60）","正常(60-79)","正常高值(80-89)", "1级高血压(90-99)", "2级高血压(100-109)", "3级高血压(≥110)"]
HR_BINS = [0, 60, 100, 120, np.inf]
HR_LABELS = ["偏慢(<60)", "正常(60-100)", "偏快(100-119)", "明显偏快(≥120)"]
POS_BINS = ["lay","sit","stand"]
# POS_LABELS = ["躺姿","坐姿","站姿"]
# 姿势映射规则（与之前保持一致）
# 姿势统计标签（包含原字符串+映射后数值，便于理解）
POS_LABELS = ["lay", "sit", "stand"]
# 姿势映射后对应的数值区间（左闭右开，用于统计）
GENDER_BINS = ["Male","Female"]
GENDER_LABELS = [1,2]
PPGQ_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
PPGQ_LABELS = [
    "0–0.25",
    "0.25–0.5",
    "0.5–0.75",
    "0.75–1.0"
]

BP_LEVEL_LABELS = [1,2,3,4,5,6]

def kruskal_test_across_splits(split_data_dict):
    """
    split_data_dict = {
        "train": np.array([...]),
        "val": np.array([...]),
        "test": np.array([...])
    }
    """
    arrays = [v for v in split_data_dict.values() if len(v) > 0]

    if len(arrays) < 2:
        return np.nan

    stat, p = kruskal(*arrays)
    return round(p, 4)
def chi_square_test_across_splits(count_df):
    """
    count_df:
        index   -> category
        columns -> train / val / test
    """
    if count_df.shape[1] < 2:
        return np.nan

    chi2, p, _, _ = chi2_contingency(count_df.fillna(0))
    return round(p, 4)

# POS_BINS = [0.5, 1.5, 2.5, 3.5]  # 对应1、2、3三个数值




def process_single_path(path, mode):
    """单文件处理函数，用于多进程并行"""
    try:
        print(path)
        data = pd.read_pickle(path)
        if data.empty: return None

        # 提取基础信息
        row0 = data.iloc[0]
        weight = row0["weight"]
        height = calibrate_height(row0["height"])
        age = row0["age"]
        gender = row0["gender"]
        if age<16:
            print(path,"Age wrong",age)
        # 预计算 BMI
        bmi = np.nan
        if 0 < height <= 250:
            bmi = weight / (height / 100) ** 2
            if not (10 <= bmi <= 50): bmi = np.nan

        # 血压去重 (关键加速点：根据 mode 选择去重列)
        subset_col = "bp_idx" if mode == "ringbp" else "Record_ID"
        df_unique = data.drop_duplicates(subset=[subset_col], keep="first")
        if mode != "ringbp":
            df_unique = df_unique.dropna(subset=["sbp_fix"])

        # 提取数组并直接过滤 (转为 numpy 速度最快)
        sbp = df_unique["sbp_fix"].to_numpy()
        dbp = df_unique["dbp_fix"].to_numpy()
        hr = df_unique["pr_ref"].to_numpy()
        pos = df_unique["position"].to_numpy()
        bp_lvl = df_unique["BP_Level"].to_numpy()

        # SBP/DBP 阈值过滤
        sbp_v = sbp[(sbp >= 40) & (sbp <= 370) & (~np.isnan(sbp))]
        dbp_v = dbp[(dbp >= 20) & (dbp <= 160) & (~np.isnan(dbp))]

        # 质量数据统计 (PPG Quality)
        ppg_q = {col: np.stack(data[col]).mean(axis=1) for col in data.columns if col.startswith("ppg_quality")}
        for col in data.columns:
            if col.startswith("ppg_quality"):
                break
        # print(ppg_q[col].shape)
        # print(data.shape,len(ppg_q))
        return {
            "age": age if (0 < age <= 120) else np.nan,
            "bmi": bmi,
            "gender": gender,
            "sbp": sbp_v,
            "dbp": dbp_v,
            "hr": hr,
            "pos": pos,
            "bp_lvl": bp_lvl,
            "ppg_q": ppg_q,
            "raw_counts": len(df_unique)
        }
    except:
        return None
def get_info(allpaths, mode="ringbp",return_result="df"):
    info = defaultdict(float)
    info["subjects"] = len(allpaths)  # 总人数
    info["BP_recordings"] = 0  # 血压记录总数

    # # ==== 基础统计汇总 ====
    # # 年龄统计（基于有效年龄样本）
    # 使用 joblib 进行并行处理 (n_jobs=-1 使用所有核心)
    results = Parallel(n_jobs=-1)(delayed(process_single_path)(p, mode) for p in allpaths)
    results = [r for r in results if r is not None]

    # 结果汇总 (使用 numpy concatenate 替代多次 append), prefer="threads"
    age_all = np.concatenate([[r['age']] for r in results if not np.isnan(r['age'])])
    bmi_all = np.concatenate([[r['bmi']] for r in results if not np.isnan(r['bmi'])])
    gender_all = np.concatenate([[r['gender']] for r in results])

    sbp_all = np.concatenate([r['sbp'] for r in results])
    dbp_all = np.concatenate([r['dbp'] for r in results])
    hr_all = np.concatenate([r['hr'] for r in results])
    pos_all = np.concatenate([r['pos'] for r in results])
    bp_lvl_all = np.concatenate([r['bp_lvl'] for r in results])
    # # PPG Quality 汇总
    ppgq_all = defaultdict(list)
    for r in results:
        for k, v in r['ppg_q'].items():
            ppgq_all[k].append(v)
    if len(age_all) > 0:
        info["age_mean"] = np.mean(age_all)
        info["age_std"] = np.std(age_all)
        info["age_min"] = np.min(age_all)
        info["age_max"] = np.max(age_all)
    else:
        info["age_mean"] = info["age_std"] = info["age_min"] = info["age_max"] = np.nan
    # 心率统计（基于有效样本）

    if len(hr_all) > 0:
        info["hr_mean"] = np.mean(hr_all)
        info["hr_std"] = np.std(hr_all)
        info["hr_min"] = np.min(hr_all)
        info["hr_max"] = np.max(hr_all)
    else:
        info["hr_mean"] = info["hr_std"] = info["hr_min"] = info["hr_max"] = np.nan

    # BMI统计（基于有效BMI样本）
    if len(bmi_all) > 0:
        info["bmi_mean"] = np.mean(bmi_all)
        info["bmi_std"] = np.std(bmi_all)
        info["bmi_min"] = np.min(bmi_all)
        info["bmi_max"] = np.max(bmi_all)
    else:
        info["bmi_mean"] = info["bmi_std"] = info["bmi_min"] = info["bmi_max"] = np.nan


    # 血压统计（基于有效血压样本）
    if len(sbp_all) > 0:
        info["sbp_mean_std"] = f"{np.round(np.mean(sbp_all), 2)} ± {np.round(np.std(sbp_all), 2)}"
    else:
        info["sbp_mean_std"] = "无有效数据"
    if len(dbp_all) > 0:
        info["dbp_mean_std"] = f"{np.round(np.mean(dbp_all), 2)} ± {np.round(np.std(dbp_all), 2)}"
    else:
        info["dbp_mean_std"] = "无有效数据"

    # ==== 年龄分布统计 ====
    age_distribution = defaultdict(dict)
    if len(age_all) > 0:
        age_counts, _ = np.histogram(age_all, bins=AGE_BINS)
        for i, (label, count) in enumerate(zip(AGE_LABELS, age_counts)):
            age_distribution[label]["频数"] = count
            age_distribution[label]["占比(%)"] = round(count / len(age_all) * 100, 2)
        age_distribution["所有"]["频数"] = len(age_all)
        age_distribution["所有"]["占比(%)"] =  round(1 * 100, 2)
    else:
        age_distribution["提示"] = "无有效年龄数据"

    # ==== BMI分布统计 ====
    bmi_distribution = defaultdict(dict)
    if len(bmi_all) > 0:
        bmi_counts, _ = np.histogram(bmi_all, bins=BMI_BINS)
        for i, (label, count) in enumerate(zip(BMI_LABELS, bmi_counts)):
            bmi_distribution[label]["频数"] = count
            bmi_distribution[label]["占比(%)"] = round(count / len(bmi_all) * 100, 2)
        bmi_distribution["所有"]["频数"] = len(bmi_all)
        bmi_distribution["所有"]["占比(%)"] =  round(1 * 100, 2)

    else:
        bmi_distribution["提示"] = "无有效BMI数据"

    # ==== 姿势分布统计 ====
    pos_distribution = defaultdict(dict)
    if len(pos_all) > 0:

        # 3. 填充每个姿势的频数和占比（占比=该姿势数/有效姿势总数）
        for i, (label) in enumerate(POS_LABELS):
            count = np.sum(pos_all == label)
            pos_distribution[label]["频数"] = count
            pos_distribution[label]["占比(%)"] = round(count / len(pos_all) * 100, 2)
        pos_distribution["所有"]["频数"] = len(pos_all)
        pos_distribution["所有"]["占比(%)"] = 100.0  # 固定100%

    else:
        # 修正提示信息（原错误提示"无有效年龄数据"，改为姿势相关）
        pos_distribution["提示"] = "无有效姿势数据"

    # ==== 姿势分布统计 ====
    bp_level_distribution = defaultdict(dict)
    if len(bp_lvl_all) > 0:

        for i, (label) in enumerate(BP_LEVEL_LABELS):
            count = np.sum(bp_lvl_all == label)
            bp_level_distribution[label]["频数"] = count
            bp_level_distribution[label]["占比(%)"] = round(count / len(bp_lvl_all) * 100, 2)
        # 4. 修正"所有"的统计
        bp_level_distribution["所有"]["频数"] = len(bp_lvl_all)
        bp_level_distribution["所有"]["占比(%)"] = 100.0  # 固定100%

    else:
        bp_level_distribution["提示"] = "无有效数据"
    # ==== 性别分布统计 ====
    gender_distribution = defaultdict(dict)
    if len(gender_all) > 0:

        # 3. 填充每个姿势的频数和占比（占比=该姿势数/有效姿势总数）
        for i, (label) in enumerate(GENDER_LABELS):
            count = np.sum(gender_all == label)
            gender_distribution[label]["频数"] = count
            gender_distribution[label]["占比(%)"] = round(count / len(gender_all) * 100, 2)
        # 4. 修正"所有"的统计
        gender_distribution["所有"]["频数"] = len(gender_all)
        gender_distribution["所有"]["占比(%)"] = 100.0  # 固定100%

    else:
        gender_distribution["提示"] = "无有效性别数据"
    # ==== SBP分布统计（临床分级） ====
    sbp_distribution = defaultdict(dict)
    if len(sbp_all) > 0:
        sbp_counts, _ = np.histogram(sbp_all, bins=SBP_BINS)
        for i, (label, count) in enumerate(zip(SBP_LABELS, sbp_counts)):
            sbp_distribution[label]["频数"] = count
            sbp_distribution[label]["占比(%)"] = round(count / len(sbp_all) * 100, 2)
        sbp_distribution["所有"]["频数"] = len(sbp_all)
        sbp_distribution["所有"]["占比(%)"] = round(1 * 100, 2)
    else:
        sbp_distribution["提示"] = "无有效SBP数据"
    # ==== HR分布统计（临床分级） ====
    hr_distribution = defaultdict(dict)
    if len(hr_all) > 0:
        hr_counts, _ = np.histogram(hr_all, bins=HR_BINS)
        for i, (label, count) in enumerate(zip(HR_LABELS, hr_counts)):
            hr_distribution[label]["频数"] = count
            hr_distribution[label]["占比(%)"] = round(count / len(dbp_all) * 100, 2)
        hr_distribution["所有"]["频数"] = len(dbp_all)
        hr_distribution["所有"]["占比(%)"] = round(1 * 100, 2)
    else:
        hr_distribution["提示"] = "无有效心率数据"
    # ==== DBP分布统计（临床分级） ====
    dbp_distribution = defaultdict(dict)
    if len(dbp_all) > 0:
        dbp_counts, _ = np.histogram(dbp_all, bins=DBP_BINS)
        for i, (label, count) in enumerate(zip(DBP_LABELS, dbp_counts)):
            dbp_distribution[label]["频数"] = count
            dbp_distribution[label]["占比(%)"] = round(count / len(dbp_all) * 100, 2)
        dbp_distribution["所有"]["频数"] = len(dbp_all)
        dbp_distribution["所有"]["占比(%)"] = round(1 * 100, 2)
    else:
        dbp_distribution["提示"] = "无有效DBP数据"

    # ==== 整合最终输出信息 ====
    ordered_info = OrderedDict()
    # 基础信息
    ordered_info["总人数, n"] = info["subjects"]
    # ordered_info["有有效信息的人数, n"] = num_valid_subjects
    # ordered_info["血压记录总数, n"] = info["BP_recordings"]
    ordered_info["有效SBP样本数, n"] = len(sbp_all)
    ordered_info["有效DBP样本数, n"] = len(dbp_all)
    # 年龄统计
    ordered_info[
        "年龄（均值±标准差）, years"] = f"{np.round(info['age_mean'], 2)} ± {np.round(info['age_std'], 2)}" if not np.isnan(
        info["age_mean"]) else "无有效数据"
    ordered_info["最小年龄, years"] = int(info["age_min"]) if not np.isnan(info["age_min"]) else "无有效数据"
    ordered_info["最大年龄, years"] = int(info["age_max"]) if not np.isnan(info["age_max"]) else "无有效数据"
    # BMI统计
    ordered_info[
        "BMI（均值±标准差）"] = f"{np.round(info['bmi_mean'], 2)} ± {np.round(info['bmi_std'], 2)}" if not np.isnan(
        info["bmi_mean"]) else "无有效数据"
    ordered_info["最小BMI"] = round(info["bmi_min"], 2) if not np.isnan(info["bmi_min"]) else "无有效数据"
    ordered_info["最大BMI"] = round(info["bmi_max"], 2) if not np.isnan(info["bmi_max"]) else "无有效数据"
    # 性别比例
    if len(gender_all)>0:

        ordered_info["男性比例, %"] = round(np.sum(gender_all==1) / len(gender_all) * 100, 2)
        ordered_info["女性比例, %"] =round(np.sum(gender_all==2) / len(gender_all) * 100, 2)
    # 心率
    ordered_info[
        "心率（均值±标准差）, bpm"] = f"{np.round(info['hr_mean'], 2)} ± {np.round(info['hr_std'], 2)}" if not np.isnan(
        info["hr_mean"]) else "无有效数据"
    # 血压统计
    ordered_info["SBP（均值±标准差）, mmHg"] = info["sbp_mean_std"]
    ordered_info["DBP（均值±标准差）, mmHg"] = info["dbp_mean_std"]
    #
    # # 血压分类占比（基于有效血压）
    if len(bp_lvl_all) > 0:
        for bp_i in range(1,6):
            ordered_info[f"BP_Level = {bp_i}, %"] = round(np.sum(bp_lvl_all==bp_i) / len(bp_lvl_all) * 100, 2)
    else:
        for bp_i in range(1,6):
            ordered_info[f"BP_Level = {bp_i}, %"] = 0
    # if len(sbp_all) > 0:
    ordered_info["SBP ≥ 160 mmHg, %"] = round(np.sum(sbp_all>=160) / len(sbp_all) * 100, 2)
    ordered_info["SBP ≥ 140 mmHg, %"] = round(np.sum(sbp_all>=140) / len(sbp_all) * 100, 2)
    ordered_info["SBP ≤ 100 mmHg, %"] = round(np.sum(sbp_all<=100) / len(sbp_all) * 100, 2)

    ordered_info["DBP ≥ 100 mmHg, %"] = round(np.sum(dbp_all>=100) / len(dbp_all) * 100, 2)
    ordered_info["DBP ≥ 85 mmHg, %"] = round(np.sum(dbp_all>=85)/ len(dbp_all) * 100, 2)
    ordered_info["DBP ≤ 60 mmHg, %"] = round(np.sum(dbp_all<=60) / len(dbp_all) * 100, 2)


    # 转换为DataFrame
    df_basic = pd.DataFrame(list(ordered_info.items()), columns=["统计项", "数值"])
    df_age_dist = pd.DataFrame.from_dict(age_distribution, orient="index").reset_index().rename(
        columns={"index": "年龄区间"})
    df_gen_dist = pd.DataFrame.from_dict(gender_distribution, orient="index").reset_index().rename(
        columns={"index": "性别"})
    df_pos_dist = pd.DataFrame.from_dict(pos_distribution, orient="index").reset_index().rename(
        columns={"index": "姿势"})

    df_bmi_dist = pd.DataFrame.from_dict(bmi_distribution, orient="index").reset_index().rename(
        columns={"index": "BMI区间"})
    df_hr_dist = pd.DataFrame.from_dict(hr_distribution, orient="index").reset_index().rename(
        columns={"index": "心率分级"})
    df_sbp_dist = pd.DataFrame.from_dict(sbp_distribution, orient="index").reset_index().rename(
        columns={"index": "SBP分级"})
    df_dbp_dist = pd.DataFrame.from_dict(dbp_distribution, orient="index").reset_index().rename(
        columns={"index": "DBP分级"})
    df_level_dist = pd.DataFrame.from_dict(bp_level_distribution, orient="index").reset_index().rename(
        columns={"index": "BP分级"})
    ppgq_stats = []

    for ch, values in ppgq_all.items():
        values = np.concatenate(values) if len(values) > 0 else np.array([])
        if len(values) == 0:
            continue

        ppgq_stats.append({
            "通道": ch,
            "样本数(n)": len(values),
            "均值": round(np.mean(values), 4),
            "标准差": round(np.std(values), 4),
            "最小值": round(np.min(values), 4),
            "最大值": round(np.max(values), 4),
        })

    df_ppgq_basic = pd.DataFrame(ppgq_stats)

    ppgq_distribution = defaultdict(dict)

    for ch, values in ppgq_all.items():
        values = np.concatenate(values) if len(values) > 0 else np.array([])
        if len(values) == 0:
            continue

        counts, _ = np.histogram(values, bins=PPGQ_BINS)

        for label, count in zip(PPGQ_LABELS, counts):
            ppgq_distribution[(ch, label)] = {
                "频数": count,
                "占比(%)": round(count / len(values) * 100, 2)
            }

    df_ppgq_dist = (
        pd.DataFrame.from_dict(ppgq_distribution, orient="index")
        .reset_index()
        .rename(columns={"index": "通道-区间"})
    )

    print("=" * 80)
    raw_stats = {
        "age": np.array(age_all),
        "bmi": np.array(bmi_all),
        "sbp": sbp_all,
        "dbp": dbp_all,
        "bp_level":bp_lvl_all,
        "hr":hr_all,
        "gender": gender_all,
        "pos": pos_all
    }

    if return_result=="df":
        return (df_basic, df_age_dist, df_gen_dist,df_bmi_dist,
                df_sbp_dist, df_dbp_dist,df_level_dist,
                df_pos_dist,df_hr_dist,df_ppgq_basic,df_ppgq_dist,
                raw_stats)
    else:
        return sbp_all, dbp_all


def split_data(args,data):
    # args.train_size = 0.6
    # args.val_size = 0.2
    # args.test_size = 0.2
    print(args.train_size,args.val_size,args.test_size)
    # folds = range(1, 11)
    data = np.asarray(data)
    folds = np.random.RandomState(args.random_seed).permutation(range(len(data))).astype(int)
    return (data[folds[:int(args.train_size*len(data))]],
            data[folds[int(args.train_size*len(data)):int((args.train_size+args.val_size)*len(data))]],
            data[folds[int((args.train_size+args.val_size)*len(data)):]])


def save_combined_paths(train_paths, val_paths, test_paths, save_path="data_splits_paths.pkl"):
    """
    将训练/验证/测试集路径保存到单个文件

    参数:
        train_paths: 训练集路径列表
        val_paths: 验证集路径列表
        test_paths: 测试集路径列表
        save_path: 保存文件的完整路径
    """
    # 创建保存目录（如果不存在）
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将三个路径列表打包成字典
    split_data = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
        "info": {
            "train_count": len(train_paths),
            "val_count": len(val_paths),
            "test_count": len(test_paths),
            "total": len(train_paths) + len(val_paths) + len(test_paths)
        }
    }

    # 保存到单个文件
    with open(save_path, "wb") as f:
        pickle.dump(split_data, f)

    print(f"已将所有路径保存到 {save_path}")
    print(f"训练集: {len(train_paths)} 个文件")
    print(f"验证集: {len(val_paths)} 个文件")
    print(f"测试集: {len(test_paths)} 个文件")
    print(f"总文件数: {split_data['info']['total']}")


def get_combined_stats(split_dict):
    """
    split_dict: {'全量数据': allpaths, '训练集': train_paths, '验证集': val_paths, '测试集': test_paths}
    """
    final_dfs = {}

    # ===== 1. 原有分布维度（不动） =====
    dimensions = {
        "SBP": ("SBP等级", SBP_BINS, SBP_LABELS),
        "DBP": ("DBP等级", DBP_BINS, DBP_LABELS),
        "BP_Level": ("血压等级", BP_LEVEL_LABELS, BP_LEVEL_LABELS),

        "HR": ("HR等级", DBP_BINS, DBP_LABELS),
         "Gender": ("性别", GENDER_BINS, GENDER_LABELS),
        "Age": ("年龄区间", AGE_BINS, AGE_LABELS),
        "BMI": ("BMI分类", BMI_BINS, BMI_LABELS),
        "POS": ("姿势", POS_BINS, POS_LABELS)
    }

    results = {dim: [] for dim in dimensions}
    basic_results = []

    # ===== 2. 新增：PPG Quality 容器 =====
    ppgq_basic_results = []   # 各 split 的 df_ppgq_basic
    ppgq_dist_results = []    # 各 split 的 df_ppgq_dist
    split_raw_dict = {}



    for name, paths in split_dict.items():
        print(f"正在分析 {name} (样本量: {len(paths)})...")

        (
            df_b,
            df_age,
            df_gen,
            df_bmi,
            df_sbp,
            df_dbp,
            df_level,
            df_pos,
            df_hr,
            df_ppgq_basic,
            df_ppgq_dist,raw_stats
        ) = get_info(paths)
        split_raw_dict[name] = raw_stats

        # ===== 3. 基础统计（原逻辑）=====
        df_b.columns = ["指标", name]
        basic_results.append(df_b.set_index("指标"))

        # ===== 4. 各维度分布（原逻辑）=====
        curr_dist = {
            "SBP": df_sbp,
            "DBP": df_dbp,
            "BP_Level":df_level,
            "HR":df_hr,
            "Age": df_age,
            "BMI": df_bmi,
            "POS": df_pos,
            "Gender":df_gen

        }
        for dim, df in curr_dist.items():
            label_col = dimensions[dim][0]

            # 重命名列
            df.columns = [label_col, "n", "%"]

            # 合并 n 和 %
            df[f"{name}"] = df["n"].astype(str) + " (" + df["%"].astype(str) + "%)"

            # 只保留合并后的列
            df_merged = df[[label_col, f"{name}"]]

            results[dim].append(df_merged.set_index(label_col))

        # ===== 5. PPG Quality：基础统计 =====
        df_ppgq_basic_copy = df_ppgq_basic.copy()
        df_ppgq_basic_copy["数据集"] = name
        ppgq_basic_results.append(df_ppgq_basic_copy)

        # ===== 6. PPG Quality：分布统计 =====
        df_ppgq_dist_copy = df_ppgq_dist.copy()
        df_ppgq_dist_copy["数据集"] = name
        ppgq_dist_results.append(df_ppgq_dist_copy)

    # ===== 7. 横向合并 =====

    # 7.1 基础统计
    df_basic_final = pd.concat(basic_results, axis=1).reset_index()
    final_dfs["基础统计"] = df_basic_final

    # 7.2 原有分布统计
    for dim in dimensions:
        sheet_name = f"{dim}分布对比"
        final_dfs[sheet_name] = pd.concat(results[dim], axis=1).reset_index()

    # # # 7.3 PPG Quality：基础统计（推荐用于 Table）
    final_dfs["PPG_Quality_基础统计"] = pd.concat(
        ppgq_basic_results, axis=0
    ).reset_index(drop=True)

    # 7.4 PPG Quality：分布统计（推荐用于 Supplementary）
    final_dfs["PPG_Quality_分布统计"] = pd.concat(
        ppgq_dist_results, axis=0
    ).reset_index(drop=True)
    df_pvals = split_difference_tests(split_raw_dict)

    return final_dfs,df_pvals
def split_comparability_analysis(stats_dict, save_path):
    """
    stats_dict: 你已有的统计结果字典
    save_path: 输出 Excel 路径
    """

    results = []

    # ========= 连续变量 =========
    continuous_vars = ["age", "bmi", "sbp", "dbp", "hr"]

    for var in continuous_vars:
        split_values = {
            "train": np.array(stats_dict["训练集"][var]),
            "val":   np.array(stats_dict["验证集"][var]),
            "test":  np.array(stats_dict["测试集"][var])
        }

        p_value = kruskal_test_across_splits(split_values)

        results.append({
            "Variable": var.upper(),
            "Type": "Continuous",
            "Test": "Kruskal–Wallis",
            "p-value": p_value
        })

    # ========= 分类变量 =========
    categorical_vars = {
        "gender": "Gender",
        "pos": "Posture",
        "bp_level": "BP Level"
    }

    for key, name in categorical_vars.items():
        count_df = pd.DataFrame({
            "train": stats_dict["训练集"][key],
            "val":   stats_dict["验证集"][key],
            "test":  stats_dict["测试集"][key]
        })

        p_value = chi_square_test_across_splits(count_df)

        results.append({
            "Variable": name,
            "Type": "Categorical",
            "Test": "Chi-square",
            "p-value": p_value
        })

    # ========= 保存 =========
    results_df = pd.DataFrame(results)
    return results_df
    # results_df.to_excel(save_path, index=False)
    #
    # print(f"[✓] Split comparability statistics saved to: {save_path}")
def split_difference_tests(split_raw_dict):
    results = []

    continuous_vars = {
        "Age": "age",
        "BMI": "bmi",
        "SBP": "sbp",
        "DBP": "dbp",
        "HR": "hr",
    }

    for name, key in continuous_vars.items():
        arrays = [v[key] for v in split_raw_dict.values()]
        stat, p = kruskal(*arrays)
        results.append({
            "Variable": name,
            "Type": "Continuous",
            "Test": "Kruskal–Wallis",
            "p-value": round(p, 4)
        })
    # ========= 分类变量 =========
    categorical_vars = {
        "Gender":"gender",
        "Posture":"pos",
        # "BP_Level":"bp_level",
    }
    # 分类变量：性别 & 姿势
    for name, key in categorical_vars.items():
        table = []
        for split in split_raw_dict:
            # a = np.float32(split_raw_dict[split][key])
            # values, counts = np.unique(
            #     a, return_counts=True
            # )
            # # print(counts)
            # if counts.ndim == 1:
            #
            if key == "gender":
                counts = np.zeros(2)
                counts[0] = np.sum(split_raw_dict[split][key]==1)
                counts[1] = np.sum(split_raw_dict[split][key] == 2)
            if key =="pos":
                counts = np.zeros(3)
                counts[0] = np.sum(split_raw_dict[split][key]=="lay")
                counts[1] = np.sum(split_raw_dict[split][key] == "sit")
                counts[2] = np.sum(split_raw_dict[split][key] == "stand")

            table.append(counts)

        chi2, p, _, _ = chi2_contingency(table)
        results.append({
            "Variable": name,
            "Type": "Categorical",
            "Test": "Chi-square",
            "p-value": round(p, 4)
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    dir_path = "D:\研究课题\datasets//Ring2Health/10_second/"
    # Attention!! The real file dir path
    result_dir = "../results/"

    allpaths = glob.glob(os.path.join(dir_path, "*.pkl"))
    save_split_path = "data_splits_paths.pkl"
    data_path_update = True
    if os.path.exists(save_split_path) and data_path_update is False:
        data = pd.read_pickle(save_split_path)
        train_paths, test_paths, val_paths = data['train'], data['test'], data['val']
    else:
        train_paths, val_paths, test_paths = split_data(args, allpaths)
        save_combined_paths(train_paths, val_paths, test_paths,save_path=save_split_path)

    # 2. 构造分集字典
    split_dict = {
        "全量数据": allpaths,
        "训练集": train_paths,
        "验证集": val_paths,
        "测试集": test_paths
    }

    #3. 计算所有统计结果（已实现横向合并）
    all_combined_results,df_pvals = get_combined_stats(split_dict)

    # 4. 保存到 Excel
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    save_path = os.path.join(result_dir, f"数据集划分分析报告_{formatted_date}.xlsx")

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        df_pvals.to_excel(writer, sheet_name="Split_Difference_Test", index=False)
        for sheet_name, df in all_combined_results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ 汇总报告已生成，包含全量及各子集对比：{save_path}")
