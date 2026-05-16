# 2026-01-29
# 得到一个大的predictions！
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import datetime
formatted_date = datetime.datetime.now().strftime("%Y-%m-%d")

from tqdm import tqdm
from models.model_loading import load_model
from evaluation.model_metric import get_performance
from utils import args
POSITION_MAPPING = {
    "lay": 1,
    "sit": 2,
    "stand": 3
}
import gc  # 垃圾回收，释放中间内存
from dataloaders.transform import normalize
def batched_forward(model, x, bs, device):

    outputs = []

    for i in range(0, x.shape[0], bs):

        batch = x[i:i+bs].to(device, dtype=torch.float, non_blocking=True)

        with torch.inference_mode():
            out = model(batch)

        outputs.append(out.cpu())

    return torch.cat(outputs, dim=0)

class PPGDataset(Dataset):
    """PPG生理数据的PyTorch自定义数据集类"""

    def __init__(self,data,args):
        ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.ppg_key], axis=1).astype(
            np.float32)
        dc_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.dc_key], axis=1).astype(
            np.float32)
        eps = 1e-6
        C = ppg_data.shape[1]
        self.ppg,_ = normalize(ppg_data, args.norm_method,C//2)
        self.label_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.label_key], axis=1).astype(
            np.float32)
    def __len__(self):
        """返回数据集总样本数（Dataset必须实现）"""
        return self.n_samples

    def __getitem__(self, idx):
        """
        按索引返回单样本数据（Dataset必须实现）
        :param idx: 样本索引（int）
        :return: dict 包含预处理后的所有特征和标签，值为torch.Tensor
        """
        # 按索引取数，并转换为PyTorch张量（float32类型，与原数据一致）
        # self.fea[idx],
        return self.ppg[idx],self.label_data[idx]

def get_model_input(data):
    data["position"] = data["position"].map(
        lambda x: POSITION_MAPPING[x] if x in POSITION_MAPPING else np.nan
    )
    ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.ppg_key], axis=1).astype(
        np.float32)
    dc_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.dc_key], axis=1).astype(
        np.float32)
    eps = 1e-6
    C = ppg_data.shape[1]
    ppg_data = torch.from_numpy(ppg_data )
    ppg_data, _ = normalize(ppg_data, args.norm_method, C // 2)
    return ppg_data

def get_model_output(pred_left,pred_right):

    pred_dict = pd.DataFrame({"pred_sbp": pred_left[:,0].cpu().detach().numpy(),
                              "pred_dbp":  pred_left[:,1].cpu().detach().numpy(),
                              "pred_hr":  pred_left[:,2].cpu().detach().numpy(),
                              "pred_age": pred_left[:,3].cpu().detach().numpy(),
                              "pred_sbp_right": pred_right[:, 0].cpu().detach().numpy(),
                              "pred_dbp_right": pred_right[:, 1].cpu().detach().numpy(),
                              "pred_hr_right": pred_right[:, 2].cpu().detach().numpy(),
                              "pred_age_right": pred_right[:, 3].cpu().detach().numpy(),
                              })

    return pred_dict




# ===================== 提前封装工具函数（减少循环内代码）=====================
def extract_subject_id(path):
    """提取subject_id，封装为函数，减少循环内重复代码"""
    return path.split("/")[-1][:-4]



def pred_bymodel(test_paths,model,args):
    results = []
    ppg_cols = None
    C = len(args.ppg_key)//2
    for test_path in tqdm(test_paths):
        test_path = data_dir + test_path.split("/")[-1]
        data = pd.read_pickle(test_path)
        data_input = get_model_input(data)
        if ppg_cols is None:
            ppg_cols = [col for col in data.columns if "ppg" in col]
            keep_cols = [col for col in data.columns if col not in ppg_cols]
        with torch.inference_mode():
            x = torch.cat([data_input[:, :C], data_input[:, C:]], dim=0)
            x = x.to(args.device, dtype=torch.float, non_blocking=True)
            # pred = model(x)
            pred = batched_forward(
                model,
                x,
                bs=args.bs,
                device=args.device
            )
            pred1, pred2 = pred.chunk(2, dim=0)
        data_dict = data[keep_cols]
        pred_dict = get_model_output(pred1,pred2)
        result_df = pd.concat([data_dict,pred_dict], axis=1, ignore_index=False)
        # 释放内存（避免循环内内存累积）
        del data, data_input, pred1,pred2, data_dict, pred_dict
        # 直接插入subject_id到第一列（最简洁高效）
        subject_id = extract_subject_id(test_path)
        result_df["subject_id"] = subject_id
        results.append(result_df)
        # 构c
    results_df = pd.concat(results)
    cols = ["subject_id"] + [c for c in results_df.columns if c != "subject_id"]
    results_df = results_df[cols]
    return results_df
def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
def pred_bymodel_with_quality(test_paths, model, args, data_dir=""):
    """
    对每个测试样本进行预测，同时保留原始 pr_est 和 ppg_quality 列表，并添加均值列
    """
    results = []
    ppg_cols = None
    C = len(args.ppg_key) // 2  # 假设左右手通道各半

    for test_path in tqdm(test_paths):
        test_path_full = data_dir + test_path.split("/")[-1]
        data = pd.read_pickle(test_path_full)

        # 模型输入
        data_input = get_model_input(data)

        # 找到 PPG 列和其他列
        if ppg_cols is None:
            quality_cols = [k for k in data.columns if "quality" in k]
            ppg_cols = [col for col in data.columns if "ppg" in col]
            keep_cols = [col for col in data.columns if col not in ppg_cols]

        with torch.inference_mode():
            x = torch.cat([data_input[:, :C], data_input[:, C:]], dim=0)
            x = x.to(args.device, dtype=torch.float, non_blocking=True)
            pred = batched_forward(model, x, bs=args.bs, device=args.device)
            pred1, pred2 = pred.chunk(2, dim=0)

        # 获取预测输出
        pred_dict = get_model_output(pred1, pred2)  # 包含 pr_est 列
        # 生成 pr_est 均值列
        for key in pred_dict.keys():
            if "pr_est" in key:
                pred_dict[f"{key}_mean"] = pred_dict[key].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

        # 同样处理 quality 列，生成均值
        for qkey in quality_cols:
            pred_dict[f"{qkey}_mean"] = data[qkey].apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)

        # 拼接其他原始列
        data_dict = data[keep_cols]
        result_df = pd.concat([data_dict, pred_dict], axis=1, ignore_index=False)

        # 添加 subject_id
        subject_id = extract_subject_id(test_path)
        result_df["subject_id"] = subject_id

        results.append(result_df)

        # 释放内存
        del data, data_input, pred1, pred2, data_dict, pred_dict

    # 合并所有样本
    results_df = pd.concat(results, ignore_index=True)
    # 调整列顺序
    cols = ["subject_id"] + [c for c in results_df.columns if c != "subject_id"]
    results_df = results_df[cols]

    return results_df
#     #
if __name__ == '__main__':
    # args.ppg_key = ['ppg_g_1','ppg_g_filter_1', 'ppg_ga_1','ppg_ga_filter_1','ppg_r_1', 'ppg_r_filter_1', 'ppg_ir_1','ppg_ir_filter_1',
    #                 'ppg_g_2','ppg_g_filter_2', 'ppg_ga_2','ppg_ga_filter_2','ppg_r_2', 'ppg_r_filter_2', 'ppg_ir_2','ppg_ir_filter_2',]
    args.ppg_key = ['ppg_g_filter_1',  'ppg_ga_filter_1',  'ppg_r_filter_1','ppg_ir_filter_1',
                    'ppg_g_filter_2','ppg_ga_filter_2', 'ppg_r_filter_2', 'ppg_ir_filter_2', ]

    args.norm_method = "z-score"
    args.device = "cuda:0"
    args.in_channels = len(args.ppg_key)//2
    # save_dir = "/home/cjr/datasets/Ring2Health/dataset/"
    pkl_file = "../Preprocessing/data_splits_paths.pkl"
    pkl_file = "../Preprocessing/data_splits_paths_nobpfilter.pkl"

    data_dir = r"D:\研究课题\datasets\Ring2Health\10_second/"
    data_dir = r"/home/cjr/datasets/Ring2Health/10_second_nobpfilter/"

    data = pd.read_pickle(pkl_file)
    train_paths,test_paths,val_paths = data['train'],data['test'],data['val']
    model_dir = "../saved_models/"
    os.makedirs("../predictions/", exist_ok=True)
    # "CRNN","ACNN","ResNet1D","Net1D","LSTM","Efficient1D","AutoFormer",
    # "PatchTST","Informer","iTransformer",
    # "ResNet1DMoE","CSFM_tiny","AttnRes",
    args.bs = 32
    for model_type in [
        # "CRNN","ACNN","ResNet1D",
        "Net1D",
        # "LSTM","Efficient1D",
        #                "AutoFormer","PatchTST",
        #                "Informer","iTransformer",
        #                "ResNet1DMoE",
        # "CSFM_tiny",
        # "AttnRes",
    ]:
        args.model_type = model_type
        subgroup = args.exp + "_" + args.model_type
        model_paths = glob.glob(model_dir + "*_"+ model_type + "_*")
        args.model_path = model_paths[-1]
        print(args.model_path)
        model = load_model(args)
        model.to(args.device)
        model.load_state_dict(torch.load(args.model_path, map_location=args.device,weights_only=False))
        # 设置为评估模式
        model.eval()
        # results_df = pred_bymodel(test_paths,model,args)
        # results_df.to_pickle(f"../predictions/{args.model_type}_prediction.pkl")
        # results_df.to_pickle(f"../predictions/{args.model_type}_prediction_nobpfilter.pkl")
        results_df = pd.read_pickle(f"../predictions/{args.model_type}_prediction_nobpfilter.pkl")
        # results_df = pred_bymodel_with_quality(test_paths,model,args,data_dir)
        # results_df.to_pickle(f"../predictions/{args.model_type}_prediction_{formatted_date}.pkl")
        targets = results_df[["sbp_fix","dbp_fix","pr_ref","age"]].values
        pred_left = results_df[["pred_sbp","pred_dbp","pred_hr","pred_age"]].values
        pred_right = results_df[["pred_sbp_right","pred_dbp_right","pred_hr_right","pred_age_right"]].values
        result_left = get_performance(pred_left,targets,)
        result_right = get_performance(pred_right,targets,)
        result = get_performance(0.5*(pred_left + pred_right),targets,)
        print(result_left)
        print(result_right)
        print(result)

    # df_all = pd.DataFrame([result_left["regression"],result_right["regression"],result["regression"]],index=["left","right","bi"])
    # df.to_csv("../results/nobpfilter_performance.csv")
    #
