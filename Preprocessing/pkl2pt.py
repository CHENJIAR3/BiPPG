import numpy as np
import pandas as pd
import os
import torch
import gc
from utils import args
import lmdb
import pickle
import os
import gc
import numpy as np
import pandas as pd
import torch

# Position映射规则
POSITION_MAPPING = {
    "lay": 1,
    "sit": 2,
    "stand": 3
}
def pkl2pt_stream(filenames, save_path, args,  chunk_size=1000):
    """
    内存优化版：直接把多个pkl文件转为一个.pt文件，避免一次性加载全部数据
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    ppg_list = []
    label_list = []

    total = 0
    for i, file_path in enumerate(filenames):
        # 分块读取 pkl
        data = pd.read_pickle(file_path)
        # data = data.dropna(subset=args.label_key)
        # 映射替换（不在映射表中的值保留原样并统计）
        data["position"] = data["position"].map(
            lambda x: POSITION_MAPPING[x] if x in POSITION_MAPPING else np.nan
        )
        #  or "all" in args.pt_type
        if "HQ" in args.pt_type:
            # print("HQ")
            # 左手
            pr_est_cols_1 = [col for col in data.keys() if "pr_est" in col and "1" in col]
            pr_est_cols_2 = [col for col in data.keys() if "pr_est" in col and "2" in col]
            pr1 = np.concatenate(
                [np.stack(data[key].values) for key in pr_est_cols_1],
                axis=1
            ).astype(np.float32)
            pr2 = np.concatenate(
                [np.stack(data[key].values) for key in pr_est_cols_2],
                axis=1
            ).astype(np.float32)
            data["pr_mean_1"] = pr1.mean(axis=1)
            data["pr_std_1"] = pr1.std(axis=1)
            data["pr_mean_2"] = pr2.mean(axis=1)
            data["pr_std_2"] = pr2.std(axis=1)
            data["pr_est_range"] = pr1.max(axis=1) - pr1.min(axis=1)
            data = data[(abs(data["pr_ref"] - data["pr_mean_1"]) <= 5) &
                        (abs(data["pr_ref"] - data["pr_mean_2"]) <= 5) &
                        (abs(data["pr_mean_1"] - data["pr_mean_2"]) <= 2) ]
            # pr_est = np.mean(np.concatenate([np.stack(data[key].values) for key in pr_est_cols], axis=1).astype(
            #     np.float32), axis=1)
            # data["pr_est"] = pr_est
            # data["delta_hr"] = abs(data["pr_ref"] - data["pr_est"])
            # data = data[data["delta_hr"] <= 5]
        if len(data) == 0:
            continue
        # 类型压缩
        # for col in data.select_dtypes(include=['float64']).columns:
        #     data[col] = pd.to_numeric(data[col], downcast="float")
        # for col in data.select_dtypes(include=['int64']).columns:
        #     data[col] = pd.to_numeric(data[col], downcast="integer")

        # 转 numpy，降低内存峰值
        # ppg_key = args.ppg_key
        # if "DC" in args.pt_type:
        #     ppg_key = args.ppg_key + args.dc_key

        #     dc_data = np.concatenate([np.stack(data[key].values)[:,None] for key in args.dc_key], axis=1)[:,:,None].astype(np.float32)
        #     mode = args.pt_type.split("_")[1]
        #     if mode == "add":
        #         ppg_data = ppg_data + dc_data
        #     if mode == "divide":
        #         ppg_data = ppg_data / dc_data
        # else:

        # 提取 PPG 和 Label (保持 float32)
        # if args.pt_type == "all":
        #     ppg_key = args.ppg_key + args.dc_key
        # elif args.pt_type == "DC":
        #     ppg_key =  args.dc_key
        # else:
        ppg_key = args.ppg_key
        # print(len(ppg_key))
        # ppg_key = args.ppg_key + args.dc_key if "DC" in args.pt_type  or args.pt_type == "all" else args.ppg_key
        # ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in ppg_key], axis=1).astype(np.float32)
        # print(ppg_data.shape)

        ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in ppg_key], axis=1).astype(
                np.float32)
        # ppg_data = ppg_data/10000
        # print(ppg_data.shape)
        label_data = np.concatenate([np.stack(data[key].values)[:,None] for key in args.label_key], axis=1).astype(np.float32)

        ppg_list.append(torch.from_numpy(ppg_data))
        label_list.append(torch.from_numpy(label_data))
        total += len(data)

        # 每处理一定数量，先写磁盘，释放内存
        if (i + 1) % 10 == 0 or (i + 1) == len(filenames):
            print(f"已处理 {i+1}/{len(filenames)} 个文件，总样本 {total}")
            # 累积写入
            if os.path.exists(save_path):
                prev = torch.load(save_path)
                prev_ppg = torch.cat([prev['ppg']] + ppg_list, dim=0)
                prev_label = torch.cat([prev['label']] + label_list, dim=0)
            else:
                prev_ppg = torch.cat(ppg_list, dim=0)
                prev_label = torch.cat(label_list, dim=0)

            torch.save({'ppg': prev_ppg, 'label': prev_label}, save_path, pickle_protocol=5)
            # torch.save({'ppg': prev_ppg, }, ptname + ".pt", pickle_protocol=5)

            # 清空临时缓存
            ppg_list.clear()
            label_list.clear()
            gc.collect()

    print(f"最终完成: 保存 {total} 条样本到 {save_path}")




def pkl2lmdb_stream(filenames, save_path, args, map_size=1024 ** 4):  # 默认 1TB 容量限制
    """
    将多个 pkl 文件转为 LMDB 数据库
    save_path: LMDB 文件夹路径
    """
    os.makedirs(save_path, exist_ok=True)
    # os.makedirs()
    # 初始化 LMDB 环境
    env = lmdb.open(save_path, map_size=map_size)

    total_samples = 0

    for i, file_path in enumerate(filenames):
        data = pd.read_pickle(file_path)

        # --- 保持你原有的预处理逻辑 ---
        data["position"] = data["position"].map(
            lambda x: POSITION_MAPPING[x] if x in POSITION_MAPPING else np.nan
        )

        if "HQ" in args.pt_type or args.pt_type == "all":
            pr_est_cols = [col for col in data.keys() if "pr_est" in col]
            pr_stack = np.concatenate([np.stack(data[key].values) for key in pr_est_cols], axis=1).astype(np.float32)
            data["pr_est_mean"] = pr_stack.mean(axis=1)
            data["pr_est_std"] = pr_stack.std(axis=1)
            data = data[(abs(data["pr_ref"] - data["pr_est_mean"]) <= 5) & (data["pr_est_std"] <= 3)]
            # data = data[(data["position"]== 1)]

        if len(data) == 0:
            continue

        # 提取 PPG 和 Label (保持 float32)
        if args.pt_type == "all":
            ppg_key = args.ppg_key + args.dc_key
        elif args.pt_type == "DC":
            ppg_key =  args.dc_key
        else:
            ppg_key = args.ppg_key
        print(len(ppg_key))
        # ppg_key = args.ppg_key + args.dc_key if "DC" in args.pt_type  or args.pt_type == "all" else args.ppg_key
        ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in ppg_key], axis=1).astype(np.float32)
        print(ppg_data.shape)
        label_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.label_key], axis=1).astype(
            np.float32)

        # --- 写入 LMDB ---
        with env.begin(write=True) as txn:
            for j in range(len(data)):
                # 将每一对 ppg 和 label 序列化
                sample = {
                    'ppg': ppg_data[j],
                    'label': label_data[j]
                }
                # 键名使用 8 位补零数字，方便排序和检索
                str_id = f'{total_samples:08d}'
                txn.put(str_id.encode('ascii'), pickle.dumps(sample))
                total_samples += 1


        # 显式清理
        del data, ppg_data, label_data
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(filenames)}，累计样本: {total_samples}")
            gc.collect()

    # 存储总样本数，方便 Dataset 调用
    with env.begin(write=True) as txn:
        txn.put(b'__len__', str(total_samples).encode('ascii'))

    env.close()
    print(f"LMDB 制作完成，总样本数: {total_samples}")

def get_ppgkey(args):
    if 'original' in args.pt_type:
        args.ppg_key = ['ppg_g_1',  'ppg_ga_1','ppg_r_1','ppg_ir_1',
                    'ppg_g_2',  'ppg_ga_2', 'ppg_r_2','ppg_ir_2',] # 'ppg_r_1','ppg_ir_1', 'ppg_r_2','ppg_ir_2',

    if "filter" in args.pt_type:
        args.ppg_key = ['ppg_g_filter_1', 'ppg_ga_filter_1','ppg_r_filter_1', 'ppg_ir_filter_1',
               'ppg_g_filter_2', 'ppg_ga_filter_2', 'ppg_r_filter_2', 'ppg_ir_filter_2',] # 'ppg_r_filter_1', 'ppg_ir_filter_1', 'ppg_r_filter_2', 'ppg_ir_filter_2',
    return args
if __name__ == '__main__':
    pkl_file = "/home/cjr/PPGBen/Preprocessing/data_splits_paths.pkl"
    save_dir = "/home/cjr/datasets/Ring2Health/dataset/"
    data = pd.read_pickle(pkl_file)
    train_paths,test_paths,val_paths = data['train'],data['test'],data['val']
    # pt_types = ["all","AC","DC"]
    pt_types = ["all"]
    # pt_types = ["HQ"]
    # all = original + filter
    # pt_types = []
    # "denoiser","original","original_DC", "filter","filter_DC",
    args.ppg_key = ['ppg_g_1','ppg_g_filter_1', 'ppg_ga_1','ppg_ga_filter_1','ppg_r_1', 'ppg_r_filter_1', 'ppg_ir_1','ppg_ir_filter_1',
                    'ppg_g_2','ppg_g_filter_2', 'ppg_ga_2','ppg_ga_filter_2','ppg_r_2', 'ppg_r_filter_2', 'ppg_ir_2','ppg_ir_filter_2',]
    # args.ppg_key = ['ppg_g_1', 'ppg_ga_1','ppg_r_1','ppg_ir_1',
    #                 'ppg_g_2',  'ppg_ga_2',  'ppg_r_2','ppg_ir_2' ]
    # args.ppg_key = ['ppg_g_filter_1', 'ppg_ga_filter_1', 'ppg_r_filter_1', 'ppg_ir_filter_1',
    #                 'ppg_g_filter_2', 'ppg_ga_filter_2', 'ppg_r_filter_2', 'ppg_ir_filter_2', ]
    # args.dc_key = ['ppg_g_DC_1', 'ppg_ga_DC_1','ppg_r_DC_1', 'ppg_ir_DC_1',
    #                     'ppg_g_DC_2', 'ppg_ga_DC_2','ppg_r_DC_2', 'ppg_ir_DC_2',] #'ppg_r_DC_1', 'ppg_ir_DC_1',, 'ppg_r_DC_2', 'ppg_ir_DC_2',
    args.label_key  = ["sbp_fix","dbp_fix","pr_ref","age"]
    # ,"bmi",'gender',"position"
    data_names = ["train", "test", "val"]
    # "BP_Level",
    # win_len = 10
    for pt_type in pt_types:
        args.pt_type = pt_type
        # args = get_ppgkey(args)
        for i,paths in enumerate([train_paths, test_paths, val_paths]):
            print(i,len(paths))
            # +"_"+ args.pt_type
            # pt_type + "_" +
            save_path = os.path.join(save_dir,pt_type + "_" +data_names[i])
            pkl2pt_stream(paths,save_path,args)
            # lmdb是否是一种更牛逼的dataloader
            # pkl2lmdb_stream(paths,save_path,args)