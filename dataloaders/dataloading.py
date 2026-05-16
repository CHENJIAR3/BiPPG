
import pandas as pd
import torch
import numpy as np
##2025-07-01 做完了数据集！
from torch.utils.data import DataLoader,  random_split
from torch.utils.data import Dataset
import lmdb
import pickle
from dataloaders.transform import *

import shutil
class myDataset(Dataset):
    def __init__(self, data,args):
        self.ppg = []
        self.label = []
        self.samples = []
        ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.ppg_key], axis=1)
        self.ppg = ppg_data
        label_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.label_key], axis=1)
        self.label= label_data

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return self.ppg[idx],self.label[idx]

class PickleDataset_all(Dataset):
    def __init__(self, file_paths,args,denoised_dir=None):
        self.data = []
        self.label = []
        for i,file_path in enumerate(file_paths):
            if denoised_dir is not None:
                file_path = denoised_dir + "/" + file_path.split("/")[-1]

            data = pd.read_pickle(file_path)
            data = data.dropna(subset=args.label_key)
            # 使用 apply() 在每一行上应用函数，创建布尔掩码
            mask = data.apply(has_nan_in_columns, axis=1,columns_to_check=args.ppg_key)
            # 使用 ~ 取反，筛选出不含 nan 的行
            data = data[~mask].copy()
            if len(data) == 0:
                continue
            ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.ppg_key], axis=1)
            self.data.append(ppg_data)
            label_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.label_key], axis=1)
            self.label.append(label_data)
        self.data = np.concatenate(self.data, axis=0)
        self.data = torch.from_numpy(self.data).float()
        self.data, self.fea = normalize(self.data, args.norm_method, C=self.data.shape[-1])
        self.label = np.concatenate(self.label, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.fea[idx],self.label[idx]

class PickleDataset(Dataset):
    def __init__(self, filename, args):
        self.ppg = []
        self.label = []
        self.samples = []
        data =  pd.read_pickle(filename)
        ppg_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.ppg_key], axis=1)
        self.ppg = ppg_data
        label_data = np.concatenate([np.stack(data[key].values)[:, None] for key in args.label_key], axis=1)
        self.label= label_data

        # torch.save({'ppg': ppg_data, 'label': label_data}, 'train_processed.pt')

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        return self.ppg[idx],self.label[idx]
# 在Dataset中加载




class TorchDataset(Dataset):
    def __init__(self, filename, args,epsilon=1e-3):
        data = torch.load(filename)
        self.index = args.index

        ppg = data['ppg']  # 假设形状为 (N, C, T)，其中N是样本数，C是通道数，T是时间步
        ppg = ppg[:,self.index]
        self.fea = torch.zeros((ppg.shape[0],2*ppg.shape[1]))
        # self.fea = data["quality"]
        # self.fea = data["ppg"]
        # self.fea = self.fea[:,self.index]
        # 关键判断：是否为NumPy数组
        if isinstance(ppg, np.ndarray):
            # 转换为Tensor（自动兼容常见数据类型：float32、int64等）
            ppg = torch.from_numpy(ppg).to(torch.float32)

        if args.norm_method is None:
            self.ppg = ppg
        else:
            B, C, L = ppg.shape

            if C == 8:
                self.ppg, self.fea = normalize(ppg, args.norm_method, C)
            else:
                self.ppg, self.fea = normalize(ppg, args.norm_method, C)


        self.only_data = args.only_data
        if args.only_data is False:
            self.label_index = args.label_index
            label =  data['label']  # 形状为 (N, 2)
            if label.ndim==3:
                label = torch.squeeze(label)
            if label.ndim==1:
                label = label[:,None]

            self.label = label
        self.epsilon = epsilon


    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        # 提取当前样本的ppg（按index取通道）
        ppg_sample = self.ppg[idx]  # 假设self.index是要选取的通道索引，形状为 (T,) 或 (C_selected, T)
        fea_sample = self.fea[idx]
        if self.only_data:
            return ppg_sample
        else:
            # ,self.label_index
            label_sample = self.label[idx]
            return ppg_sample,fea_sample, label_sample


def get_dataloader(data_dir,args,split='train'):
    # pklname = data_dir+split+'.pkl'
    # data_set = PickleDataset(pklname, args)
    # ptname = data_dir+split
    # +'.pt'
    # if 'train' in split:
    data_set = TorchDataset(data_dir+split, args)
    data_loader = DataLoader(data_set, batch_size=args.bs,
                              generator=torch.Generator().manual_seed(args.random_seed),  # ✅ 固定shuffle顺序
                              num_workers=args.num_workers, shuffle=False,
                            pin_memory=True,  # 核心：锁页内存，避免CPU内存二次拷贝，加速CPU→GPU传输
                            persistent_workers=True,  # 核心：保持子进程存活，避免每个epoch重建，减少CPU开销
                             prefetch_factor=4,
                             # collate_fn=ppg_collate_fn  # 绑定预处理函数
                             )
    # else:
    #     data_set = TorchDataset(ptname, args)
    #     data_loader = DataLoader(data_set, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, pin_memory=True, prefetch_factor=4)
    return data_loader




def preprocess_to_mmap(pth_file, save_dir):
    data = torch.load(pth_file)
    # 将主要的 PPG 数据存为 .npy
    ppg_np = data['ppg'].numpy().astype(np.float32)
    np.save(f"{save_dir}/ppg_data.npy", ppg_np)


