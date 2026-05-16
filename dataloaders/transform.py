import torch
import numpy as np

# 定义一个函数，检查给定行中所有指定列的列表中是否包含 nan
def has_nan_in_columns(row,columns_to_check):
    for col in columns_to_check:
        # 检查列表内部是否有 nan
        if np.isnan(row[col]).any():
            return True
    return False
def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, 1,X.shape[-1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1,1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[-1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig
def normalize(ppg, norm_method, C, epsilon=1e-8):
    # Calculate common statistics
    min_v = torch.min(ppg, dim=2, keepdim=True)[0]
    max_v = torch.max(ppg, dim=2, keepdim=True)[0]
    mean = torch.mean(ppg, dim=2, keepdim=True)
    std = torch.std(ppg, dim=2, keepdim=True)
    rms = torch.sqrt(torch.mean(ppg ** 2, dim=2, keepdim=True))

    # Normalize the data based on the method
    if norm_method == "maxmin":
        ppg_norm = (ppg - min_v) / (max_v - min_v + epsilon)
        fea = torch.cat([max_v[:, :C // 2, 0], min_v[:, :C // 2, 0],
                         max_v[:, C // 2:, 0], min_v[:, C // 2:, 0]], dim=1)
    elif norm_method == "z-score":
        ppg_norm = (ppg - mean) / (std + epsilon)
        fea = torch.cat([mean[:, :C // 2, 0], std[:, :C // 2, 0],
                         mean[:, C // 2:, 0], std[:, C // 2:, 0]], dim=1)
    elif norm_method == "rms":
        ppg_norm = ppg / (rms + epsilon)
        fea = torch.cat([rms[:, :C // 2, 0], rms[:, :C // 2, 0],
                         rms[:, C // 2:, 0], rms[:, C // 2:, 0]], dim=1)
    return ppg_norm, fea