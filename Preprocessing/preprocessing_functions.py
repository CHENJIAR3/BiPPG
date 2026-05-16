# ===== 内置库 =====
import os
import glob
import pywt
import pandas as pd
import json
import hashlib
from datetime import datetime
from collections import Counter, defaultdict
import numpy as np
from scipy import signal
import neurokit2 as nk
from functools import lru_cache

def getDivisor(rf):
    d = np.ones_like(rf)
    idx1 = np.where(rf==5)[0]
    d[idx1]= 100
    idx2= np.where(rf == 2)[0]
    d[idx2] = 10
    return d
finger_map = {
    ("左手", "拇指"): 1,
    ("左手", "食指"): 2,
    ("左手", "中指"): 3,
    ("左手", "无名指"): 4,
    ("左手", "小指"): 5,
    ("右手", "拇指"): 6,
    ("右手", "食指"): 7,
    ("右手", "中指"): 8,
    ("右手", "无名指"): 9,
    ("右手", "小指"): 10,
}

def get_finger(checkcsv):
    csv_config = os.path.basename(checkcsv).split("_")
    ppg_finger = 0
    if len(csv_config) >= 4 and csv_config[-4] == "PPG":
        # has_ppg = True
        hand, finger = csv_config[-3], csv_config[-2]
        # 记录具体手指编号
        if (hand, finger) in finger_map:
            ppg_finger = finger_map[(hand, finger)]
    return ppg_finger
def hash_to_word(name, length=8):
    # 创建哈希对象
    hash_obj = hashlib.sha256(name.encode())
    # 获取十六进制摘要
    hex_digest = hash_obj.hexdigest()
    # 转换为简单单词
    return hex_digest[:length]

def check_packet_loss(timestamps, expected_interval= 10 ):
    # 采样周期为10ms

    lost_packets = []

    # 遍历时间戳列表，检查相邻时间戳的差值
    for i in range(1, len(timestamps)):
        interval = timestamps[i] - timestamps[i - 1]
        if interval > expected_interval:
            lost_packets.append((timestamps[i - 1], timestamps[i], interval))
    if lost_packets:
        print("存在丢包情况:")
        for start, end, interval in lost_packets:
            print(f"从 {start} 到 {end} 丢失了 {interval / 1000.0} 秒数据")

    else:
        pass
    return lost_packets

@lru_cache(maxsize=8)
def butter_bandpass_sos(order=4, low=0.5, high=8, fs=100):
    return signal.butter(
        order, [low, high],
        btype='bandpass',
        fs=fs,
        output='sos'
    )

def band_pass_filter(input_data, order=4, low=0.5, high=8, fs=100):
    sos =  butter_bandpass_sos(order, low, high, fs)
    if input_data.ndim == 1:
        return signal.sosfiltfilt(sos, input_data)
    else:
        return signal.sosfiltfilt(sos, input_data, axis=-1)


    # return (filter_data - np.mean(filter_data,axis=-1,keepdims=True)) / np.std(filter_data,axis=-1,keepdims=True)

@lru_cache(maxsize=8)
def butter_low_sos(order=4, low=0.5, fs=100):
    return signal.butter(
        order, low,
        btype='low',
        fs=fs,
        output='sos'
    )
def extract_dc(sig, order=4,fs=100, cutoff=0.5):
    sos = butter_low_sos(order, cutoff,  fs)
    if sig.ndim == 1:
        return signal.sosfiltfilt(sos, sig)
    else:
        return signal.sosfiltfilt(sos, sig, axis=-1)


def wavelet_denoise(data, wavelet='sym8', level=4):
    # 1. 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # 2. 阈值处理 (Soft Thresholding)
    # 估算噪声标准差
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # 对细节系数进行阈值过滤
    new_coeffs = list(coeffs)
    for i in range(1, len(coeffs)):
        new_coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    # 3. 重构
    return pywt.waverec(new_coeffs, wavelet)
def batch_calculate_hr(ppg_signals, fs=100, min_peak_distance=0.5, peak_height=0.3):
    """
    批量计算PPG信号的心率值

    参数:
        ppg_signals: 输入PPG信号，可以是1D数组(单个信号)或2D数组(N, L)
                     N为样本数量，L为每个样本的长度
        fs: 采样频率(Hz)
        min_peak_distance: 最小峰值间隔(秒)
        peak_height: 峰值高度阈值(相对于标准化信号)

    返回:
        心率数组，长度为N(每个样本的心率)
    """
    # 确保输入是2D数组以便统一处理
    if ppg_signals.ndim == 1:
        ppg_signals = ppg_signals[np.newaxis, :]

    n_samples = ppg_signals.shape[0]
    hr_results = np.zeros(n_samples)

    # 对每个样本进行处理
    for i in range(n_samples):
        # 1. 带通滤波
        filtered_ppg = ppg_signals[i]

        # 2. 检测峰值(脉搏波)
        min_samples_between_peaks = int(min_peak_distance * fs)
        try:
            _, info = nk.ppg_peaks(filtered_ppg, method="elgendi", sampling_rate=fs)
        # peaks, _ = find_peaks(
        #     filtered_ppg,
        #     distance=min_samples_between_peaks,
        #     height=peak_height
        # )
            peaks = np.asarray(info["PPG_Peaks"])
            # 3. 计算心率
            if len(peaks) >= 2:  # 需要至少两个峰值才能计算间隔
                # 计算峰值间隔(秒)
                peak_intervals = np.diff(peaks) / fs
                # 平均心率(bpm) = 60 / 平均间隔
                avg_hr = 60 / np.mean(peak_intervals)
                hr_results[i] = avg_hr
            else:
                hr_results[i] = 0.0  # 无法检测到足够的峰值
        except:
            hr_results[i] = 0.0  # 无法检测到足够的峰值
    return hr_results

    # return filter_data
def clear_dir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):  # 确保是文件
            os.remove(file_path)
def get_user_info(txt_data):

    # 存储提取的数据
    users_data = []
    for item in txt_data:
        value = item['value']
        if isinstance(value, dict):
            if 'userInfo' in value:
                # print(value)
                user_info = value["userInfo"]
                #
                # for match in matches:
                #     name,phone, height, weight, birthday, email = match
                height = calibrate_height(user_info["height"])
                weight = float(user_info["weight"])  # 体重转换为浮点数
                if height>0:
                    bmi =  weight / (height/100) ** 2
                else:
                    bmi = 0
                birthday_date = datetime.strptime(user_info["birthday"], "%Y-%m-%d")  # 将生日字符串转换为日期对象
                current_year = datetime.now().year  # 当前年份
                age = current_year - birthday_date.year - ((current_year == birthday_date.year) and (datetime.now().month, datetime.now().day) < (birthday_date.month, birthday_date.day))  # 计算年龄
                age = float(age)
                # print(email)
                gender = user_info["gender"]

                users_data = {
                    # 'name': user_info["name"],
                    'gender':gender,
                    'height': height,
                    'weight': weight,
                    "bmi":bmi,
                    'birthday': user_info["birthday"],
                    'age': age,
                }
    return users_data

def get_timestamp(txt_data):

    action_dict = {}

    for entry in txt_data:
        # print(entry)
        timestamp = entry['timestamp']
        value = entry['value']
        if isinstance(value, dict) and 'action' in value:
            # print(entry)
            action = value['action']
            if action != '开始':
                continue
        else:
            if "第" in value and "次测量" in value:
                # print(value)
                action = value
        action_dict[action] = timestamp
    results = [(timestamp, action) for action, timestamp in action_dict.items()]
    return results
def get_position(txt_data):
    position_map = {'坐':'sit', '躺': 'lay', '站立':'stand'}

    position = None
    for entry in txt_data:
        value = entry['value']
        # print(value)
        if isinstance(value, dict) and value['action']  == '开始' and 'posture' in value:
            position = position_map[value['posture']]
            break
    return position

def get_bloodpressure(txt_data):


    bp_value = defaultdict(list)
    for entry in txt_data:
        timestamp = entry['timestamp']
        value = entry['value']
        if isinstance(value, dict) and 'sys' in value and value['errcode'] == 0:
            bp_value["sbp"].append(value['sys'])
            bp_value["dbp"].append(value['dia'])
            bp_value["mbp"].append(value['mean'])
            bp_value["pulse_rate"].append(value['pulse'])

    return bp_value
def get_bpLevel(sbp, dbp):
    if sbp < 120 and dbp < 80:
        return 1
    elif 120 <= sbp <= 139 or 80 <= dbp <= 89:
        return 2
    elif 140 <= sbp <= 159 or 90 <= dbp <= 99:
        return 3
    elif 160 <= sbp <= 179 or 100 <= dbp <= 109:
        return 4
    elif sbp >= 180 or dbp >= 110:
        return 5
    elif sbp >= 140 or dbp < 90:
        return 6

def get_warn_list(checkdir,writetxt=1):
    checkfiles = glob.glob(os.path.join(checkdir, "*"))
    info_list = []

    for checkfile in checkfiles:
        checkcsvs = glob.glob(os.path.join(checkfile, "*", "*", "*PPG*.csv"))
        checkcsvs.sort(key=lambda x: 0 if "_左手_" in x else 1)

        checktxts = glob.glob(os.path.join(checkfile, "*", "*", "*.txt"))

        if not checkcsvs:
            continue

        has_txt = len(checktxts) > 0
        has_ppg = False
        has_left_index = False
        has_right_index = False
        bp_hand = "No"
        ppg_finger = set()

        # 检查 txt 判断 BP 手
        if has_txt:
            txt_config = os.path.basename(checktxts[0]).split("_")
            if txt_config[-2] == "左手":
                bp_hand = "left"
            else:
                bp_hand = "right"
        # ppg_fingers = []
        # 遍历 CSV
        # ppg_finger1 = 0
        # ppg_finger2 = 0
        for checkcsv in checkcsvs:
            ppg_finger.add(get_finger(checkcsv))
        finger_num = len(list(ppg_finger))

        # if len(checkcsvs) >=2:
        #     ppg_finger1 = get_finger(checkcsvs[0])
        #     ppg_finger2 = get_finger(checkcsvs[1])
        # if (hand == "左手" and finger == "食指"):
        #     has_left_index = True
        # elif (hand == "右手" and finger == "食指"):
        #     has_right_index = True

        info_list.append({
            "folder": checkfile,
            "has_txt": has_txt,
            "bp_hand": bp_hand,
            "finger_num":finger_num,
            # "ppg_finger1":sorted(list(ppg_finger))[0],
            # "ppg_finger2":sorted(list(ppg_finger))[1],
            # "has_ppg": has_ppg,
            # "has_left_index": has_left_index,
            # "has_right_index": has_right_index,
            "ppg_fingers": sorted(list(ppg_finger))  # 转成排序好的 list
        })
        # csvname = get_user_time_ring(checkcsv)
        # info_list.append([csvname, "PPG " + str(bool(ppgflag)), "Position " + str(bool(posflag))])
    df = pd.DataFrame(info_list)
    df.to_csv("check_result.csv", index=False, encoding="utf-8-sig")
    # ===== 保存统计结果到 CSV =====
    summary = {}
    for col in ["has_txt", "bp_hand", "finger_num"]:
        summary[col] = df[col].value_counts()
    summary_df = pd.DataFrame(summary).fillna(0).astype(int)
    summary_df.to_csv("check_summary.csv", encoding="utf-8-sig")
    # warn_list = []
    # if writetxt:
    #     with open('info_list.txt', 'w', encoding='utf-8') as file:
    #         for info in info_list:
    #             file.write(info[0] + " ")
    #             file.write(info[1] + " ")
    #             file.write(info[2] + " \n")
    # for info in info_list:
    #         if info[1] != "PPG True" or info[-1] != "Position True":
    #             warn_list.append(info[0])
    return summary_df

def find_nearest_timestamp_index(target_timestamp, timestamp_list):
    # CJR-2025-09-11注：这里可能需要修改

    # 计算目标时间戳与列表中每个时间戳的差值
    time_diffs = [abs(target_timestamp - ts) for ts in timestamp_list]
    # if np.min(time_diffs) > 3*1000:
    #     return np.nan
    # 找到最小差值的索引
    nearest_index = time_diffs.index(min(time_diffs))

    return nearest_index

def fill_missing_mirror(data, step=10, cols=["led_G","led_IR"]):
    """
    使用镜像复制法填充缺失的 PPG 信号
    """
    # 确保时间戳升序
    data = data.sort_values("timestamp").reset_index(drop=True)

    # 构造完整时间戳
    full_range = np.arange(data["timestamp"].iloc[0],
                           data["timestamp"].iloc[-1] + step,
                           step)
    full_data = pd.DataFrame({"timestamp": full_range})

    # 合并
    full_data = full_data.merge(data, on="timestamp", how="left")

    # 找出缺失区间
    # ts = full_data["timestamp"].values
    for col in cols:
        values = full_data[col].values

        i = 0
        while i < len(values):
            if np.isnan(values[i]):
                # 找到缺失段的起点
                start = i - 1
                end = i
                while end < len(values) and np.isnan(values[end]):
                    end += 1

                # 缺失段长度
                gap_len = end - start - 1

                if start >= 0 and end < len(values):
                    # 镜像复制：翻转前一段数据来补齐
                    segment = values[max(0, start - gap_len + 1): start + 1][::-1]
                    segment = np.tile(segment, int(np.ceil(gap_len / len(segment))))[:gap_len]
                    values[start + 1:end] = segment

            i += 1

        full_data[col] = values
    return full_data

# def fill_missing(data, step=10, cols=["led_G", "led_IR"]):
#     # 排序并重置索引
#     data = data.sort_values("timestamp").reset_index(drop=True)
#     if data.empty:  # 处理空数据情况
#         return pd.DataFrame(columns=["timestamp"] + cols)
#
#     # 生成完整的timestamp序列
#     start_ts = data["timestamp"].iloc[0]
#     end_ts = data["timestamp"].iloc[-1]
#     full_range = np.arange(start_ts, end_ts + step, step)
#     full_data = pd.DataFrame({"timestamp": full_range}).merge(data, on="timestamp", how="left")
#
#     # 对每个指定列进行处理
#     for col in cols:
#         # 获取该列非缺失值的记录（包含timestamp和对应值）
#         non_missing = full_data[["timestamp", col]].dropna(subset=[col])
#         if len(non_missing) < 2:  # 非缺失值不足2个，无法计算均值，保持NaN
#             continue
#
#         # 遍历相邻的非缺失值对，处理中间的缺失
#         for i in range(len(non_missing) - 1):
#             prev_row = non_missing.iloc[i]
#             next_row = non_missing.iloc[i + 1]
#
#             ts_prev, val_prev = prev_row["timestamp"], prev_row[col]
#             ts_next, val_next = next_row["timestamp"], next_row[col]
#             time_diff = ts_next - ts_prev
#
#             # 仅处理间隔为2*step的情况（刚好缺失一帧）
#             if time_diff == 2 * step:
#                 # 计算缺失位置的timestamp
#                 missing_ts = ts_prev + step
#                 # 找到对应索引并填充均值
#                 missing_idx = full_data[full_data["timestamp"] == missing_ts].index
#                 if not missing_idx.empty:
#                     full_data.loc[missing_idx, col] = (val_prev + val_next) / 2
#
#     return full_data


def fill_missing(data, step=10, cols=["led_G", "led_IR"]):
    # 排序并重置索引
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=["timestamp"] + cols)

    # 生成完整的timestamp序列（等差序列）
    start_ts = data["timestamp"].iloc[0]
    end_ts = data["timestamp"].iloc[-1]
    full_range = np.arange(start_ts, end_ts + step, step)
    full_data = pd.DataFrame({"timestamp": full_range}).merge(data, on="timestamp", how="left")

    # 预存full_data的timestamp起始值和列索引，避免重复计算
    ts_start = full_data["timestamp"].iloc[0]
    col_indices = {col: full_data.columns.get_loc(col) for col in cols}  # 列名到索引的映射

    for col in cols:
        # 提取非缺失值的timestamp和对应值（转为numpy数组，加速操作）
        non_missing = full_data[["timestamp", col]].dropna(subset=[col])
        if len(non_missing) < 2:
            continue

        ts = non_missing["timestamp"].to_numpy()  # 非缺失值的timestamp数组
        vals = non_missing[col].to_numpy()  # 非缺失值的数值数组

        # 向量化计算相邻非缺失值的时间差
        time_diffs = ts[1:] - ts[:-1]
        # 筛选出时间差为2*step的位置（需要填充的相邻对）
        mask = (time_diffs == 2 * step)
        valid_indices = np.where(mask)[0]  # 符合条件的相邻对索引

        if len(valid_indices) == 0:
            continue  # 没有需要填充的位置，直接跳过

        # 批量计算缺失的timestamp和填充值
        missing_ts = ts[valid_indices] + step  # 缺失位置的timestamp（等差计算）
        fill_vals = (vals[valid_indices] + vals[valid_indices + 1]) / 2  # 均值

        # 利用timestamp的规律性，直接计算缺失行在full_data中的索引
        # 因为full_data的timestamp是等差序列：ts = ts_start + index * step → index = (ts - ts_start) // step
        row_indices = (missing_ts - ts_start) // step

        # 批量填充（直接通过索引赋值，避免循环）
        full_data.iloc[row_indices, col_indices[col]] = fill_vals
        # ----------- 最终线性插值，确保无 NaN -----------
    full_data[cols] = full_data[cols].interpolate(method="linear", limit_direction="both")

    # 再次填充（防止线性插值首尾仍出现 NaN）
    full_data[cols] = full_data[cols].ffill().bfill()

    return full_data
def safe_ppg_quality(ppg_f, sr=100):
    """安全计算 PPG 质量，异常时返回 0.0"""
    try:
        return float(np.mean(nk.ppg_quality(ppg_f, sampling_rate=sr)))
    except Exception:
        return 0.0
def record_wrong(record,records_file):
    if record in records_file:
        pass
    else:
        records_file.append(record)
    return records_file


def save_all_errors_to_one_file(
        user_info_wrong_txts,
        bp_wrong_records,
        ppg_wrong_records,
        save_path="./logs/all_error_records.txt"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    merged = []
    merged += ["user_info_wrong",user_info_wrong_txts,"="*20]
    merged += ["bp_wrong",bp_wrong_records,"="*20  ]
    merged += ["ppg_wrong",ppg_wrong_records,"="*20 ]

    # 写入文件，每行一个 JSON
    with open(save_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n\n\n")

    print(f"✔ 合并错误记录已保存到: {save_path}")
    print(f"   总条目: {len(merged)}")
    return merged
def calibrate_height(height_value):
    """
    校准身高数据，处理各种可能的错误格式

    参数:
        height_value: 原始身高值，可以是字符串、数字等

    返回:
        float: 校准后的身高值(单位: cm)，如果无法校准返回None
    """
    # 尝试转换为浮点数
    height_num = float(height_value)

    # 根据数值范围判断可能的格式并校准
    if 0.5 <= height_num <= 2.5:
        # 可能是米单位，如1.75 -> 175cm
        return height_num * 100
    elif 10 <= height_num <= 30:
        # 可能是分米单位或有小数点错误，如17.5 -> 175cm
        # 检查是否可能是漏了小数点
        if height_num > 25:  # 不太可能有人超过2.5米
            return height_num * 10
        else:
            return height_num * 10  # 17.5 -> 175
    elif 500 <= height_num <= 3000:
        # 可能是毫米单位或多余的数字，如1750 -> 175cm
         return height_num / 10  # 1750 -> 175
    elif 50 <= height_num <= 250:
        # 正常的厘米范围
        return height_num
    else:
        return 0.0
