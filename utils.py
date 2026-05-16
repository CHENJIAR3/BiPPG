import argparse
import torch
import numpy as np
import random
parser = argparse.ArgumentParser(description='PPG信号分析程序')

#数据集的划分
parser.add_argument('--train_size', default=0.6, type=float, help='训练数据集比例.')
parser.add_argument('--val_size', default=0.2, type=float, help='验证数据集比例.')
parser.add_argument('--test_size', default=0.2, type=float, help='测试数据集比例.')
# 实验范式
parser.add_argument('--exp', default="ALL", type=str, help='Path to save or load models.')

# 数据维度
parser.add_argument('--out_dim',type=int,default=4,help="输出维度")
parser.add_argument('--in_channels',type=int,default=4,help="输入维度")
parser.add_argument('--seq_len',type=int,default=1000,help="输入长度")
#需要的ppg和label的key值
parser.add_argument('--win_len', type=int, default=10,)
parser.add_argument('--ppg_key', type=list, default=['ppg_g_filter_1',  'ppg_ga_filter_1',  'ppg_r_filter_1','ppg_ir_filter_1',
                    'ppg_g_filter_2','ppg_ga_filter_2', 'ppg_r_filter_2', 'ppg_ir_filter_2', ],)

parser.add_argument('--dc_key', type=list, default=['ppg_g_DC_1', 'ppg_ga_DC_1','ppg_r_DC_1', 'ppg_ir_DC_1',
                        'ppg_g_DC_2', 'ppg_ga_DC_2','ppg_r_DC_2', 'ppg_ir_DC_2',],)

parser.add_argument('--quality_key', type=list, default= ['ppg_quality(ppg_g_1)','ppg_quality(ppg_ga_1)','ppg_quality(ppg_r_1)','ppg_quality(ppg_ir_1)',
                        'ppg_quality(ppg_g_2)','ppg_quality(ppg_ga_2)','ppg_quality(ppg_r_2)','ppg_quality(ppg_ir_2)',],)

parser.add_argument('--index', type=list, default=range(16))
parser.add_argument('--label_key', type=list, default=["sbp_fix","dbp_fix","pr_ref","bmi","age",'gender',"position"],)
parser.add_argument('--label_index', type=list, default=range(8))
parser.add_argument('--norm_method', default=None,help="归一化方法")

parser.add_argument('--modelpath', default="./saved_models/PPG2BP", type=str, help='Path to save or load models.')
# 实验范式


# 模型训练超参数
parser.add_argument('--only_data', default=False, type=bool, help='只需要数据.')
parser.add_argument('--patience', default=50,help="模型耐心次数")
parser.add_argument('--epochs', default=500,help="模型训练迭代次数")

parser.add_argument('--random_seed', default=42, type=int, help='随机种子.')
parser.add_argument('--bs', default=256, type=int, help='Batch size for training.')
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--num_workers',type=int,default=4,help="工作节点数量")
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimization.')

parser.add_argument('--lr_scheduler', default="CosineAnnealingLR", type=str, help='Learning rate scheduler for optimization.')
parser.add_argument('--clip_value', default=1.0, type=float, help="clip_value")
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight_decay')
parser.add_argument('--recon_weight', default=1.0, type=float, help='reconstruction weight')
parser.add_argument('--vqvae_path', default="./saved_models/PPG2VQ", type=str, help='Path to save or load models.')


parser.add_argument('--trainflag', default=True,help="是否训练模型")
parser.add_argument('--testflag', default=True,help="是否测试模型")
parser.add_argument('--mhc_ratio',type=int,default=4,help="mhc长度")

parser.add_argument('--nhead', default=8)
parser.add_argument('--patch_size',default=10)
parser.add_argument('--dropout', default=0.0)
parser.add_argument("--num_encoder_layers",default=8)
parser.add_argument("--num_decoder_layers",default=8)
parser.add_argument('--d_model',type=int,default=256,help="d_model")
parser.add_argument('--embed',type=int,default=64,help="嵌入维度")
parser.add_argument('--freq',type=int,default=64,help="频率")

parser.add_argument('--factor',type=int,default=4,help="比例")
parser.add_argument('--max_len',default=4096,type=int)

args = parser.parse_args()



def to_tensor(array):
    return torch.from_numpy(array).float()

def setup_seed(args):
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

def get_total_params(model):
    """计算模型总参数数（兼容PyTorch）"""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
