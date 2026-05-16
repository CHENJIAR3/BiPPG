# 导入各种模型
from models.net1D import myNet1D,myNet1D_moe
from models.CRNN import mycrnn
from models.ACNN import myacnn
from models.ResNet1D import myResNet1D
from models.efficientnet import myEfficientNet
from models.LSTM import mylstm
from models.PatchTST import myPatchTST
from models.Autoformer import myAutoformer
from models.iTransformers import myiTransformer
from models.Informer import  myInformer
from models.CSFM import CSFM_model
from models.papagei_resnet import myResNet1DMoE
from torch import nn
def init_weights(model, init_type="kaiming"):
    """
    init_type:
        "kaiming"  → Kaiming 正态（适合 ReLU/GELU 激活，CNN/ResNet 首选）
        "xavier"   → Xavier 均匀（适合 Sigmoid/Tanh，Transformer 常用）
        "orthogonal" → 正交初始化（适合 RNN/LSTM）
        "normal"   → 简单正态 N(0, 0.02)
        "default"  → 不做任何初始化，用 PyTorch 默认
    """
    if init_type == "default":
        return

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif init_type == "normal":
                nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)   # RNN hidden 用正交更稳定
                elif "bias" in name:
                    nn.init.zeros_(param)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def load_model(args):
    model_type = getattr(args,"model_type","Net1D")
    print(model_type)



    if model_type == "CRNN":
        model = mycrnn(args)
    if model_type == "ACNN":
        model = myacnn(args)

    if model_type == "ResNet1D":
        model = myResNet1D(args)
    if model_type == "Net1D":
        model = myNet1D(args)
    if model_type == "LSTM":
        model = mylstm(args)

    if model_type == "Efficient1D":
        model = myEfficientNet(args)

    if model_type == "AutoFormer":
        model = myAutoformer(args)
    if model_type =="PatchTST":
        model = myPatchTST(args)
    if model_type =="Informer":
        model = myInformer(args)
    if model_type == "iTransformer":
        model = myiTransformer(args)
    if model_type == "ResNet1DMoE":
        model = myResNet1DMoE(args)

    if "CSFM" in model_type:
        model_size = model_type.split("_")[1]
        model = CSFM_model(args,model_size)

    return model