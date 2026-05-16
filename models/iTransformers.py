# : Liu, J., Wang, Y., & Yang, Y. (2023).
# iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. arXiv preprint arXiv:2310.06610.

import torch.nn as nn
import torch.nn.functional as F
import torch
class iTransformer(nn.Module):
    def __init__(self, in_channels, seq_len, num_classes, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        patch_len = 20
        self.stride = patch_len // 2   # overlap patch
        # temporal embedding (better for physiological signals)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=patch_len,
                stride=self.stride
            ),
            nn.GELU(),
        )

        # Inverted dimension: 将C视为序列长度
        self.linear = nn.Linear(in_channels, d_model)  # (L, C) -> (L, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: (N, C, L)
        x = self.patch_embed(x)
        # Step 1: Invert dimensions: (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)
        # Step 2: Linear projection per "time step" (L)
        x = self.linear(x)  # (N, L, d_model)
        # Step 3: Transformer (L as sequence length)
        x = x.permute(1, 0, 2)  # (L, N, d_model)
        out = self.transformer(x)  # (L, N, d_model)
        out = out.permute(1, 0, 2).mean(dim=1)  # Pooling over L: (N, d_model)
        # Step 4: Classifier
        return self.classifier(out)  # (N, m)
def myiTransformer(args):
    return iTransformer(in_channels=args.in_channels,
                        seq_len=args.seq_len,
                        num_classes=args.out_dim)
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from utils import  args,get_total_params

    model =  myiTransformer(args)
    x = torch.randn((1,4,1000))
    print(model(x).shape)
    # 3. 计算 FLOPs
    flops = FlopCountAnalysis(model, x)
    print(f"Total FLOPs: {flops.total() / 1e6:.2f} M")  # 以百万为单位
    total_params = get_total_params(model)
    print(f"模型参数总数: {total_params:,}")
    # 4. 计算并打印参数表格 (非常适合放进 PPT 或论文附录)
    # print(parameter_count_table(model))
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model,
        (4, 1000),
        as_strings=True,
        print_per_layer_stat=False
    )
    print('MACs:', macs)
    print('Params:', params)