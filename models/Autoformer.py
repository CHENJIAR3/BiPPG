import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2)

    def forward(self, x):
        trend = self.avg(x)
        season = x - trend
        return season, trend


class AutoCorrelation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # (B, T, D)
        corr = torch.matmul(x, x.transpose(1, 2))
        attn = torch.softmax(corr, dim=-1)
        return self.proj(torch.matmul(attn, x))


class AutoformerCI(nn.Module):
    """
    Input : (N, C, L)
    Output: (N, m)
    """
    def __init__(
        self,
        in_channels,
        seq_len,
        d_model=256,
        depth=8,
        out_dim=1
    ):
        super().__init__()

        self.embed = nn.Linear(seq_len, d_model)
        self.decomp = SeriesDecomp()

        self.blocks = nn.ModuleList([
            AutoCorrelation(d_model) for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):
        N, C, L = x.shape
        x = x.reshape(N * C, 1, L)

        season, trend = self.decomp(x)
        x = self.embed(season.squeeze(1)).unsqueeze(1)

        for blk in self.blocks:
            x = x + blk(x)

        x = x.mean(dim=1)
        x = x.reshape(N, C, -1).mean(dim=1)

        return self.head(x)


def myAutoformer(args):
    return AutoformerCI(
        in_channels=args.in_channels,
        seq_len=args.seq_len,
        out_dim=args.out_dim
    )
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from utils import  args,get_total_params

    model =  myAutoformer(args)
    x = torch.randn((1,4,1000))
    print(model(x).shape)
    # 3. 计算 FLOPs
    flops = FlopCountAnalysis(model, x)
    print(f"Total FLOPs: {flops.total() / 1e6:.2f} M")  # 以百万为单位
    total_params = get_total_params(model)
    print(f"模型参数总数: {total_params:,}")
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model,
        (4, 1000),
        as_strings=True,
        print_per_layer_stat=True
    )
    print('MACs:', macs)
    print('Params:', params)
    # 4. 计算并打印参数表格 (非常适合放进 PPT 或论文附录)
    # print(parameter_count_table(model))