# “We adopt a channel-independent patch-based Transformer to reduce temporal redundancy and mitigate cross-channel interference.”
# “Patch length is aligned with physiological periodicity.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps

        x = (x - mean) / std

        return x


class AttentionPooling(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):

        w = self.attn(x)
        w = torch.softmax(w, dim=1)

        return (w * x).sum(dim=1)


class PatchTST(nn.Module):

    """
    Input : (N, C, L)
    Output: (N, m)
    """

    def __init__(
        self,
        in_channels,
        seq_len=1000,
        patch_len=20,
        d_model=256,
        depth=4,
        n_heads=4,
        out_dim=1,
        dropout=0.2
    ):

        super().__init__()

        self.patch_len = patch_len
        self.stride = patch_len // 2   # overlap patch

        self.num_patches = (seq_len - patch_len) // self.stride + 1

        # self.revin = RevIN()

        # temporal embedding (better for physiological signals)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(
                in_channels,
                d_model,
                kernel_size=patch_len,
                stride=self.stride
            ),
            nn.GELU(),
            # nn.LayerNorm([1,d_model])
        )
        self.patch_norm = nn.LayerNorm(d_model)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            depth
        )

        self.pool = AttentionPooling(d_model)

        # channel fusion
        self.channel_fusion = nn.Sequential(
            nn.Linear(in_channels * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):

        # x: (N, C, L)

        # N, C, L = x.shape
        # self.mean = torch.mean(x, dim=[1,2], keepdim=False)
        # self.std = torch.std(x, dim=[1,2], keepdim=False)
        #
        #
        # x = self.revin(x)

        outputs = []

        # for c in range(C):

            # xc = x[:, c:c+1]

        # conv patch embedding
        xc = self.patch_embed(x)
        xc = xc.transpose(1, 2)  # (N, patches, d_model)

        xc = self.patch_norm(xc)

        xc = xc + self.pos_embedding

        xc = self.encoder(xc)

        outputs = self.pool(xc)

            # outputs.append(xc)

        # x = torch.cat(outputs, dim=1)

        # x = self.channel_fusion(x)
        y = self.head(outputs)

        return y


def myPatchTST(args):

    return PatchTST(
        in_channels=args.in_channels,
        seq_len=args.seq_len,
        out_dim=args.out_dim
    )
def myPatchTST(args):
    return  PatchTST(
        in_channels=args.in_channels,
        seq_len=args.seq_len,
        out_dim=args.out_dim)

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from utils import  args,get_total_params
    model =  myPatchTST(args)
    x = torch.randn((1,4,1000))
    # print(model(x).shape)
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
        print_per_layer_stat=False
    )
    print('MACs:', macs)
    print('Params:', params)
    # 4. 计算并打印参数表格 (非常适合放进 PPT 或论文附录)
    # print(parameter_count_table(model))
    #