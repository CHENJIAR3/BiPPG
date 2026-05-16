import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """
    LSTM for 1D signal

    Input:
        X: (B, C, L)

    Output:
        out: (B, n_classes)
    """

    def __init__(self, in_channels, hidden_dim, n_len_seg, n_classes, num_layers=1, verbose=False):
        super(LSTM, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.verbose = verbose

        # LSTM（替代CNN）
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )


        # 输出层
        self.fc = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, x):
        """
        x: (B, C, L)
        """

        B, C, L = x.shape
        assert (L % self.n_len_seg == 0), "Input length must be divisible by n_len_seg"

        n_seg = L // self.n_len_seg

        # ===== reshape 和 ACNN 一致 =====
        # (B, C, L) → (B, L, C)
        out = x.permute(0, 2, 1)

        # → (B*n_seg, n_len_seg, C)
        out = out.reshape(-1, self.n_len_seg, C)

        if self.verbose:
            print("After reshape:", out.shape)

        # ===== LSTM =====
        out, _ = self.lstm(out)  # (B*n_seg, n_len_seg, hidden_dim)

        # 取时间平均（类似CNN的global avg）
        out = out.mean(dim=1)  # (B*n_seg, hidden_dim)

        # reshape回 segment
        out = out.reshape(B, n_seg, 2*self.hidden_dim)

        if self.verbose:
            print("After LSTM:", out.shape)

        # # ===== Attention（segment级）=====
        # e = torch.matmul(out, self.W_att)              # (B, n_seg, att_channels)
        # e = torch.matmul(torch.tanh(e), self.v_att)    # (B, n_seg, 1)
        #
        # alpha = torch.softmax(e, dim=1)                # attention weights
        # out = torch.sum(alpha * out, dim=1)            # (B, hidden_dim)
        out = out.mean(dim=1)
        if self.verbose:
            print("After attention:", out.shape)

        # ===== FC =====
        out = self.fc(out)

        return out
def mylstm(args):
    return LSTM(
        in_channels=args.in_channels,
        hidden_dim=64,
        n_len_seg=10,
        n_classes=args.out_dim,
        num_layers=4,
        verbose=False
    )
if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from utils import  args,get_total_params
    model =  mylstm(args)
    x = torch.randn((1,4,1000))
    # print(model(x)["sbp"].shape)
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