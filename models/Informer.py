#还没实现，需要查看一下！
# : Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021).
# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProbSparseAttention(nn.Module):
    """ProbSparse Self-Attention Mechanism (Section 3.1)"""

    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.factor = factor  # Sampling factor (default=5 per paper)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _prob_qr(self, Q, K):
        """Probabilistic sampling of queries and keys (Eq 6)"""
        # Q: (N, H, L, D), K: (N, H, L, D)
        B, H, L, D = Q.shape

        # Step 1: Compute u = top_k(||QK^T||, factor*ln(L))
        QK = torch.einsum("nhld,nhmd->nhlm", Q, K)  # (N, H, L, L)
        QK = QK.abs().mean(dim=1)  # (N, L, L) - average over heads

        # Step 2: Get top indices (u = factor * ln(L) samples)
        u = int(self.factor * math.log(L))
        U = QK.max(-1).values  # (N, L)
        U = U.sort(dim=-1, descending=True)[0]
        u = min(u, L // 2)  # Safety check
        cutoff = U[:, u:u + 1].mean()  # (N, 1)

        # Step 3: Create mask (M_q for queries, M_k for keys)
        M_q = (QK.mean(dim=-1) > cutoff).float()  # (N, L)
        M_k = (QK.mean(dim=-2) > cutoff).float()  # (N, L)

        return M_q, M_k

    def forward(self, x, attn_mask=None):
        """Input: x (N, L, D)"""
        B, L, _ = x.shape
        residual = x

        # Linear projections
        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # (B, H, L, D)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # Probabilistic sampling (M_q, M_k)
        M_q, M_k = self._prob_qr(Q, K)

        # Apply masks (B, H, L, D) * (B, 1, L, 1)
        Q = Q * M_q.unsqueeze(1).unsqueeze(-1)
        K = K * M_k.unsqueeze(1).unsqueeze(-1)

        # Scaled dot-product attention
        scores = torch.einsum("nhld,nhmd->nhlm", Q, K) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)

        A = F.softmax(scores, dim=-1)
        A = self.dropout(A)
        V = torch.einsum("nhlm,nhmd->nhld", A, V)  # (B, H, L, D)

        # Concat heads
        V = V.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        out = self.fc(V)

        return out + residual, None  # Return attention weights for completeness


class DistilLayer(nn.Module):
    """Distilling operation (Section 3.2) - reduces sequence length"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='zeros'
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Input: x (N, L, D) -> Output: (N, L/2, D)"""
        x = x.permute(0, 2, 1)  # (N, D, L)
        x = self.norm(self.conv(x))
        x = F.gelu(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)  # (N, L/2, D)


class InformerEncoderLayer(nn.Module):
    """Single Informer Encoder Layer (Fig 2)"""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # ProbSparse attention
        attn_out, _ = self.attn(x, attn_mask)
        x = self.norm1(x + attn_out)

        # Position-wise feedforward
        y = x.permute(0, 2, 1)
        y = self.dropout(F.gelu(self.conv1(y)))
        y = self.conv2(y).permute(0, 2, 1)
        y = self.dropout(y)

        return self.norm2(x + y)


class InformerEncoder(nn.Module):
    """Full Informer Encoder with Distilling (Fig 3)"""

    def __init__(self,
                 in_channels,
                 seq_len,
                 d_model=512,
                 n_heads=8,
                 d_ff=2048,
                 n_layers=2,
                 dropout=0.1):
        super().__init__()
        # Channel-independent processing
        self.channel_proj = nn.Linear(in_channels, d_model)

        # Positional encoding (fixed frequency - Section 3.3)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, D)

        # Encoder layers with distilling
        self.layers = nn.ModuleList()
        current_len = seq_len
        for _ in range(n_layers):
            self.layers.append(InformerEncoderLayer(d_model, n_heads, d_ff, dropout))
            if current_len > 1:  # Only add distill if length > 1
                self.layers.append(DistilLayer(d_model, dropout))
                current_len = current_len // 2  # Distill halves length

    def forward(self, x):
        """Input: x (N, C, L)"""
        N, C, L = x.shape

        # Step 1: Channel-independent projection (N, C, L) -> (N, L, D)
        x = x.permute(0, 2, 1)  # (N, L, C)
        x = self.channel_proj(x)  # (N, L, D)

        # Step 2: Add positional encoding
        x = x + self.pe[:, :L, :]

        # Step 3: Encoder layers with distilling
        for layer in self.layers:
            if isinstance(layer, DistilLayer):
                x = layer(x)  # (N, L/2, D)
            else:
                x = layer(x)  # (N, L, D)

        # Step 4: Global pooling (N, D)
        return x.mean(dim=1)


class InformerClassifier(nn.Module):
    """Full Informer for Classification Tasks"""

    def __init__(self,
                 in_channels,
                 seq_len,
                 num_classes,
                 d_model=256,
                 n_heads=8,
                 d_ff=1024,
                 n_layers=8,
                 dropout=0.1):
        super().__init__()
        self.encoder = InformerEncoder(
            in_channels, seq_len, d_model, n_heads, d_ff, n_layers, dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """Input: x (N, C, L) -> Output: (N, m)"""
        features = self.encoder(x)  # (N, D)
        return self.classifier(features)


def myInformer(args):
    """Model builder function matching your baseline interface"""
    return InformerClassifier(
        in_channels=args.in_channels,
        seq_len=args.seq_len,  # Must be provided
        num_classes=args.out_dim,
        d_model=256,  # Reduced for classification (paper uses 512 for prediction)
        n_heads=4,  # Reduced heads for smaller models
        n_layers=4  # Fewer layers (classification needs less depth)
    )


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # from types import SimpleNamespace
    # # Simulate your args structure
    # args = SimpleNamespace(
    #     in_channels=8,  # Example: 8-channel sensor data
    #     seq_len=1000,  # Sequence length required
    #     out_dim=5  # Example: 5-class classification
    # )
    from utils import args,get_total_params


    model = myInformer(args)
    x = torch.randn((1, 4, 1000))  # (N, C, L) = (2, 8, 1000)

    # Verify output shape
    output = model(x)
    # assert output.shape == (1, 4), f"Output shape mismatch: {output.shape}"
    # print("Output shape verified: (N, m) = (2, 4)")

    # Compute FLOPs and parameters
    flops = FlopCountAnalysis(model, x)
    print(f"Total FLOPs: {flops.total() / 1e6:.2f} M")
    total_params = get_total_params(model)
    print(f"模型参数总数: {total_params:,}")
    # print(parameter_count_table(model))
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(
        model,
        (4, 1000),
        as_strings=True,
        print_per_layer_stat=True
    )
    print('MACs:', macs)
    print('Params:', params)