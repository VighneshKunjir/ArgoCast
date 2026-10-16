"""Comparison forecasters for the ArgoCast 10-model benchmark.

Every neural model here solves exactly the ArgoCast task:
    input  x : (batch, 3, 32)  three past cycles, each [16 T levels, 16 S levels], standardised
    output   : (T_hat, S_hat), each (batch, 16), standardised next-cycle profile

They return the same (T, S) tuple as app.ml.model.PhysicsInformedBiLSTM, so the
training loop, loss and evaluation code are shared without changes.

The input history is only three steps long, so the time-series architectures
(TCN, DLinear, N-HiTS, TFT) are compact adaptations of the published designs to
this short-window, 32-channel, single-step-ahead setting. Each class docstring
states what was kept and what was simplified, so the paper can describe it
accurately.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FEAT = 32
N_LEV = 16
SEQ_LEN = 3


class _DualHead(nn.Module):
    """Two linear heads (temperature, salinity), as in the ArgoCast model."""

    def __init__(self, d_in: int):
        super().__init__()
        self.t = nn.Linear(d_in, N_LEV)
        self.s = nn.Linear(d_in, N_LEV)

    def forward(self, h):
        return self.t(h), self.s(h)


# --------------------------------------------------------------------- GRU
class GRUForecaster(nn.Module):
    """2-layer unidirectional GRU, 128 units, dropout 0.2, dual heads.

    Direct ablation of the ArgoCast PI-BiLSTM: same depth, width, dropout and
    heads, but GRU gating and a single (forward) direction.
    """

    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.GRU(N_FEAT, hidden, layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = _DualHead(hidden)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


# ------------------------------------------------------- Seq2Seq + attention
class Seq2SeqAttention(nn.Module):
    """LSTM encoder-decoder with additive (Bahdanau) attention.

    Encoder: 2-layer LSTM (128) over the three input cycles.
    Decoder: one LSTMCell step initialised from the top encoder state, fed the
    most recent input cycle; additive attention over the three encoder outputs
    gives a context vector. The attention weights over (t-3, t-2, t-1) are kept
    in `self.last_attention` for the explainability comparison.
    """

    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.enc = nn.LSTM(N_FEAT, hidden, layers, batch_first=True, dropout=dropout)
        self.dec = nn.LSTMCell(N_FEAT + hidden, hidden)
        self.W_enc = nn.Linear(hidden, hidden, bias=False)
        self.W_dec = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
        self.combine = nn.Linear(2 * hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = _DualHead(hidden)
        self.last_attention = None

    def _attend(self, H, h):
        score = self.v(torch.tanh(self.W_enc(H) + self.W_dec(h).unsqueeze(1))).squeeze(-1)  # (B, L)
        alpha = torch.softmax(score, dim=1)
        ctx = (alpha.unsqueeze(-1) * H).sum(1)
        return ctx, alpha

    def forward(self, x):
        H, (h_n, c_n) = self.enc(x)
        h, c = h_n[-1], c_n[-1]
        ctx0, _ = self._attend(H, h)
        h, c = self.dec(torch.cat([x[:, -1], ctx0], dim=1), (h, c))
        ctx, alpha = self._attend(H, h)
        self.last_attention = alpha.detach()
        z = torch.tanh(self.combine(torch.cat([h, ctx], dim=1)))
        return self.head(self.drop(z))


# --------------------------------------------------------------------- TCN
class _CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, dropout):
        super().__init__()
        self.pad = (k - 1) * d
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=d)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.drop(F.relu(self.c1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.c2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


class TCNForecaster(nn.Module):
    """Temporal Convolutional Network (Bai et al., 2018 style).

    Two residual blocks of causal dilated 1-D convolutions over time
    (kernel 2, dilations 1 and 2, 128 channels, dropout 0.2). The receptive
    field (7 steps) covers the whole 3-cycle input. The 32 profile values are
    the input channels; the output at the last time step feeds the dual heads.
    """

    def __init__(self, channels: int = 128, kernel: int = 2, dilations=(1, 2), dropout: float = 0.2):
        super().__init__()
        blocks, c_in = [], N_FEAT
        for d in dilations:
            blocks.append(_CausalBlock(c_in, channels, kernel, d, dropout))
            c_in = channels
        self.net = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = _DualHead(channels)

    def forward(self, x):
        y = self.net(x.transpose(1, 2))  # (B, C, L)
        return self.head(self.drop(y[:, :, -1]))


# ------------------------------------------------------------------ DLinear
class DLinear(nn.Module):
    """DLinear (Zeng et al., 2023), channel-independent ("individual") version.

    Each of the 32 series (16 T levels, 16 S levels) is split into a trend
    (moving average, kernel 3, edge-replicated) and a remainder; one linear
    layer per channel maps each 3-step component to the next value, and the two
    are added. There is no mixing across depths or variables, which is the
    defining property of the model. No dropout, so MC dropout does not apply.
    """

    def __init__(self, kernel: int = 3):
        super().__init__()
        self.kernel = kernel
        self.w_season = nn.Parameter(torch.zeros(N_FEAT, SEQ_LEN))
        self.w_trend = nn.Parameter(torch.zeros(N_FEAT, SEQ_LEN))
        self.b = nn.Parameter(torch.zeros(N_FEAT))
        with torch.no_grad():  # start from persistence (weight 1 on the last step)
            self.w_season[:, -1] = 1.0
            self.w_trend[:, -1] = 1.0

    def forward(self, x):
        xt = x.transpose(1, 2)  # (B, C, L)
        p = (self.kernel - 1) // 2
        trend = F.avg_pool1d(F.pad(xt, (p, p), mode="replicate"), self.kernel, stride=1)
        season = xt - trend
        out = (season * self.w_season).sum(-1) + (trend * self.w_trend).sum(-1) + self.b
        return out[:, :N_LEV], out[:, N_LEV:]


# ------------------------------------------------------------------- N-HiTS
class _NHiTSBlock(nn.Module):
    def __init__(self, pool: int, hidden: int, dropout: float):
        super().__init__()
        self.pool = pool
        n_in = N_FEAT * -(-SEQ_LEN // pool)  # ceil(L / pool) points per channel
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.backcast = nn.Linear(hidden, N_FEAT * SEQ_LEN)
        self.forecast = nn.Linear(hidden, N_FEAT)

    def forward(self, r):  # r: (B, C, L)
        z = F.max_pool1d(r, self.pool, stride=self.pool, ceil_mode=True) if self.pool > 1 else r
        h = self.mlp(z.flatten(1))
        return self.backcast(h).view(-1, N_FEAT, SEQ_LEN), self.forecast(h)


class NHiTS(nn.Module):
    """N-HiTS (Challu et al., 2023), multivariate adaptation.

    Three stacks of one MLP block each (512 units, two hidden layers) with
    multi-rate input pooling (kernels 1, 2, 3 over the time axis) and doubly
    residual stacking: each block removes its backcast from the input and adds
    its forecast to the output. With a one-step horizon the hierarchical
    interpolation of the forecast reduces to a direct linear output, and the
    32 channels are modelled jointly so cross-depth structure can be used.
    """

    def __init__(self, hidden: int = 512, pools=(1, 2, 3), dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([_NHiTSBlock(p, hidden, dropout) for p in pools])

    def forward(self, x):
        r = x.transpose(1, 2)
        out = 0.0
        for b in self.blocks:
            bc, fc = b(r)
            r = r - bc
            out = out + fc
        return out[:, :N_LEV], out[:, N_LEV:]


# ---------------------------------------------------------------------- TFT
class GRN(nn.Module):
    """Gated residual network (Lim et al., 2021)."""

    def __init__(self, d_in, d_hid, d_out, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_hid)
        self.drop = nn.Dropout(dropout)
        self.glu = nn.Linear(d_hid, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a):
        h = self.fc2(F.elu(self.fc1(a)))
        h = F.glu(self.glu(self.drop(h)), dim=-1)
        return self.norm(self.skip(a) + h)


class _GateAddNorm(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.glu = nn.Linear(d, 2 * d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, skip):
        return self.norm(skip + F.glu(self.glu(self.drop(x)), dim=-1))


class TemporalFusionTransformer(nn.Module):
    """Compact Temporal Fusion Transformer (Lim et al., 2021).

    Kept from the original: per-variable embeddings with a softmax variable
    selection network (one weight per input feature per time step), an LSTM
    encoder with gated skip connection, interpretable multi-head attention
    (heads share the value projection, attention averaged over heads), gated
    residual networks and layer norms throughout.
    Simplified: there are no static covariates and no known-future inputs in
    this task, so the static encoders and the decoder LSTM are omitted; the
    single forecast step attends over the three encoded past cycles.
    Variable-selection weights (B, 3, 32) and attention weights (B, 3) are kept
    in `self.last_var_weights` / `self.last_attention` for interpretation.
    """

    def __init__(self, d: int = 64, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d, self.heads = d, heads
        self.emb_w = nn.Parameter(torch.randn(N_FEAT, d) * 0.1)
        self.emb_b = nn.Parameter(torch.zeros(N_FEAT, d))
        self.var_grn = GRN(d, d, d, dropout)                 # shared across variables
        self.sel_grn = GRN(N_FEAT * d, d, N_FEAT, dropout)    # selection weights
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.post_lstm = _GateAddNorm(d, dropout)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d // heads)                    # shared value (interpretable MHA)
        self.attn_out = nn.Linear(d // heads, d)
        self.post_attn = _GateAddNorm(d, dropout)
        self.pos_grn = GRN(d, d, d, dropout)
        self.post_ff = _GateAddNorm(d, dropout)
        self.drop = nn.Dropout(dropout)
        self.head = _DualHead(d)
        self.last_var_weights = None
        self.last_attention = None

    def forward(self, x):
        B, L, C = x.shape
        e = x.unsqueeze(-1) * self.emb_w + self.emb_b             # (B, L, C, d)
        e_proc = self.var_grn(e)                                  # (B, L, C, d)
        w = torch.softmax(self.sel_grn(e.reshape(B, L, C * self.d)), dim=-1)  # (B, L, C)
        sel = (w.unsqueeze(-1) * e_proc).sum(2)                   # (B, L, d)
        self.last_var_weights = w.detach()

        h, _ = self.lstm(sel)
        h = self.post_lstm(h, sel)                                # (B, L, d)

        dk = self.d // self.heads
        q = self.q(h[:, -1:]).view(B, 1, self.heads, dk).transpose(1, 2)   # (B, H, 1, dk)
        k = self.k(h).view(B, L, self.heads, dk).transpose(1, 2)            # (B, H, L, dk)
        v = self.v(h)                                                        # (B, L, dk)
        att = torch.softmax(q @ k.transpose(-1, -2) / dk ** 0.5, dim=-1)    # (B, H, 1, L)
        att_mean = att.mean(1)                                               # (B, 1, L)
        self.last_attention = att_mean.squeeze(1).detach()
        a = self.attn_out(att_mean @ v)                                      # (B, 1, d)
        z = self.post_attn(a, h[:, -1:])
        z = self.post_ff(self.pos_grn(z), h[:, -1:]).squeeze(1)
        return self.head(self.drop(z))


def build(name: str) -> nn.Module:
    from app.ml.model import PhysicsInformedBiLSTM
    table = {
        "GRU": GRUForecaster,
        "Seq2Seq LSTM + attention": Seq2SeqAttention,
        "TCN": TCNForecaster,
        "DLinear": DLinear,
        "N-HiTS": NHiTS,
        "TFT": TemporalFusionTransformer,
        "Bi-LSTM (no physics loss)": PhysicsInformedBiLSTM,
        "ArgoCast PI-BiLSTM": PhysicsInformedBiLSTM,
    }
    return table[name]()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
