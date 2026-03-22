"""
Hybrid Transformer v2
  - RoPE    : 絶対位置ではなくトークン間距離を回転行列で表現
  - GQA     : Q=n_head, KV=n_kv_head (< n_head) で推論メモリを削減
  - GatedDeltaNet : 線形Attention。O(n)メモリで長文脈に対応
  - SwiGLU  : FFN の活性化関数（Llama/Qwen 系で標準）
  - RMSNorm : LayerNorm より軽量な正規化

2層構成:
  Layer 0 → GatedDeltaNet (線形Attention)
  Layer 1 → GQA Attention (標準Attention + RoPE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Config
# =============================================================================
class ModelConfigV2:
    def __init__(self,
                 vocab_size,
                 max_seq_len = 10000,   # RoPE が対応する最大コンテキスト長
                 n_layer     = 2,
                 n_head      = 4,       # Q ヘッド数
                 n_kv_head   = 2,       # KV ヘッド数 (GQA: n_head より少なくてOK)
                 n_embd      = 64,
                 ffn_mult    = 4,       # FFN の中間次元 = n_embd * ffn_mult
                 rope_base   = 10000):
        assert n_embd % n_head == 0,   "n_embd は n_head で割り切れる必要があります"
        assert n_head % n_kv_head == 0, "n_head は n_kv_head で割り切れる必要があります"
        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layer     = n_layer
        self.n_head      = n_head
        self.n_kv_head   = n_kv_head
        self.n_embd      = n_embd
        self.head_dim    = n_embd // n_head
        self.ffn_dim     = n_embd * ffn_mult
        self.rope_base   = rope_base


# =============================================================================
# RMSNorm (LayerNorm の軽量版)
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================
def precompute_rope(head_dim: int, max_seq_len: int, base: int = 10000, device='cpu'):
    """cos/sin テーブルを事前計算して register_buffer に登録する用"""
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t     = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, theta)           # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)  # 各 (T, head_dim/2)

def apply_rope(x, cos, sin):
    """
    x   : (B, n_head, T, head_dim)
    cos : (T, head_dim/2)
    sin : (T, head_dim/2)
    """
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., ::2], x[..., 1::2]     # 偶数・奇数インデックス
    # 回転: [x1, x2] → [x1·cos - x2·sin,  x1·sin + x2·cos]
    out = torch.stack([x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# =============================================================================
# SwiGLU FFN
# =============================================================================
class SwiGLU(nn.Module):
    """FFN: down_proj( silu(gate_proj(x)) * up_proj(x) )"""
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.up   = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# =============================================================================
# GQA Attention (Grouped Query Attention + RoPE + Causal mask)
# =============================================================================
class GQAAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim  = config.head_dim
        self.n_rep     = config.n_head // config.n_kv_head  # KV を何回繰り返すか

        self.q_proj = nn.Linear(config.n_embd, config.n_head    * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.n_embd,    bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)  # (B, nH, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)  # (B, nKV, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE を Q と K に適用
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV を Q ヘッド数に合わせて拡張 (GQA の核心)
        k = k.repeat_interleave(self.n_rep, dim=1)   # (B, nH, T, hd)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale    # (B, nH, T, T)
        mask  = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att   = att.masked_fill(~mask, float('-inf'))
        att   = F.softmax(att, dim=-1)

        y = att @ v                                   # (B, nH, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


# =============================================================================
# Gated DeltaNet (線形Attention)
# デルタルール: M_t = M_{t-1} + β(v - M·k)kᵀ
# メモリ行列 M に情報を蓄積し、クエリで読み出す
# =============================================================================
class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head   = config.n_head
        self.head_dim = config.head_dim
        d = config.n_embd

        self.q_proj    = nn.Linear(d, d, bias=False)
        self.k_proj    = nn.Linear(d, d, bias=False)
        self.v_proj    = nn.Linear(d, d, bias=False)
        self.beta_proj = nn.Linear(d, config.n_head, bias=True)   # ヘッドごとの書き込み強度
        self.gate_proj = nn.Linear(d, d, bias=False)              # 出力ゲート
        self.o_proj    = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        H, d = self.n_head, self.head_dim

        q    = self.q_proj(x).view(B, T, H, d)       # (B, T, H, d)
        k    = self.k_proj(x).view(B, T, H, d)
        v    = self.v_proj(x).view(B, T, H, d)
        beta = torch.sigmoid(self.beta_proj(x))       # (B, T, H)
        gate = torch.sigmoid(self.gate_proj(x))       # (B, T, C) — 出力ゲート

        # K を単位ベクトルに正規化（デルタルールの安定化）
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        # メモリ行列 M の初期化: (B, H, d, d)
        M = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        outputs = []

        # 時系列ループ: 各トークンでメモリを更新し読み出す
        for t in range(T):
            q_t = q[:, t]    # (B, H, d)
            k_t = k[:, t]
            v_t = v[:, t]
            b_t = beta[:, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # 読み出し: o = M @ q
            o_t = torch.einsum('bhdc,bhc->bhd', M, q_t)  # (B, H, d)

            # デルタ更新: M ← M + β·(v − M@k)·kᵀ
            Mk  = torch.einsum('bhdc,bhc->bhd', M, k_t)  # M @ k
            dv  = v_t - Mk                                # 予測誤差
            M   = M + b_t * torch.einsum('bhd,bhc->bhdc', dv, k_t)

            outputs.append(o_t)

        o = torch.stack(outputs, dim=1).view(B, T, C)  # (B, T, C)
        return self.o_proj(gate * o)                    # ゲートを掛けて出力


# =============================================================================
# Block: 層インデックスで DeltaNet か GQA かを切り替え
#   偶数層 → GatedDeltaNet
#   奇数層 → GQA (ただし最終層は必ず GQA)
# =============================================================================
class HybridBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn   = SwiGLU(config)

        # 最終層は GQA、それ以外は DeltaNet (3:1 の Qwen3.5 パターンに準拠)
        is_last = (layer_idx == config.n_layer - 1)
        self.use_gqa = is_last or (layer_idx % 4 == 3)
        if self.use_gqa:
            self.attn = GQAAttention(config)
        else:
            self.attn = GatedDeltaNet(config)

    def forward(self, x, cos, sin):
        if self.use_gqa:
            x = x + self.attn(self.norm1(x), cos, sin)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# HybridTransformer — メインモデル
# =============================================================================
class HybridTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.embed   = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks  = nn.ModuleList([HybridBlock(config, i) for i in range(config.n_layer)])
        self.norm_f  = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # RoPE テーブルを事前計算してバッファとして保持
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_base)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"入力長 {T} が max_seq_len {self.config.max_seq_len} を超えています"
        x   = self.embed(idx)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits   = self(idx_cond)[:, -1, :]
            idx_next = torch.argmax(F.softmax(logits, dim=-1), dim=-1, keepdim=True)
            idx      = torch.cat([idx, idx_next], dim=1)
        return idx
