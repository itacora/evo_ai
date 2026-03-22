"""
chat_model.pth の構成を確認するスクリプト

使い方:
    python check_model.py
    python check_model.py --path chat_model_v2.pth
"""

import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

def check_model(path: str):
    if not os.path.exists(path):
        print(f"ファイルが見つかりません: {path}")
        return

    ckpt = torch.load(path, map_location="cpu")

    # config の取得（旧形式はデフォルト値で推定）
    if isinstance(ckpt, dict) and "config" in ckpt:
        c      = ckpt["config"]
        state  = ckpt["model"]
        suffix = ""
    else:
        state = ckpt
        vocab = state["transformer.wte.weight"].shape[0]
        block = state["transformer.wpe.weight"].shape[0]
        embd  = state["transformer.wpe.weight"].shape[1]
        n_lay = sum(1 for k in state if k.startswith("transformer.h.") and k.endswith(".attn.c_attn.weight"))
        n_hd  = embd // 16
        c = {"n_layer": n_lay, "n_head": n_hd, "n_embd": embd, "block_size": block, "vocab_size": vocab}
        suffix = "  (推定)"

    # v1 は block_size、v2 は max_seq_len で保存されている
    seq_len  = c.get("block_size") or c.get("max_seq_len", "不明")
    is_v2    = "max_seq_len" in c  # model_v2 かどうか

    print(f"=== {path} ===")
    print(f"  モデル種別         : {'HybridTransformer v2 (RoPE+GQA+DeltaNet)' if is_v2 else 'EvoTransformer (Dense)'}")
    print(f"  層数    (n_layer)  : {c['n_layer']}{suffix}")
    print(f"  ヘッド数 (n_head)  : {c['n_head']}{suffix}")
    print(f"  埋め込み (n_embd)  : {c['n_embd']}{suffix}")
    print(f"  文脈長             : {seq_len}{suffix}")
    print(f"  語彙数  (vocab_sz) : {c.get('vocab_size', '不明')}{suffix}")
    if is_v2:
        print(f"  KVヘッド数         : {c.get('n_kv_head', '不明')}")

    size_kb = os.path.getsize(path) / 1024
    print(f"  ファイルサイズ     : {size_kb:.1f} KB")

    # パラメータ数・層ごとの内訳
    try:
        if is_v2:
            from model_v2 import HybridTransformer, ModelConfigV2
            config = ModelConfigV2(
                vocab_size  = c["vocab_size"],
                max_seq_len = c["max_seq_len"],
                n_layer     = c["n_layer"],
                n_head      = c["n_head"],
                n_kv_head   = c.get("n_kv_head", 2),
                n_embd      = c["n_embd"],
                ffn_mult    = c.get("ffn_mult", 4),
            )
            model = HybridTransformer(config)
            model.load_state_dict(state, strict=False)

            total   = sum(p.numel() for p in model.parameters())
            buffers = sum(b.numel() for b in model.buffers())
            print(f"\n  学習可能パラメータ : {total:,}")
            print(f"  固定バッファ       : {buffers:,}  (RoPEテーブル等)")
            print(f"  規模               : {total/1e9:.6f}B  ({total/1e6:.4f}M)")

            print("\n--- 層ごとのパラメータ数 ---")
            emb = sum(p.numel() for p in model.embed.parameters())
            print(f"  embed (トークン埋め込み) : {emb:>8,}")
            for i, block in enumerate(model.blocks):
                bp   = sum(p.numel() for p in block.parameters())
                kind = "GQA      " if block.use_gqa else "DeltaNet "
                print(f"  Block[{i}] ({kind})    : {bp:>8,}")
            nf   = sum(p.numel() for p in model.norm_f.parameters())
            head = sum(p.numel() for p in model.lm_head.parameters())
            print(f"  norm_f                   : {nf:>8,}")
            print(f"  lm_head                  : {head:>8,}")

        else:
            from model import EvoTransformer, ModelConfig
            config = ModelConfig(c["vocab_size"], seq_len, c["n_layer"], c["n_head"], c["n_embd"])
            model  = EvoTransformer(config)
            model.load_state_dict(state, strict=False)

            total   = sum(p.numel() for p in model.parameters())
            buffers = sum(b.numel() for b in model.buffers())
            print(f"\n  学習可能パラメータ : {total:,}")
            print(f"  固定バッファ       : {buffers:,}  (推論マスク等)")
            print(f"  規模               : {total/1e9:.6f}B  ({total/1e6:.4f}M)")

            print("\n--- 層ごとのパラメータ数 ---")
            wte  = sum(p.numel() for p in model.transformer.wte.parameters())
            wpe  = sum(p.numel() for p in model.transformer.wpe.parameters())
            ln_f = sum(p.numel() for p in model.transformer.ln_f.parameters())
            head = sum(p.numel() for p in model.lm_head.parameters())
            print(f"  wte  (トークン埋め込み) : {wte:>8,}")
            print(f"  wpe  (位置埋め込み)     : {wpe:>8,}")
            for i, block in enumerate(model.transformer.h):
                bp = sum(p.numel() for p in block.parameters())
                print(f"  Block[{i}]               : {bp:>8,}")
            print(f"  ln_f (最終LayerNorm)    : {ln_f:>8,}")
            print(f"  lm_head                 : {head:>8,}")

        # 全レイヤー名と形状
        print("\n--- 全レイヤー一覧 ---")
        for name, param in model.named_parameters():
            print(f"  {name:<50} {str(list(param.shape)):<20} {param.numel():>8,}")

    except Exception as e:
        print(f"\n(詳細表示エラー: {e})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="chat_model.pth", help="確認するモデルのパス")
    args = parser.parse_args()
    check_model(args.path)
