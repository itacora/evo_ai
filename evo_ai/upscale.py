"""
Depth Upscaling: 学習済みモデルの層を複製してより深いモデルを作成する
元の重みを引き継ぎつつモデルを拡張する

使い方:
    python upscale.py                        # 2層 → 4層
    python upscale.py --layers 6            # 2層 → 6層
    python upscale.py --input my_model.pth  # 別のpthを使う
"""

import torch
import copy
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import EvoTransformer, ModelConfig
from utils import CharTokenizer

def upscale_model(src_path: str, dst_path: str, target_layers: int):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # 元モデルをロード（新形式: {"model":..., "config":...} / 旧形式: state_dict直接）
    checkpoint = torch.load(src_path, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config_dict = checkpoint["config"]
    else:
        # 旧形式: configがないのでデフォルト値を使う
        config_dict = {"block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 32}
        print("旧形式のチェックポイントを検出。デフォルトconfig (n_layer=2, n_head=2, n_embd=32) を使用します。")
    src_layers = config_dict["n_layer"]

    if target_layers <= src_layers:
        print(f"target_layers ({target_layers}) は現在の層数 ({src_layers}) より大きくしてください。")
        return

    print(f"層数: {src_layers} → {target_layers}")

    # 元モデルを構築して重みをロード
    tokenizer = CharTokenizer()
    src_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config_dict["block_size"],
        n_layer=src_layers,
        n_head=config_dict["n_head"],
        n_embd=config_dict["n_embd"],
    )
    src_model = EvoTransformer(src_config).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    src_model.load_state_dict(state)
    src_model.eval()

    # 新しいモデルを構築（層数だけ変更）
    dst_config = ModelConfig(
        vocab_size=src_config.vocab_size,
        block_size=src_config.block_size,
        n_layer=target_layers,
        n_head=src_config.n_head,
        n_embd=src_config.n_embd,
    )
    dst_model = EvoTransformer(dst_config).to(device)

    # 埋め込み・最終LayerNorm・lm_headはそのままコピー
    dst_model.transformer.wte.load_state_dict(src_model.transformer.wte.state_dict())
    dst_model.transformer.wpe.load_state_dict(src_model.transformer.wpe.state_dict())
    dst_model.transformer.ln_f.load_state_dict(src_model.transformer.ln_f.state_dict())
    dst_model.lm_head.load_state_dict(src_model.lm_head.state_dict())

    # 層を循環コピー: 元の層を繰り返し使って埋める
    # 例: 元2層 [0,1] → 4層 [0,1,0,1]
    src_blocks = list(src_model.transformer.h)
    for i, dst_block in enumerate(dst_model.transformer.h):
        src_block = src_blocks[i % src_layers]
        dst_block.load_state_dict(copy.deepcopy(src_block.state_dict()))

    # 保存
    torch.save({
        "model": dst_model.state_dict(),
        "config": {
            "vocab_size": dst_config.vocab_size,
            "block_size": dst_config.block_size,
            "n_layer": dst_config.n_layer,
            "n_head": dst_config.n_head,
            "n_embd": dst_config.n_embd,
        }
    }, dst_path)

    # パラメータ数を計算して表示
    src_params = sum(p.numel() for p in src_model.parameters())
    dst_params = sum(p.numel() for p in dst_model.parameters())
    print(f"パラメータ数: {src_params:,} → {dst_params:,}")
    print(f"保存先: {dst_path}")
    return dst_model, dst_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="chat_model.pth", help="元モデルのパス")
    parser.add_argument("--output", default="chat_model.pth", help="保存先のパス（デフォルトは上書き）")
    parser.add_argument("--layers", type=int, default=4,       help="拡張後の層数（デフォルト: 4）")
    args = parser.parse_args()

    src = args.input
    dst = args.output

    if not os.path.exists(src):
        print(f"モデルファイルが見つかりません: {src}")
        print("先に pretrain.py または finetune.py を実行してください。")
        sys.exit(1)

    # 上書きの場合はバックアップを作成
    if src == dst:
        backup = src.replace(".pth", f"_before_upscale.pth")
        import shutil
        shutil.copy(src, backup)
        print(f"バックアップ保存: {backup}")

    upscale_model(src, dst, args.layers)
    print("Upscaling 完了！続けて pretrain.py や finetune.py で追加学習できます。")
