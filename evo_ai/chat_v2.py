"""
HybridTransformer v2 とインタラクティブに会話するスクリプト

使い方:
    python chat_v2.py
    python chat_v2.py --path chat_model_v2.pth --max_tokens 50
"""

import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from model_v2 import HybridTransformer
from utils import CharTokenizer
from pretrain_v2 import load_ckpt, SAVE_PATH


STOP_CHARS = set("\n")
SENT_END   = set(".!?")


def generate_response(model, tokenizer, prompt, device, max_new_tokens=120, temperature=1.0):
    prefix = f"Q:{prompt}\nA:"
    ids    = tokenizer.encode(prefix)
    idx    = torch.tensor([ids], dtype=torch.long).to(device)

    generated_chars = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.max_seq_len:]
            logits   = model(idx_cond)[:, -1, :]

            if temperature == 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs   = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            ch  = tokenizer.decode([next_id.item()])

            # \n で即停止
            if ch in STOP_CHARS:
                break

            generated_chars.append(ch)

            # 文末記号（.!?）で即停止 — 小さいモデルはここ以降がゴミになりやすい
            if ch in SENT_END:
                break

    response = "".join(generated_chars).strip()

    # 万が一 "Q:" が混入していたらそこで切る（保険）
    if "Q:" in response:
        response = response[:response.index("Q:")].strip()

    # 文末記号より後ろのゴミを除去（.!? の最後の出現位置まで残す）
    last_sent_end = max((response.rfind(c) for c in SENT_END), default=-1)
    if last_sent_end != -1:
        response = response[:last_sent_end + 1]

    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",       default=SAVE_PATH, help="モデルのパス")
    parser.add_argument("--max_tokens", type=int,   default=120, help="最大生成トークン数")
    parser.add_argument("--temp",       type=float, default=0.8, help="温度 (0=greedy, 高=多様)")
    args = parser.parse_args()

    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --- Model 読み込み ---
    if not os.path.exists(args.path):
        print(f"エラー: {args.path} が見つかりません。")
        print("先に pretrain_v2.py または finetune_v2.py を実行してください。")
        sys.exit(1)

    print(f"Loading {args.path} ...")
    model, config = load_ckpt(args.path, device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, "
          f"n_head={config.n_head}, n_kv_head={config.n_kv_head}")
    print(f"  Parameters: {total:,}  ({total/1e6:.3f}M)")
    print(f"  Temperature: {args.temp}  Max tokens: {args.max_tokens}")
    print()
    print("=" * 50)
    print("  HybridTransformer v2 Chat")
    print("  'quit' または 'exit' で終了")
    print("=" * 50)

    tokenizer = CharTokenizer()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("終了します。")
            break

        response = generate_response(
            model, tokenizer, user_input, device,
            max_new_tokens=args.max_tokens,
            temperature=args.temp
        )
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
