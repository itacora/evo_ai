"""
HybridTransformer v2 の事前学習スクリプト
  RoPE + GQA + GatedDeltaNet + SwiGLU + RMSNorm

使い方:
    python pretrain_v2.py
    python pretrain_v2.py --layers 4 --embd 128
"""

import torch
import torch.nn as nn
import os
import signal
import argparse
import urllib.request
from model_v2 import HybridTransformer, ModelConfigV2
from utils import CharTokenizer

DATASET_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_PATH = "tinyshakespeare.txt"
SAVE_PATH    = "chat_model_v2.pth"

def download_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading TinyShakespeare (~1MB)...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
    else:
        print(f"Dataset: {DATASET_PATH}")

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i     : i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def save_ckpt(model, config, path):
    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size" : config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "n_layer"    : config.n_layer,
            "n_head"     : config.n_head,
            "n_kv_head"  : config.n_kv_head,
            "n_embd"     : config.n_embd,
            "ffn_mult"   : config.ffn_dim // config.n_embd,
            "rope_base"  : config.rope_base,
        }
    }, path)

def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device)
    c    = ckpt["config"]
    config = ModelConfigV2(
        vocab_size  = c["vocab_size"],
        max_seq_len = c["max_seq_len"],
        n_layer     = c["n_layer"],
        n_head      = c["n_head"],
        n_kv_head   = c["n_kv_head"],
        n_embd      = c["n_embd"],
        ffn_mult    = c.get("ffn_mult", 4),
        rope_base   = c.get("rope_base", 10000),
    )
    model = HybridTransformer(config).to(device)
    model.load_state_dict(ckpt["model"])
    return model, config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers",    type=int,   default=2)
    parser.add_argument("--embd",      type=int,   default=64)
    parser.add_argument("--heads",     type=int,   default=4)
    parser.add_argument("--kv_heads",  type=int,   default=2)
    parser.add_argument("--block",     type=int,   default=256,   help="学習時のコンテキスト長")
    parser.add_argument("--batch",     type=int,   default=32)
    parser.add_argument("--iters",     type=int,   default=5000)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--eval_every",type=int,   default=500)
    args = parser.parse_args()

    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Dataset ---
    download_dataset()
    tokenizer = CharTokenizer()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    data  = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n     = int(0.9 * len(data))
    train = data[:n]
    val   = data[n:]
    print(f"Train: {len(train):,} tokens / Val: {len(val):,} tokens")

    # --- Model ---
    if os.path.exists(SAVE_PATH):
        print(f"Resuming from {SAVE_PATH}...")
        try:
            model, config = load_ckpt(SAVE_PATH, device)
            print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, n_kv_head={config.n_kv_head}")
        except Exception as e:
            print(f"  Load failed ({e}). Starting fresh.")
            os.remove(SAVE_PATH)

    if not os.path.exists(SAVE_PATH):
        config = ModelConfigV2(
            vocab_size  = tokenizer.vocab_size,
            max_seq_len = 10000,
            n_layer     = args.layers,
            n_head      = args.heads,
            n_kv_head   = args.kv_heads,
            n_embd      = args.embd,
        )
        model = HybridTransformer(config).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}  ({total/1e6:.3f}M)")
    print(f"Layer pattern: ", end="")
    for i, b in enumerate(model.blocks):
        print("GQA" if b.use_gqa else "DeltaNet", end=" → " if i < len(model.blocks)-1 else "\n")

    # --- Train ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    stop = False
    def on_stop(sig, frame):
        nonlocal stop
        print("\nStop requested. Saving...")
        stop = True
    signal.signal(signal.SIGINT,  on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    print(f"\nTraining for {args.iters} steps (block_size={args.block}, batch={args.batch})...")
    print("Ctrl+C to stop and save.\n")

    for step in range(1, args.iters + 1):
        if stop:
            break

        model.train()
        xb, yb = get_batch(train, args.block, args.batch, device)
        logits  = model(xb)
        B, T, V = logits.shape
        loss    = nn.functional.cross_entropy(logits.view(B * T, V), yb.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                xv, yv  = get_batch(val, args.block, args.batch, device)
                lv      = model(xv)
                Bv,Tv,Vv = lv.shape
                val_loss = nn.functional.cross_entropy(lv.view(Bv*Tv,Vv), yv.view(Bv*Tv))
            print(f"Step {step:>6} | train_loss: {loss.item():.4f} | val_loss: {val_loss.item():.4f}")
            save_ckpt(model, config, SAVE_PATH)
            print(f"         Saved → {SAVE_PATH}")

    save_ckpt(model, config, SAVE_PATH)
    print(f"\nDone. Model saved to {SAVE_PATH}")
    print("次のステップ: finetune_v2.py で QA 学習、または check_model.py --path chat_model_v2.pth で確認")

if __name__ == "__main__":
    main()
