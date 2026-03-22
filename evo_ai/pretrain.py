import torch
import torch.nn as nn
import os
import urllib.request
from model import EvoTransformer, ModelConfig
from utils import CharTokenizer

DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATASET_PATH = "tinyshakespeare.txt"

def download_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Downloading TinyShakespeare (~1MB)...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print(f"Saved to {DATASET_PATH}")
    else:
        print(f"Dataset already exists: {DATASET_PATH}")

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def main():
    # --- Config ---
    block_size  = 32
    batch_size  = 64
    max_iters   = 10000
    eval_interval = 500
    learning_rate = 1e-3
    n_layer = 2
    n_head  = 2
    n_embd  = 32

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

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"Train tokens: {len(train_data):,} / Val tokens: {len(val_data):,}")

    # --- Model ---
    # チェックポイントにconfigがある場合（upscale後など）はそちらを優先
    if os.path.exists("chat_model.pth"):
        ckpt = torch.load("chat_model.pth", map_location=device)
        if isinstance(ckpt, dict) and "config" in ckpt:
            c = ckpt["config"]
            n_layer = c["n_layer"]
            n_head  = c["n_head"]
            n_embd  = c["n_embd"]
            block_size = c["block_size"]
            print(f"Checkpoint config: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )
    model = EvoTransformer(config).to(device)

    if os.path.exists("chat_model.pth"):
        print("Found chat_model.pth — resuming from checkpoint...")
        try:
            ckpt = torch.load("chat_model.pth", map_location=device)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state)
        except Exception as e:
            print(f"Load failed ({e}). Starting from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nStarting pre-training for {max_iters} iterations...")
    print("Press Ctrl+C to stop early (model will be saved).\n")

    try:
        for step in range(1, max_iters + 1):
            model.train()
            xb, yb = get_batch(train_data, block_size, batch_size, device)

            logits = model(xb)                          # (B, T, vocab_size)
            B, T, V = logits.shape
            loss = nn.functional.cross_entropy(logits.view(B * T, V), yb.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    xv, yv = get_batch(val_data, block_size, batch_size, device)
                    logits_v = model(xv)
                    Bv, Tv, Vv = logits_v.shape
                    val_loss = nn.functional.cross_entropy(
                        logits_v.view(Bv * Tv, Vv), yv.view(Bv * Tv)
                    )
                print(f"Step {step:>5} | train loss: {loss.item():.4f} | val loss: {val_loss.item():.4f}")
                torch.save({"model": model.state_dict(), "config": {"vocab_size": config.vocab_size, "block_size": config.block_size, "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd}}, "chat_model.pth")
                print(f"         Saved to chat_model.pth")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    torch.save({"model": model.state_dict(), "config": {"vocab_size": config.vocab_size, "block_size": config.block_size, "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd}}, "chat_model.pth")
    print("\nPre-training complete. Model saved to chat_model.pth")
    print("You can now run chat_learn.py or auto_train.py to continue with evolutionary training.")

if __name__ == "__main__":
    main()
