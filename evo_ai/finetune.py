import torch
import torch.nn as nn
import os
import signal
import random
from model import EvoTransformer, ModelConfig
from utils import CharTokenizer

# --- QAデータセット ---
# フォーマット: "Q:<質問>\nA:<回答>\n"
# block_size=32 の制約上、1ペアは32文字以内に収めること
QA_PAIRS = [
    ("Hello",        "Hello! How are you?"),
    ("Hi",           "Hi there!"),
    ("Good morning", "Good morning!"),
    ("Good evening", "Good evening!"),
    ("How are you",  "I am fine, thank you!"),
    ("What is up",   "Not much, you?"),
    ("Hey",          "Hey! Good to see you."),
    ("Goodbye",      "Goodbye! Take care."),
    ("See you",      "See you later!"),
    ("Thank you",    "You are welcome!"),
]

def make_pair_tokens(q, a, tokenizer, block_size):
    """1ペアを block_size+1 トークンに変換（パディングあり）"""
    text = f"Q:{q}\nA:{a}\n"
    tokens = tokenizer.encode(text)
    # block_size+1 トークン必要（x: 0..block_size-1, y: 1..block_size）
    length = block_size + 1
    if len(tokens) >= length:
        tokens = tokens[:length]
    else:
        tokens = tokens + [0] * (length - len(tokens))
    return tokens

def get_batch(qa_pairs, tokenizer, block_size, batch_size, device):
    """ペアをランダムに選んで常に位置0から始まるバッチを作る"""
    xs, ys = [], []
    for _ in range(batch_size):
        q, a = random.choice(qa_pairs)
        tokens = make_pair_tokens(q, a, tokenizer, block_size)
        xs.append(tokens[:block_size])
        ys.append(tokens[1:block_size + 1])
    x = torch.tensor(xs, dtype=torch.long).to(device)
    y = torch.tensor(ys, dtype=torch.long).to(device)
    return x, y

def generate_response(model, tokenizer, prompt, device, max_new_tokens=20):
    """Q:<prompt>\nA: を入力してモデルの応答を生成する"""
    prefix = f"Q:{prompt}\nA:"
    ids = tokenizer.encode(prefix)
    idx = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens)
    new_tokens = out[0, len(ids):].tolist()
    return tokenizer.decode(new_tokens).split("\n")[0]  # 最初の行だけ返す

def main():
    # --- Config ---
    block_size    = 32
    batch_size    = 32
    max_iters     = 600000
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

    # --- Tokenizer ---
    tokenizer = CharTokenizer()

    # --- Model ---
    # チェックポイントに config が含まれている場合（upscale後など）はそちらを優先
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
        print("Loading chat_model.pth...")
        try:
            ckpt = torch.load("chat_model.pth", map_location=device)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state)
            print("Loaded. Fine-tuning from checkpoint.")
        except Exception as e:
            print(f"Load failed ({e}). Starting from scratch.")
    else:
        print("No checkpoint found. Starting from scratch.")

    # --- Fine-tuning ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    stop_requested = False

    def handle_stop(sig, frame):
        nonlocal stop_requested
        print("\nStop requested. Saving after this step...")
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print(f"\nStarting fine-tuning for {max_iters} iterations...")
    print("Press Ctrl+C to stop and save.\n")

    for step in range(1, max_iters + 1):
        if stop_requested:
            break

        model.train()
        xb, yb = get_batch(QA_PAIRS, tokenizer, block_size, batch_size, device)
        logits = model(xb)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, V), yb.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            model.eval()
            print(f"Step {step:>5} | loss: {loss.item():.4f}")
            print("  Sample responses:")
            for q, _ in QA_PAIRS[:3]:
                resp = generate_response(model, tokenizer, q, device)
                print(f"    Q:{q} -> A:{resp}")
            torch.save({"model": model.state_dict(), "config": {"vocab_size": config.vocab_size, "block_size": config.block_size, "n_layer": config.n_layer, "n_head": config.n_head, "n_embd": config.n_embd}}, "chat_model.pth")
            print("  Saved to chat_model.pth\n")

    torch.save(model.state_dict(), "chat_model.pth")
    print("Fine-tuning complete. Model saved to chat_model.pth")

    # --- 最終確認 ---
    print("\n--- Final check ---")
    model.eval()
    for q, a in QA_PAIRS:
        resp = generate_response(model, tokenizer, q, device)
        print(f"Q:{q}")
        print(f"  Expected : {a}")
        print(f"  Got      : {resp}")

if __name__ == "__main__":
    main()
