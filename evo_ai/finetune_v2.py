import torch
import torch.nn as nn
import os
import signal
import random
from model_v2 import HybridTransformer, ModelConfigV2
from utils import CharTokenizer
from pretrain_v2 import save_ckpt, load_ckpt, SAVE_PATH

# --- QAデータセット ---
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
    text   = f"Q:{q}\nA:{a}\n"
    tokens = tokenizer.encode(text)
    length = block_size + 1
    if len(tokens) >= length:
        tokens = tokens[:length]
    else:
        tokens = tokens + [0] * (length - len(tokens))
    return tokens

def get_batch(qa_pairs, tokenizer, block_size, batch_size, device):
    xs, ys = [], []
    for _ in range(batch_size):
        q, a   = random.choice(qa_pairs)
        tokens = make_pair_tokens(q, a, tokenizer, block_size)
        xs.append(tokens[:block_size])
        ys.append(tokens[1:block_size + 1])
    return (torch.tensor(xs, dtype=torch.long).to(device),
            torch.tensor(ys, dtype=torch.long).to(device))

def generate_response(model, tokenizer, prompt, device, max_new_tokens=20):
    prefix = f"Q:{prompt}\nA:"
    ids    = tokenizer.encode(prefix)
    idx    = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0, len(ids):].tolist()).split("\n")[0]

def main():
    batch_size    = 32
    max_iters     = 5000
    eval_interval = 500
    learning_rate = 1e-3
    block_size    = 64   # v2 は長いコンテキストに対応しているので少し広げる

    # --- Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    tokenizer = CharTokenizer()

    # --- Model 読み込み ---
    if os.path.exists(SAVE_PATH):
        print(f"Loading {SAVE_PATH}...")
        try:
            model, config = load_ckpt(SAVE_PATH, device)
            print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, n_kv_head={config.n_kv_head}")
            print("  Fine-tuning from checkpoint.")
        except Exception as e:
            print(f"  Load failed ({e}). Starting from scratch.")
            model = None
    else:
        model = None

    if model is None:
        config = ModelConfigV2(vocab_size=tokenizer.vocab_size)
        model  = HybridTransformer(config).to(device)
        print("Starting from scratch.")

    # --- Fine-tuning ---
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    stop_requested = False
    def handle_stop(sig, frame):
        nonlocal stop_requested
        print("\nStop requested. Saving after this step...")
        stop_requested = True
    signal.signal(signal.SIGINT,  handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print(f"\nStarting fine-tuning for {max_iters} iterations...")
    print("Press Ctrl+C to stop and save.\n")

    for step in range(1, max_iters + 1):
        if stop_requested:
            break

        model.train()
        xb, yb  = get_batch(QA_PAIRS, tokenizer, block_size, batch_size, device)
        logits  = model(xb)
        B, T, V = logits.shape
        loss    = nn.functional.cross_entropy(logits.view(B * T, V), yb.view(B * T))

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
            save_ckpt(model, config, SAVE_PATH)
            print(f"  Saved to {SAVE_PATH}\n")

    save_ckpt(model, config, SAVE_PATH)
    print(f"Fine-tuning complete. Saved to {SAVE_PATH}")

    print("\n--- Final check ---")
    model.eval()
    for q, a in QA_PAIRS:
        resp = generate_response(model, tokenizer, q, device)
        print(f"Q:{q}")
        print(f"  Expected : {a}")
        print(f"  Got      : {resp}")

if __name__ == "__main__":
    main()
