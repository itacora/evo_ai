"""
Knowledge Distillation: Qwen2.5-0.5B → HybridTransformer v2
============================================================

【2つの蒸留方式】

  ① CPT 蒸留 (Continued Pre-Training Distillation)
      DeepSeek が OpenAI API から疑われた方式に近い。
      Teacher に自由文を大量生成させ、それをコーパスとして事前学習。
      → 言語能力そのものを転移する

  ② SFT 蒸留 (Supervised Fine-Tuning Distillation)
      DeepSeek-R1 公式論文の小モデル蒸留に近い方式。
      Teacher に Q&A 応答を生成させ、そのペアで Student を学習。
      → 特定の応答パターンを習得させる

使い方:
    python distill_v2.py --mode both       # CPT → SFT の順で両方実行（推奨）
    python distill_v2.py --mode cpt        # CPT 蒸留のみ
    python distill_v2.py --mode sft        # SFT 蒸留のみ
    python distill_v2.py --mode both --gen_only   # データ生成のみ（学習しない）
    python distill_v2.py --mode both --train_only # 学習のみ（既存データ使用）
"""

import torch
import torch.nn as nn
import os
import signal
import json
import random
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_v2 import HybridTransformer, ModelConfigV2
from utils import CharTokenizer
from pretrain_v2 import save_ckpt, load_ckpt, SAVE_PATH

TEACHER_MODEL   = "Qwen/Qwen2.5-0.5B-Instruct"
CPT_DATA_PATH   = "distill_cpt.txt"     # CPT 用テキストコーパス
SFT_DATA_PATH   = "distill_sft.jsonl"   # SFT 用 Q&A ペア
MAX_RESP_CHARS  = 300                   # 応答の最大文字数
BLOCK_SIZE      = 128                   # 学習時のコンテキスト長


# =============================================================================
# CPT 用プロンプト集（自由文生成）
# =============================================================================
CPT_PROMPTS = [
    # 日常・会話
    "Write a short friendly conversation between two people meeting for the first time.",
    "Write a short paragraph about what makes a good morning routine.",
    "Write a short story about someone helping a stranger.",
    "Describe what a perfect day looks like.",
    "Write a friendly letter to a neighbor.",

    # 知識・説明
    "Explain what the sun is in simple words.",
    "Describe how rain forms in simple terms.",
    "Explain why exercise is good for your health.",
    "Write a short explanation of what a computer does.",
    "Describe what the ocean looks like.",

    # 感情・思考
    "Write about what it feels like to be happy.",
    "Describe what it means to be a good friend.",
    "Write a short reflection on the importance of kindness.",
    "Describe a moment when someone felt proud of themselves.",
    "Write about what makes people laugh.",

    # 日常シーン
    "Describe a busy city street scene.",
    "Write about what happens in a small cafe in the morning.",
    "Describe the feeling of walking in a park in autumn.",
    "Write a short description of a cozy home on a rainy day.",
    "Describe a farmer's market on a Saturday morning.",

    # 自然・科学
    "Write a short paragraph about stars and the night sky.",
    "Describe what it is like to be near the ocean.",
    "Write about the changing of seasons.",
    "Describe how plants grow from seeds.",
    "Write about the sounds of a forest.",

    # 社会・人間
    "Write about what makes a community strong.",
    "Describe the importance of listening to others.",
    "Write about how people learn new skills.",
    "Describe what teamwork looks like in practice.",
    "Write about why people enjoy music.",
]

# =============================================================================
# SFT 用プロンプト集（Q&A 応答生成）
# =============================================================================
SFT_PROMPTS = [
    "Hello", "Hi there", "Good morning", "Good evening", "Good afternoon",
    "Hey", "How are you?", "How do you do?", "Nice to meet you", "What's up?",
    "How's it going?", "Howdy", "Goodbye", "See you later", "Bye",
    "Take care", "Have a good day", "Talk to you later", "Thank you",
    "Thanks a lot", "I appreciate it", "Thank you so much",
    "What is your name?", "Who are you?", "What can you do?", "Are you an AI?",
    "Can you help me?", "I need help", "What do you think?", "That's great",
    "Really?", "Sounds good", "Sure", "Of course", "No problem",
    "You're welcome", "Excuse me", "Sorry", "That's okay",
    "Tell me something interesting", "Say something nice",
]


# =============================================================================
# Teacher の読み込みとテキスト生成ユーティリティ
# =============================================================================
def load_teacher(device_str):
    print(f"  Teacher モデル ({TEACHER_MODEL}) を読み込み中...")
    tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.float16 if device_str == "mps" else torch.float32,
        trust_remote_code=True,
    ).to(device_str)
    mdl.eval()
    print("  Teacher 読み込み完了\n")
    return tok, mdl


def unload_teacher(mdl, device_str):
    del mdl
    if device_str == "mps":
        torch.mps.empty_cache()
    elif device_str == "cuda":
        torch.cuda.empty_cache()


def to_safe_ascii(text, char_tok):
    """CharTokenizer が扱えない文字を除去し、ASCII のみに絞る"""
    safe = "".join(c for c in text if c in char_tok.stoi)
    return safe or "I see."


def teacher_generate(tok, mdl, system_prompt, user_prompt, device_str,
                     max_new_tokens=120, temperature=0.8):
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_prompt},
    ]
    text   = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(device_str)
    with torch.no_grad():
        out_ids = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    new_ids  = out_ids[0, inputs["input_ids"].shape[1]:]
    response = tok.decode(new_ids, skip_special_tokens=True).strip()
    return response[:MAX_RESP_CHARS]


# =============================================================================
# ① CPT 蒸留 — データ生成
# =============================================================================
def generate_cpt_data(output_path, device_str):
    """Qwen に自由文を生成させ、テキストコーパスとして保存する"""
    print(f"\n[CPT Step 1] 自由文コーパスを生成中 ({len(CPT_PROMPTS)} プロンプト)...")
    tok, mdl = load_teacher(device_str)
    char_tok = CharTokenizer()

    corpus_lines = []
    for i, prompt in enumerate(CPT_PROMPTS):
        system = (
            "You are a natural writer. "
            "Write clear, simple English sentences. "
            "Use only standard ASCII characters. "
            "Write 2-4 sentences only."
        )
        raw  = teacher_generate(tok, mdl, system, prompt, device_str,
                                max_new_tokens=100, temperature=0.85)
        safe = to_safe_ascii(raw, char_tok)
        corpus_lines.append(safe)
        print(f"  [{i+1:>2}/{len(CPT_PROMPTS)}] {prompt[:40]:<40} → {safe[:60]}...")

    unload_teacher(mdl, device_str)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_lines) + "\n")

    total_chars = sum(len(l) for l in corpus_lines)
    print(f"\n  CPT コーパス: {len(corpus_lines)} 段落, {total_chars:,} 文字 → {output_path}\n")
    return corpus_lines


# =============================================================================
# ① CPT 蒸留 — 学習
# =============================================================================
def get_cpt_batch(data_tensor, block_size, batch_size, device):
    ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
    x  = torch.stack([data_tensor[i     : i + block_size    ] for i in ix])
    y  = torch.stack([data_tensor[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train_cpt(corpus_lines, device, iters, batch_size, lr, eval_interval):
    tokenizer = CharTokenizer()

    # コーパス全体を 1 本のトークン列に結合
    full_text   = "\n".join(corpus_lines)
    data_tensor = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)
    print(f"[CPT Step 2] コーパストークン数: {len(data_tensor):,}")

    model, config = _load_or_create_model(device, tokenizer)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Student パラメータ数: {total:,}  ({total/1e6:.3f}M)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    stop = _setup_stop_signal()

    print(f"[CPT Step 2] 事前学習を開始 ({iters} steps)...")
    print("  Ctrl+C で安全に停止\n")

    for step in range(1, iters + 1):
        if stop[0]:
            break
        model.train()
        xb, yb  = get_cpt_batch(data_tensor, BLOCK_SIZE, batch_size, device)
        logits  = model(xb)
        B, T, V = logits.shape
        loss    = nn.functional.cross_entropy(logits.view(B * T, V), yb.view(B * T))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0:
            print(f"Step {step:>5} | CPT loss: {loss.item():.4f}")
            save_ckpt(model, config, SAVE_PATH)
            print(f"  → {SAVE_PATH} に保存\n")

    save_ckpt(model, config, SAVE_PATH)
    print(f"[CPT Step 2] 完了。モデルを {SAVE_PATH} に保存\n")


# =============================================================================
# ② SFT 蒸留 — データ生成
# =============================================================================
def generate_sft_data(output_path, device_str):
    """Qwen に各プロンプトへの短い応答を生成させ JSONL に保存する"""
    print(f"\n[SFT Step 1] Q&A データを生成中 ({len(SFT_PROMPTS)} プロンプト)...")
    tok, mdl = load_teacher(device_str)
    char_tok = CharTokenizer()

    records = []
    system  = (
        "You are a friendly conversational AI. "
        "Reply in ONE SHORT sentence (under 12 words). "
        "Use only standard ASCII characters."
    )
    for i, prompt in enumerate(SFT_PROMPTS):
        raw  = teacher_generate(tok, mdl, system, prompt, device_str,
                                max_new_tokens=40, temperature=0.7)
        safe = to_safe_ascii(raw, char_tok)
        # 1 文目だけ取る
        safe = safe.split(".")[0].strip()
        if not safe:
            safe = "Hello!"
        records.append({"prompt": prompt, "response": safe})
        print(f"  [{i+1:>2}/{len(SFT_PROMPTS)}] Q: {prompt:<28} → A: {safe}")

    unload_teacher(mdl, device_str)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  SFT データ {len(records)} 件 → {output_path}\n")
    return records


# =============================================================================
# ② SFT 蒸留 — 学習
# =============================================================================
def make_qa_tokens(prompt, response, tokenizer, block_size):
    text   = f"Q:{prompt}\nA:{response}\n"
    tokens = tokenizer.encode(text)
    length = block_size + 1
    if len(tokens) >= length:
        tokens = tokens[:length]
    else:
        tokens = tokens + [0] * (length - len(tokens))
    return tokens


def get_sft_batch(records, tokenizer, block_size, batch_size, device):
    xs, ys = [], []
    for _ in range(batch_size):
        r      = random.choice(records)
        tokens = make_qa_tokens(r["prompt"], r["response"], tokenizer, block_size)
        xs.append(tokens[:block_size])
        ys.append(tokens[1:block_size + 1])
    return (torch.tensor(xs, dtype=torch.long).to(device),
            torch.tensor(ys, dtype=torch.long).to(device))


def train_sft(records, device, iters, batch_size, lr, eval_interval):
    tokenizer = CharTokenizer()
    model, config = _load_or_create_model(device, tokenizer)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Student パラメータ数: {total:,}  ({total/1e6:.3f}M)\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stop = _setup_stop_signal()

    sample_prompts = [r["prompt"] for r in records[:4]]

    print(f"[SFT Step 2] ファインチューニングを開始 ({iters} steps)...")
    print("  Ctrl+C で安全に停止\n")

    for step in range(1, iters + 1):
        if stop[0]:
            break
        model.train()
        xb, yb  = get_sft_batch(records, tokenizer, BLOCK_SIZE, batch_size, device)
        logits  = model(xb)
        B, T, V = logits.shape
        loss    = nn.functional.cross_entropy(
            logits.view(B * T, V), yb.view(B * T), ignore_index=0
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0:
            model.eval()
            print(f"Step {step:>5} | SFT loss: {loss.item():.4f}")
            print("  サンプル応答:")
            for prompt in sample_prompts:
                student_resp = _student_reply(model, tokenizer, prompt, device)
                teacher_resp = next((r["response"] for r in records if r["prompt"] == prompt), "?")
                print(f"    Q: {prompt}")
                print(f"       Teacher: {teacher_resp}")
                print(f"       Student: {student_resp}")
            save_ckpt(model, config, SAVE_PATH)
            print(f"  → {SAVE_PATH} に保存\n")

    save_ckpt(model, config, SAVE_PATH)
    print(f"[SFT Step 2] 完了。モデルを {SAVE_PATH} に保存")

    # 最終評価
    model.eval()
    print("\n=== 最終評価 ===")
    correct = 0
    for r in records:
        resp  = _student_reply(model, tokenizer, r["prompt"], device)
        match = any(w in resp.lower() for w in r["response"].lower().split()[:2] if len(w) > 2)
        if match:
            correct += 1
        mark = "✓" if match else " "
        print(f"  [{mark}] Q: {r['prompt']:<28} | T: {r['response']:<35} | S: {resp}")
    print(f"\n  一致率: {correct}/{len(records)} ({100*correct/len(records):.0f}%)\n")


# =============================================================================
# 共通ユーティリティ
# =============================================================================
def _load_or_create_model(device, tokenizer):
    if os.path.exists(SAVE_PATH):
        print(f"  既存モデルを読み込み: {SAVE_PATH}")
        try:
            model, config = load_ckpt(SAVE_PATH, device)
            print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}")
            return model, config
        except Exception as e:
            print(f"  読み込み失敗 ({e}). 新規作成します。")
    config = ModelConfigV2(vocab_size=tokenizer.vocab_size)
    model  = HybridTransformer(config).to(device)
    return model, config


def _setup_stop_signal():
    stop = [False]
    def handle(sig, frame):
        print("\n停止リクエスト。このステップ後に保存します...")
        stop[0] = True
    signal.signal(signal.SIGINT,  handle)
    signal.signal(signal.SIGTERM, handle)
    return stop


def _student_reply(model, tokenizer, prompt, device, max_new_tokens=40):
    prefix = f"Q:{prompt}\nA:"
    ids    = tokenizer.encode(prefix)
    idx    = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0, len(ids):].tolist())
    return text.split("\n")[0].strip()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation (CPT + SFT): Qwen2.5-0.5B → HybridTransformer v2"
    )
    parser.add_argument("--mode",       default="both",
                        choices=["cpt", "sft", "both"],
                        help="蒸留モード: cpt / sft / both (default: both)")
    parser.add_argument("--gen_only",   action="store_true", help="データ生成のみ（学習しない）")
    parser.add_argument("--train_only", action="store_true", help="学習のみ（既存データを使用）")
    parser.add_argument("--cpt_data",   default=CPT_DATA_PATH)
    parser.add_argument("--sft_data",   default=SFT_DATA_PATH)
    parser.add_argument("--cpt_iters",  type=int,   default=3000,  help="CPT 学習ステップ数")
    parser.add_argument("--sft_iters",  type=int,   default=5000,  help="SFT 学習ステップ数")
    parser.add_argument("--batch",      type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int,   default=500)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device     = torch.device("mps")
        device_str = "mps"
        print("Using MPS (Apple GPU)")
    else:
        device     = torch.device("cpu")
        device_str = "cpu"
        print("Using CPU")

    print("\n" + "=" * 60)
    print("  Knowledge Distillation: Qwen2.5-0.5B → HybridTransformer v2")
    print(f"  モード: {args.mode.upper()}")
    print("=" * 60)

    run_cpt = args.mode in ("cpt", "both")
    run_sft = args.mode in ("sft", "both")

    # ─── CPT ───────────────────────────────────────────
    if run_cpt:
        print("\n" + "─" * 40)
        print("  ① CPT 蒸留 (Continued Pre-Training)")
        print("─" * 40)

        if not args.train_only:
            cpt_lines = generate_cpt_data(args.cpt_data, device_str)
        else:
            if not os.path.exists(args.cpt_data):
                print(f"エラー: {args.cpt_data} が見つかりません。--train_only を外してください。")
                return
            with open(args.cpt_data, "r", encoding="utf-8") as f:
                cpt_lines = [l.rstrip() for l in f if l.strip()]
            print(f"[CPT Step 1 スキップ] 既存データを使用: {len(cpt_lines)} 段落")

        if not args.gen_only:
            train_cpt(cpt_lines, device, args.cpt_iters, args.batch, args.lr, args.eval_every)

    # ─── SFT ───────────────────────────────────────────
    if run_sft:
        print("\n" + "─" * 40)
        print("  ② SFT 蒸留 (Supervised Fine-Tuning)")
        print("─" * 40)

        if not args.train_only:
            sft_records = generate_sft_data(args.sft_data, device_str)
        else:
            if not os.path.exists(args.sft_data):
                print(f"エラー: {args.sft_data} が見つかりません。--train_only を外してください。")
                return
            with open(args.sft_data, "r", encoding="utf-8") as f:
                sft_records = [json.loads(l) for l in f]
            print(f"[SFT Step 1 スキップ] 既存データを使用: {len(sft_records)} 件")

        if not args.gen_only:
            train_sft(sft_records, device, args.sft_iters, args.batch, args.lr, args.eval_every)

    if args.gen_only:
        print("\nデータ生成完了。学習するには --train_only オプションで再実行してください。")


if __name__ == "__main__":
    main()
