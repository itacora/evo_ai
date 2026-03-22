import torch
import random
import time
import signal
from model_v2 import HybridTransformer
from evolution import GeneticAlgorithm
from utils import CharTokenizer
from judge import LLMJudge
from pretrain_v2 import save_ckpt, load_ckpt, SAVE_PATH

PROMPTS = [
    "Hello",
    "Hi",
    "How are you?",
    "Good morning",
    "Say something"
]

def main():
    import os

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = CharTokenizer()

    # モデル読み込み（なければゼロから）
    if os.path.exists(SAVE_PATH):
        print(f"Resuming from {SAVE_PATH}...")
        try:
            model, config = load_ckpt(SAVE_PATH, device)
            print(f"  n_layer={config.n_layer}, n_embd={config.n_embd}, n_head={config.n_head}, n_kv_head={config.n_kv_head}")
        except Exception as e:
            print(f"Failed to load ({e}). Starting fresh.")
            model = None
    else:
        model = None

    if model is None:
        from model_v2 import ModelConfigV2
        config = ModelConfigV2(vocab_size=tokenizer.vocab_size)
        model  = HybridTransformer(config).to(device)

    ga    = GeneticAlgorithm(model, pop_size=20, mutation_rate=0.05, mutation_power=0.02, device=device)
    judge = LLMJudge(device=device.type)

    print("\n--- Automated Evolutionary Learning (v2) ---")
    print("Judge Model will supervise the training.")
    print("Press Ctrl+C to stop and save.")

    stop_requested = False
    def handle_stop(sig, frame):
        nonlocal stop_requested
        print("\nStop requested. Saving after this generation...")
        stop_requested = True
    signal.signal(signal.SIGINT,  handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    generation = 0
    start_time = time.time()

    while not stop_requested:
        generation += 1

        prompt_text  = random.choice(PROMPTS)
        input_ids    = tokenizer.encode(prompt_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        candidate_indices = random.sample(range(ga.pop_size), 5)
        candidates, cand_texts = [], []

        for idx in candidate_indices:
            model.load_state_dict(ga.population[idx])
            with torch.no_grad():
                gen_out = model.generate(input_tensor, max_new_tokens=20)
            text_out = tokenizer.decode(gen_out[0, len(input_ids):].tolist())
            candidates.append((idx, text_out))
            cand_texts.append(text_out)

        best_local_idx = judge.evaluate(prompt_text, cand_texts)
        best_pop_idx   = candidates[best_local_idx][0]
        selected_text  = candidates[best_local_idx][1]

        print(f"Gen {generation} | Prompt: '{prompt_text}' | Selected: '{selected_text}'")

        scores = [(i, 1.0 if i == best_pop_idx else 0.0) for i in range(ga.pop_size)]
        ga.evolve(scores)

        if generation % 10 == 0:
            model.load_state_dict(ga.population[0])
            save_ckpt(model, config, SAVE_PATH)
            print(f"Saved (Time: {time.time()-start_time:.1f}s)")

    model.load_state_dict(ga.population[0])
    save_ckpt(model, config, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH} (Total generations: {generation})")

if __name__ == "__main__":
    main()
