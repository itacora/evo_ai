import torch
import copy
import random
import time
from model import EvoTransformer, ModelConfig
from evolution import GeneticAlgorithm
from utils import CharTokenizer
from judge import LLMJudge

def main():
    # 1. Setup
    tokenizer = CharTokenizer()
    
    config = ModelConfig(vocab_size=tokenizer.vocab_size, 
                         block_size=32, 
                         n_layer=2, 
                         n_head=2, 
                         n_embd=32)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Initialize Template
    model = EvoTransformer(config).to(device)
    
    # Load existing if available
    import os
    if os.path.exists("chat_model.pth"):
        print("Resuming from chat_model.pth...")
        try:
            model.load_state_dict(torch.load("chat_model.pth", map_location=device))
        except:
            print("Failed to load. Starting fresh.")
    
    # Initialize GA (Small population for speed with Judge)
    ga = GeneticAlgorithm(model, pop_size=20, mutation_rate=0.05, mutation_power=0.02, device=device)
    
    # Initialize Judge
    # Judge runs on the same device? Or CPU if memory tight?
    # Qwen 0.5B is tiny, can fit on MPS easily with the EvoAI.
    judge = LLMJudge(device=device.type) 
    
    # Training Prompts
    # We want the EvoAI to learn basic greetings.
    prompts = [
        "Hello",
        "Hi",
        "How are you?",
        "Good morning",
        "Say something"
    ]
    
    print("\n--- Automated Evolutionary Learning ---")
    print("Judge Model will supervise the training.")
    print("Press Ctrl+C to stop.")
    
    generation = 0
    start_time = time.time()
    
    while True:
        generation += 1
        
        # 1. Pick a prompt
        prompt_text = random.choice(prompts)
        
        # 2. Generate Candidates
        input_ids = tokenizer.encode(prompt_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        candidate_indices = random.sample(range(ga.pop_size), 5)
        candidates = [] # list of (idx, text)
        cand_texts = [] # list of text only for judge
        
        for i, idx in enumerate(candidate_indices):
            state = ga.population[idx]
            model.load_state_dict(state)
            with torch.no_grad():
                gen_out = model.generate(input_tensor, max_new_tokens=20)
            new_tokens = gen_out[0, len(input_ids):].tolist()
            text_out = tokenizer.decode(new_tokens)
            candidates.append((idx, text_out))
            cand_texts.append(text_out)
        
        # 3. Judge Selection
        best_local_idx = judge.evaluate(prompt_text, cand_texts)
        best_pop_idx = candidates[best_local_idx][0]
        selected_text = candidates[best_local_idx][1]
        
        print(f"Gen {generation} | Prompt: '{prompt_text}' | Selected: '{selected_text}'")
        
        # 4. Evolution
        scores = []
        for i in range(ga.pop_size):
            if i == best_pop_idx:
                scores.append((i, 1.0))
            else:
                scores.append((i, 0.0))
        
        ga.evolve(scores)
        
        # Save occasionally
        if generation % 10 == 0:
            torch.save(ga.population[0], "chat_model.pth")
            print(f"Saved (Time: {time.time()-start_time:.1f}s)")

if __name__ == "__main__":
    main()
