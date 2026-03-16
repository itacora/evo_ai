import torch
import copy
import random
from model import EvoTransformer, ModelConfig
from evolution import GeneticAlgorithm
from utils import CharTokenizer

def main():
    # 1. Setup
    tokenizer = CharTokenizer()
    print(f"Vocab Size: {tokenizer.vocab_size}")
    
    # Config for text model
    # Slightly larger than numeric model to handle characters
    config = ModelConfig(vocab_size=tokenizer.vocab_size, 
                         block_size=32, # Length of context
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
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Try to load existing chat model
    import os
    if os.path.exists("chat_model.pth"):
        print("Found existing 'chat_model.pth'. Resuming training...")
        try:
            state_dict = torch.load("chat_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            print("Successfully loaded model.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")
    else:
        print("No existing model found. Starting from scratch.")
    
    # Initialize GA
    # Population size doesn't need to be huge for interactive, 
    # but we need enough variation to show the user.
    # We will show the user prompts from a SUBSET of the population.
    ga = GeneticAlgorithm(model, pop_size=20, mutation_rate=0.05, mutation_power=0.02, device=device)
    
    print("\n--- Interactive Evolutionary Learning ---")
    print("You are the teacher. The AI starts knowing nothing.")
    print("You will see several candidate responses. Pick the best one (or the one you want to encourage).")
    print("Type 'exit' to quit/save.")
    
    generation = 0
    
    # Main Loop
    while True:
        generation += 1
        print(f"\n[Generation {generation}]")
        
        # 1. Get Input
        prompt_text = input("User (You): ")
        if prompt_text.lower() == 'exit':
            break
            
        # Prepare Input Tensor
        # Format: "User: <text>\nAI:" 
        # To keep it simple, we just feed the raw text and expect continuation.
        input_ids = tokenizer.encode(prompt_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # 2. Generate candidates
        # We need to sample from the population. 
        # Showing all 20 is too much. Let's show 5 candidates.
        # We pick 5 random indices from the population to present.
        candidate_indices = random.sample(range(ga.pop_size), 5)
        candidates = []
        
        print("\nCandidates:")
        for i, idx in enumerate(candidate_indices):
            # Load individual
            state = ga.population[idx]
            model.load_state_dict(state)
            
            # Generate
            # We enforce a limit
            with torch.no_grad():
                # For very early models, they might output garbage or end immediately.
                # We force generate 20 tokens.
                gen_out = model.generate(input_tensor, max_new_tokens=20)
                
            # Decode
            # Only look at the new tokens
            new_tokens = gen_out[0, len(input_ids):].tolist()
            text_out = tokenizer.decode(new_tokens)
            
            print(f"[{i+1}] {text_out}")
            candidates.append((idx, text_out))
            
        # 3. User Selection
        while True:
            try:
                choice = input("\nSelect best (1-5): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < 5:
                    break
                print("Please enter 1-5.")
            except ValueError:
                print("Invalid input.")
        
        best_pop_idx = candidates[choice_idx][0]
        print(f"Selected: '{candidates[choice_idx][1]}'")
        
        # 4. Evolution Step
        # We construct a 'score' vector where the winner gets 1.0, others 0.0?
        # Standard ES needs a gradient or population distribution.
        # Our `evolution.py` expects a list of (index, score).
        # We can give the winner score 1.0, and maybe others 0.0.
        # Or maybe we give the others shown a slightly lower score if they were "Okay"? 
        # Simplify: Winner takes all. The winner becomes the seed for the next generation elites.
        
        scores = []
        for i in range(ga.pop_size):
            if i == best_pop_idx:
                scores.append((i, 1.0))
            else:
                # If they were shown but rejected, maybe negative score? 
                # Or just 0.
                scores.append((i, 0.0))
                
        # Run Evolution
        # This will clone the winner and mutate it to fill the population.
        ga.evolve(scores)
        
        # Optional: Save every 5 gens
        if generation % 5 == 0:
            torch.save(ga.population[0], "chat_model.pth")
            print("Model saved to chat_model.pth")

    # Save on exit
    torch.save(ga.population[0], "chat_model.pth")
    print("Model saved. Goodbye!")

if __name__ == "__main__":
    main()
