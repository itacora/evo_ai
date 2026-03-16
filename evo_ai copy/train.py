import torch
import time
from model import EvoTransformer, ModelConfig
from task import ReverseTask
from evolution import GeneticAlgorithm

def main():
    # Configuration
    # Small model for speed and demonstration
    vocab_size = 10 # 0..8 data, 9 SEP
    block_size = 16 
    n_layer = 1
    n_head = 2
    n_embd = 16 # Tiny embedding
    
    config = ModelConfig(vocab_size=vocab_size, 
                         block_size=block_size, 
                         n_layer=n_layer, 
                         n_head=n_head, 
                         n_embd=n_embd)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (M1/M2) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
        
    # Initialize Template Model
    template_model = EvoTransformer(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in template_model.parameters())}")
    
    # Initialize Task
    task = ReverseTask(vocab_size=vocab_size, sequence_length=3)
    
    # Initialize GA
    # Pop size 20 for speed demo, check memory usage.
    ga = GeneticAlgorithm(template_model, pop_size=100, mutation_rate=0.05, mutation_power=0.01, device=device)
    
    generations = 500
    
    print("Starting Evolution...")
    start_time = time.time()
    
    for gen in range(generations):
        population = ga.population # List of state_dicts
        scores = []
        
        # Evaluation Loop
        # In a real heavy setup, this would be parallelized. 
        # For M1 + PyTorch, sequential eval on GPU might be fast enough for tiny models due to batching overheads if we tried to hack it.
        # But properly, we should batch inference across population if possible.
        # Implementing population batching is complex (requires modifying model to accept Batch x Population).
        # We will loop for simplicity and "From Scratch" readability.
        
        for i, individual_state in enumerate(population):
            # Load weights
            template_model.load_state_dict(individual_state)
            
            # Evaluate
            score = task.evaluate(template_model, batch_size=128, device=device)
            scores.append((i, score))
        
        # Evolve
        best_score = ga.evolve(scores)
        
        if (gen+1) % 10 == 0:
            print(f"Gen {gen+1}/{generations} | Best Score: {best_score:.4f} | Time: {time.time()-start_time:.2f}s")
            
        if best_score >= 0.99:
            print(f"Solved in generation {gen+1}!")
            break
            
    print("Evolution Finished.")
    
    # Save best model
    best_state = ga.population[0] # Elitism puts best at 0
    save_path = "best_model.pth"
    torch.save(best_state, save_path)
    print(f"Saved best model to {save_path}")

    # Test final best
    template_model.load_state_dict(best_state)
    print("\nVisualizing Best Model:")
    inputs, targets = task.get_batch(batch_size=3, device=device)
    output = template_model.generate(inputs, max_new_tokens=5)
    
    for i in range(3):
        inp = inputs[i].tolist()
        tgt = targets[i].tolist()
        out = output[i].tolist()
        print(f"Input:    {inp}")
        print(f"Target:   {tgt} (Last 5 tokens)")
        print(f"Output:   {out}")
        print("-" * 20)

if __name__ == "__main__":
    main()
