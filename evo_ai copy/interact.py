import torch
from model import EvoTransformer, ModelConfig

def main():
    # Load Config (Match train.py)
    vocab_size = 10 
    block_size = 16 
    n_layer = 1
    n_head = 2
    n_embd = 16 
    
    config = ModelConfig(vocab_size=vocab_size, 
                         block_size=block_size, 
                         n_layer=n_layer, 
                         n_head=n_head, 
                         n_embd=n_embd)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Load Model
    try:
        model = EvoTransformer(config).to(device)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: best_model.pth not found. Please run train.py first.")
        return

    print("Type a sequence of numbers (0-8) separated by spaces to see the model reverse it.")
    print("Example: 1 2 3")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nInput: ")
            if user_input.lower() == 'exit':
                break
                
            # Parse input
            tokens = [int(x) for x in user_input.split()]
            
            # Validation
            if any(t < 0 or t > 8 for t in tokens):
                print("Error: Numbers must be between 0 and 8.")
                continue
            if len(tokens) > block_size - 1: # Reserve 1 for SEP
                 print(f"Error: Sequence too long. Max {block_size - 1} numbers.")
                 continue
                 
            # Add SEP token (9)
            tokens.append(9)
            
            # Convert to tensor
            idx = torch.tensor([tokens], dtype=torch.long).to(device)
            
            # Generate
            # We want to generate len(tokens)-1 new tokens (the reverse sequence)
            num_to_generate = len(tokens) - 1
            
            generated = model.generate(idx, max_new_tokens=num_to_generate)
            
            # Extract output
            output_tokens = generated[0, len(tokens):].tolist()
            
            print(f"Output: {output_tokens}")
            
        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
