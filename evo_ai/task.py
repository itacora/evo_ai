import torch
import random

class ReverseTask:
    def __init__(self, vocab_size, sequence_length):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        # Exclude 0, reserved for padding/special if needed, but here simple.
        # Let's say tokens are 1..vocab_size-1
        
    def get_batch(self, batch_size, device='cpu'):
        # Generate random sequences
        # Input: [A, B, C]
        # Target: [C, B, A]
        
        # We need to be careful with GPT style. 
        # Usually GPT is trained to predict next token.
        # Here we want to prompt it: "Input: A B C Output:" -> expect "C B A"
        # But for simple evolution, we can simplify. 
        # Input to model: [A, B, C, SEP]
        # Expected generation: [C, B, A]
        
        # Let's use a separator token. vocab_size should include it.
        # Tokens: 0..vocab_size-2 are data, vocab_size-1 is SEP.
        
        max_val = self.vocab_size - 2
        
        data = torch.randint(0, max_val + 1, (batch_size, self.sequence_length))
        sep = torch.full((batch_size, 1), self.vocab_size - 1)
        
        inputs = torch.cat([data, sep], dim=1).to(device)
        targets = torch.flip(data, [1]).to(device)
        
        return inputs, targets

    def evaluate(self, model, batch_size=10, device='cpu'):
        model.eval()
        with torch.no_grad():
            inputs, targets = self.get_batch(batch_size, device)
            
            # Generate
            # We need to generate sequence_length tokens
            # inputs is (B, L+1)
            generated = model.generate(inputs, max_new_tokens=self.sequence_length)
            
            # generated is (B, L+1 + L)
            # The part we care about is the last L tokens
            outputs = generated[:, -self.sequence_length:]
            
            # Calculate accuracy
            # Perfect match for the whole sequence? Or per token?
            # Per token is smoother for evolution.
            
            matches = (outputs == targets).float()
            score = matches.mean().item()
            
            return score
