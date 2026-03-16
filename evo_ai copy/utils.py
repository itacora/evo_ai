class CharTokenizer:
    def __init__(self):
        # Define characters: printable ASCII
        # 0 is reserved for padding/unknown, though we might not use it.
        # Let's map strict chars.
        # We want to support: a-z, A-Z, 0-9, punctuation, space.
        chars = sorted(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        self.stoi = { ch:i+1 for i,ch in enumerate(chars) }
        self.itos = { i+1:ch for i,ch in enumerate(chars) }
        self.vocab_size = len(chars) + 1 # +1 for 0 (PAD)
        # Add special token for End of Sequence? 
        # For simplicity, let's just use fixed length or rely on space padding.
        
    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]
        
    def decode(self, tokens):
        return ''.join([self.itos.get(t, '') for t in tokens])
