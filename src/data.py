import torch
from torch.utils.data import Dataset

def load_corpus(file_path: str) -> str:
    """Load entire text corpus as a single string."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def tokenize_corpus(tokenizer, text: str):

    token_ids = tokenizer.encode(text)
    return torch.tensor(token_ids, dtype=torch.long)

def split_tokens(tokens, train_ratio=0.93):
    """
    sequential splits
    args:
        tokens (torch.tensor): 1D tensor of token IDs
        ratio (float): proportion of data for training  
    """

    n = tokens.size(0)
    split_idx = int(n * train_ratio)

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    return train_tokens, val_tokens

def create_sequences(tokens: torch.Tensor, block_size: int):

    """
    Converts a 1D token stream into (input, target) sequences.

    - no overlap
    - target is shifted by +1
    - no padding

    returns:
        x : (num_sequences, block_size)
        y : (num_sequences, block_size)

    """

    n = tokens.size(0)

    usable_tokens = (n // (block_size + 1)) * (block_size + 1)
    tokens = tokens[:usable_tokens]

    chunks = tokens.view(-1, block_size + 1)

    x = chunks[:, :-1]
    y = chunks[:, 1:]

    assert x.shape == y.shape
    assert x.shape[1] == block_size

    if x.numel() > 0:
        k = 0
        assert torch.all(x[k, 1:] == y[k, :-1])

    return x,y

class GPTDataset(Dataset):

    def __init__(self, tokens: torch.Tensor, block_size: int):
        
        assert tokens.dim() == 1 # 1D Tensor

        self.tokens = tokens
        
        self.block_size = block_size

        self.num_sequences = len(tokens) // (block_size + 1)

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * (self.block_size +1)
        end = start + self.block_size + 1

        chunk = self.tokens[start:end]

        x = chunk[:-1]
        y = chunk[1:]

        if idx < 3:
            assert x.shape[0] == self.block_size
            assert y.shape[0] == self.block_size
            assert torch.equal(x[1:], y[:-1]), "Shift mismatch"

        return x,y