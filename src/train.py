import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        assert embed_dim%num_heads==0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        out = att @ v  # (B, heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)
            

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4* embed_dim, embed_dim),
        )
    def forward(self, x):
        return self.net(x)
    


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_state):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    


class GPTScratch(nn.Module):
    def __init__(
            self,
            vocab_size = 50257,
            block_size = 128,
            embed_dim = 256,
            num_heads = 4,
            num_layers = 4,
    ):
        super().__init__()

        self.block_size = block_size

        # Token + Pos embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, block_size)
                                     for _ in range(num_layers)
                                     ])
        
        self.ln_f = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)


    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Revisa Longitud de Secuencia"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits



