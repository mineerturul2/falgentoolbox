# Basic Causal Autoregressive Transformer Model

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal position encoding, reused for token positions."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class CausalAttention(nn.Module):
    """Multi-head causal self-attention with an explicit causal mask."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads
        ).unbind(0)
        # Causal mask via is_causal flag (FlashAttention supported paths)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block with causal self-attention and
    a SwiGLU MLP."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attn = CausalAttention(hidden_size, num_heads=num_heads)

        mlp_hidden = int(hidden_size * mlp_ratio)
        # SwiGLU MLP: gate_proj + up_proj → SiLU(gate) * up → down_proj
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.down_proj = nn.Linear(mlp_hidden, hidden_size, bias=False)

    def forward(self, x):
        # Causal self-attention
        residual = x
        x = F.rms_norm(x, (x.size(-1),))
        x = residual + self.attn(x)

        # SwiGLU MLP
        residual = x
        x = F.rms_norm(x, (x.size(-1),))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = residual + self.down_proj(gate * up)
        return x


class SimpleAR(nn.Module):
    """A minimal autoregressive transformer (decoder-only) for next-token
    prediction over a discrete vocabulary.

    Mirrors the design of SimpleDiT but simplified for causal language
    modeling: no conditioning, no time embedding, no modulation — just
    token embeddings, sinusoidal positional encoding, and stacked
    transformer blocks.
    """

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        max_seq_len=1024,
        tie_embeddings=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        # Learned position embeddings
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size)
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # Small init for position embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)
        # Init every transformer block's final projection near zero
        for block in self.blocks:
            nn.init.zeros_(block.down_proj.weight)

    def forward(self, input_ids, return_logits=True):
        """
        Args:
            input_ids: (B, T) long tensor of token ids.

        Returns:
            logits: (B, T, vocab_size) if return_logits else hidden states.
        """
        B, T = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)  # (B, T, D)

        # Add position embeddings
        x = x + self.pos_embedding[:, :T, :]

        # Pass through blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_logits:
            return self.lm_head(x)  # (B, T, V)

        return x


### TESTS ###


def _test_output_shape():
    model = SimpleAR(vocab_size=1000, hidden_size=64, depth=4, num_heads=4)
    input_ids = torch.randint(0, 1000, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, 1000), f"Expected (2, 16, 1000), got {logits.shape}"
    print(f"Output shape OK: {logits.shape}")


def _test_causal_structure():
    """Verify that later positions cannot attend to earlier ones by
    checking that the logit for the first token does not depend on a
    later token embedded at position 1 (via a simple gradient check)."""
    model = SimpleAR(vocab_size=100, hidden_size=32, depth=2, num_heads=4)
    model.eval()
    input_ids = torch.randint(0, 100, (1, 5))
    pos_embed_copy = model.pos_embedding[:, :5, :].clone()
    logits = model(input_ids)

    # Loss on the first position only
    loss = logits[0, 0].sum()
    grad = torch.autograd.grad(loss, model.pos_embedding)[0]
    # Gradient on positions after the first should be zero for position 0's logit
    assert grad[:, 1:, :].abs().sum().item() == 0.0, (
        "Causal structure violated: position 0 depends on later positions"
    )
    print("Causal structure verified: no gradient flows backward in time.")


if __name__ == "__main__":
    _test_output_shape()
    _test_causal_structure()