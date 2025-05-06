# Basic Concatendated DiT Model

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class Attention(nn.Module):
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
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()

        self.self_attn = Attention(hidden_size, num_heads=num_heads)

        mlp_hidden = int(hidden_size * mlp_ratio)

        self.fc1 = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size, bias=False)

    def forward(self, x, scale, shift, gate):
        # Use the same modulation parameters for both attention and MLP
        norm_x = F.rms_norm(x, (x.size(-1),))
        norm_x = norm_x * (1 + scale) + shift
        attn_out = self.self_attn(norm_x)
        x = x + attn_out * gate

        norm_x = F.rms_norm(x, (x.size(-1),))
        norm_x = norm_x * (1 + scale) + shift
        x = x + self.fc2(self.fc1(norm_x).relu().pow(2)) * gate

        return x


class SimpleDiT(nn.Module):
    def __init__(
        self,
        in_channels=4 * 4 * 4,
        out_channels=4 * 4 * 4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_seq_len=1024,
        # contexts
        is_discrete_context=False,
        vocab_size=None,
        context_dim=256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(in_channels, hidden_size, bias=False)

        # context
        if is_discrete_context:
            assert (
                vocab_size is not None
            ), "vocab_size must be provided if is_discrete_context is True"
            self.context_embedding = nn.Embedding(vocab_size, hidden_size)
        else:
            self.context_embedding = nn.Linear(context_dim, hidden_size, bias=False)

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
        )

        # Single modulation for all layers
        self.modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(
                hidden_size, 3 * hidden_size, bias=True
            ),  # Only 3 parameters now: scale, shift, gate
        )
        # Initialize with zeros
        self.modulation[-1].weight.data.zero_()
        self.modulation[-1].bias.data.zero_()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.patch_posemb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))

        self.final_modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(
                hidden_size, 2 * hidden_size, bias=True
            ),  # Only 3 parameters now: scale, shift
        )

        self.final_proj = nn.Linear(hidden_size, out_channels)

        # Initialize final layer with zeros
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, cond, t):
        t_emb = timestep_embedding(t * 1000, self.hidden_size).to(
            x.device, dtype=x.dtype
        )
        t_emb = self.time_embed(t_emb)
        #print(t_emb.shape)
        # Get modulation parameters - shared across all layers
        scale, shift, gate = self.modulation(t_emb).chunk(3, dim=1)
        scale = scale[:, None, :]  # [B, 1, D]
        shift = shift[:, None, :]  # [B, 1, D]
        gate = gate[:, None, :]  # [B, 1, D]

        context = self.context_embedding(cond)

        B, T_input, _ = x.shape
        x = self.input_proj(x)
        x = torch.cat([x, context], dim=1)
        x = x + self.patch_posemb[:, : x.shape[1], :]

        for block in self.blocks:
            x = block(x, scale, shift, gate)

        x = x[:, :T_input, :]
        x = F.rms_norm(x, (x.size(-1),))
        final_scale, final_shift = self.final_modulation(t_emb).chunk(2, dim=1)
        final_scale = final_scale[:, None, :]  # [B, 1, D]
        final_shift = final_shift[:, None, :]  # [B, 1, D]
        x = x * (1 + final_scale) + final_shift
        x = self.final_proj(x)
        return x


### TESTS ###


def _test_discrete_context_model():
    model = SimpleDiT(is_discrete_context=True, vocab_size=1000)
    x = torch.randn(3, 16, 64)
    discrete_context = torch.randint(0, 1000, (3, 10))
    timesteps = torch.tensor([0.5, 0.1, 1.0])
    print("Discrete context output shape:", model(x, discrete_context, timesteps).shape)


def _test_continuous_context_model():
    model = SimpleDiT(is_discrete_context=False, context_dim=256)
    x = torch.randn(3, 16, 64)
    continuous_context = torch.randn(3, 10, 256)
    timesteps = torch.tensor([0.5, 0.1, 1.0])
    print(
        "Continuous context output shape:",
        model(x, continuous_context, timesteps).shape,
    )


def _visualize_time_embedding():
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a range of timesteps from 0 to 1
    timesteps = torch.linspace(0, 1, 100)

    # Get the time embeddings (using dim=64 for visualization)
    dim = 256
    embeddings = timestep_embedding(timesteps * 1000, dim)

    # Convert to numpy for plotting
    embeddings_np = embeddings.numpy()

    # Create a heatmap visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(embeddings_np.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Embedding Value")
    plt.xlabel("Timestep Index (0 to 1)")
    plt.ylabel("Embedding Dimension")
    plt.title("Time Embedding Visualization (t * 1000)")

    # Add ticks for timesteps
    tick_indices = np.linspace(0, len(timesteps) - 1, 11).astype(int)
    tick_values = [f"{timesteps[i]:.1f}" for i in tick_indices]
    plt.xticks(tick_indices, tick_values)

    plt.tight_layout()
    plt.savefig("time_embedding_visualization.png")
    plt.close()

    # Also plot a few specific dimensions to see the sinusoidal patterns
    plt.figure(figsize=(10, 6))
    for i in range(0, dim, 16):  # Plot every 16th dimension
        plt.plot(timesteps.numpy(), embeddings_np[:, i], label=f"Dim {i}")

    plt.xlabel("Timestep (t)")
    plt.ylabel("Embedding Value")
    plt.title("Selected Dimensions of Time Embedding")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_embedding_dimensions.png")
    plt.close()

    print(
        "Time embedding visualizations saved to 'time_embedding_visualization.png' and 'time_embedding_dimensions.png'"
    )


if __name__ == "__main__":
    _visualize_time_embedding()
    _test_discrete_context_model()
    _test_continuous_context_model()
