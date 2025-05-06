from basic import *


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cond):
        q = self.q(x)
        k, v = self.kv(cond).chunk(2, dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x


class DiTBlockWithCross(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        
        self.self_attn = Attention(hidden_size, num_heads=num_heads)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        
        self.fc1 = nn.Linear(hidden_size, mlp_hidden, bias=False)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size, bias=False)

    def forward(self, x, cond, self_attn_scale, self_attn_shift, self_attn_gate, 
                cross_attn_scale, cross_attn_shift, cross_attn_gate,
                mlp_scale, mlp_shift, mlp_gate):
        # Self-attention with modulation
        norm_x = F.rms_norm(x, (x.size(-1),))
        norm_x = norm_x * (1 + self_attn_scale) + self_attn_shift
        attn_out = self.self_attn(norm_x)
        x = x + attn_out * self_attn_gate
        
        # Cross-attention with modulation
        norm_x = F.rms_norm(x, (x.size(-1),))
        norm_x = norm_x * (1 + cross_attn_scale) + cross_attn_shift
        cross_out = self.cross_attn(norm_x, cond)
        x = x + cross_out * cross_attn_gate
        
        # MLP with modulation
        norm_x = F.rms_norm(x, (x.size(-1),))
        norm_x = norm_x * (1 + mlp_scale) + mlp_shift
        x = x + self.fc2(self.fc1(norm_x).relu().pow(2)) * mlp_gate
        
        return x


class CrossDiT(nn.Module):
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

        # Nine modulations for all layers: 3 sets of (scale, shift, gate)
        self.modulation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(
                hidden_size, 9 * hidden_size, bias=True
            ),  # 9 parameters: 3 sets of (scale, shift, gate)
        )
        # Initialize with zeros
        self.modulation[-1].weight.data.zero_()
        self.modulation[-1].bias.data.zero_()

        self.blocks = nn.ModuleList(
            [
                DiTBlockWithCross(
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
            ),  # Only 2 parameters: scale, shift
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
        
        # Get modulation parameters - shared across all layers
        # 9 parameters: 3 sets of (scale, shift, gate)
        modulations = self.modulation(t_emb).chunk(9, dim=1)
        
        # Split modulations into self-attention, cross-attention, and mlp params
        self_attn_scale = modulations[0][:, None, :]  # [B, 1, D]
        self_attn_shift = modulations[1][:, None, :]  # [B, 1, D]
        self_attn_gate = modulations[2][:, None, :]   # [B, 1, D]
        
        cross_attn_scale = modulations[3][:, None, :] # [B, 1, D]
        cross_attn_shift = modulations[4][:, None, :] # [B, 1, D]
        cross_attn_gate = modulations[5][:, None, :]  # [B, 1, D]
        
        mlp_scale = modulations[6][:, None, :]        # [B, 1, D]
        mlp_shift = modulations[7][:, None, :]        # [B, 1, D]
        mlp_gate = modulations[8][:, None, :]         # [B, 1, D]

        # Process context
        context = self.context_embedding(cond)

        B, T_input, _ = x.shape
        x = self.input_proj(x)
        
        # Apply positional embeddings
        x = x + self.patch_posemb[:, :T_input, :]

        # Pass through blocks
        for block in self.blocks:
            x = block(
                x, context, 
                self_attn_scale, self_attn_shift, self_attn_gate,
                cross_attn_scale, cross_attn_shift, cross_attn_gate,
                mlp_scale, mlp_shift, mlp_gate
            )

        # Final normalization and projection
        x = F.rms_norm(x, (x.size(-1),))
        final_scale, final_shift = self.final_modulation(t_emb).chunk(2, dim=1)
        final_scale = final_scale[:, None, :]  # [B, 1, D]
        final_shift = final_shift[:, None, :]  # [B, 1, D]
        x = x * (1 + final_scale) + final_shift
        x = self.final_proj(x)
        return x


### TESTS ###

def _test_discrete_context_model():
    model = CrossDiT(is_discrete_context=True, vocab_size=1000)
    x = torch.randn(3, 16, 64)
    discrete_context = torch.randint(0, 1000, (3, 10))
    timesteps = torch.tensor([0.5, 0.1, 1.0])
    print("Discrete context output shape:", model(x, discrete_context, timesteps).shape)


def _test_continuous_context_model():
    model = CrossDiT(is_discrete_context=False, context_dim=256)
    x = torch.randn(3, 16, 64)
    continuous_context = torch.randn(3, 10, 256)
    timesteps = torch.tensor([0.5, 0.1, 1.0])
    print(
        "Continuous context output shape:",
        model(x, continuous_context, timesteps).shape,
    )


if __name__ == "__main__":
    _test_discrete_context_model()
    _test_continuous_context_model()
    
    

    