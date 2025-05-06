# Generative Model Toolbox

[WIP] A specialized toolkit for implementing and experimenting with generative models in PyTorch. The library provides modular implementations of diffusion and autoregressive models with a focus on simplicity and reusability.


### Development Installation

To install the package in development mode (changes to the source code will be immediately reflected):

```bash
# Clone the repository
git clone https://github.com/fal-ai-community/falgentoolbox
cd falgentoolbox

# Install in development mode
pip install -e .
```

### Example

```python

from falgen.diffusion import SimpleDiT

# Create a basic diffusion model
model = SimpleDiT(
    in_channels=64,
    out_channels=64,
    hidden_size=768,
    depth=12,
    is_discrete_context=False,
    context_dim=256
)

# Process data
output = model(input_data, context_data, timesteps)
```

## License

MIT License
