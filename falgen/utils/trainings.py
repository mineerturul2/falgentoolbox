import os
import random

import numpy as np
import torch
import lovely_tensors

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def typical_setup():

    lovely_tensors.monkey_patch()
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.empty(1, device="cuda", requires_grad=True).backward()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 512
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    """Count the number of parameters in a model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        Dictionary with "total", "trainable", and "frozen" counts (integers).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}



def forward_with_activation_hooks(model, **kwargs):
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                tensor = output
            else:
                tensor = output[0] if isinstance(output, (tuple, list)) else output

            with torch.no_grad():
                activations[name] = (
                    tensor.mean().item(),
                    tensor.std().item(),
                    tensor.min().item(),
                    tensor.max().item(),
                )

        return hook

    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if name:  # Skip the root module
            handle = module.register_forward_hook(hook_fn(name))
            hooks.append(handle)

    # Forward pass
    output = model(**kwargs)

    # Remove hooks
    for handle in hooks:
        handle.remove()

    return output, activations


def _test_forward_with_activation_hooks():

    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, width=128, depth=4):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(10, width))
            for _ in range(depth - 2):
                self.layers.append(nn.Linear(width, width))
            self.layers.append(nn.Linear(width, 1))

            # Initialize with std(1)
            for layer in self.layers:
                nn.init.zeros_(layer.bias)

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # No activation on last layer
                    x = torch.relu(x)
            return x

    # Test with different widths
    widths = [32, 64, 128, 256, 512]
    plt.figure(figsize=(15, 6))
    x = torch.randn(10, 10)

    # Prepare data structures to collect statistics across widths
    all_stds = []
    all_maxs = []
    layer_indices = []

    for width in widths:
        model = MLP(width=width)

        # Forward pass with activation hooks
        _, activations = forward_with_activation_hooks(model, x=x)

        # Extract statistics
        layer_names = []
        stds = []
        maxs = []

        for name, stats in activations.items():
            if "layers" in name:  # Only plot layer activations
                layer_names.append(name)
                stds.append(stats[1])  # std
                maxs.append(stats[3])  # max

        # Sort by layer depth
        indices = sorted(range(len(layer_names)), key=lambda i: layer_names[i])
        layer_names = [layer_names[i] for i in indices]
        stds = [stds[i] for i in indices]
        maxs = [maxs[i] for i in indices]

        all_stds.append(stds)
        all_maxs.append(maxs)

        if not layer_indices:
            layer_indices = list(range(len(layer_names)))

    # Plot standard deviations
    plt.subplot(1, 2, 1)
    for i, width in enumerate(widths):
        plt.plot(layer_indices, all_stds[i], label=f"Width = {width}")
    plt.title("Standard Deviation vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Standard Deviation")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)

    # Plot maximums
    plt.subplot(1, 2, 2)
    for i, width in enumerate(widths):
        plt.plot(layer_indices, all_maxs[i], label=f"Width = {width}")
    plt.title("Maximum Value vs Layer")
    plt.xlabel("Layer")
    plt.ylabel("Maximum Value")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Create directory if it doesn't exist
    os.makedirs("_activation_hooks", exist_ok=True)

    # Save the plot
    plt.savefig("_activation_hooks/activation_statistics.png")
    plt.close()

    print(
        "Test completed. Activation statistics saved to _activation_hooks/activation_statistics.png"
    )


if __name__ == "__main__":
    _test_forward_with_activation_hooks()
