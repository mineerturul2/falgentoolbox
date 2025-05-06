import os
import re
import sys
import warnings

import numpy as np
import sympy
import torch
import torch.nn as nn


def get_fan_in(param):

    if not hasattr(param, "shape"):
        return 1

    shape = param.shape

    if len(shape) == 1:
        return 1
    else:
        return np.prod(shape[1:])


def parse_trainable_config(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(
                {
                    "name": name,
                    "lr_value": 1.0,
                    "init_value": 1.0,
                    "fan_in": get_fan_in(param),
                }
            )
    return trainable_params


def evaluate_expression(expr_str, **kwargs):
    if any(var in expr_str for var in kwargs):

        symbols = {var: sympy.Symbol(var) for var in kwargs}
        expr = sympy.sympify(expr_str.replace("/", "*1.0/"))

        result = float(
            expr.subs({symbols[var]: value for var, value in kwargs.items()})
        )
        return result
    else:

        return float(expr_str)


def parse_rules(rules):
    parsed_rules = []
    for rule in rules:
        if ":" not in rule:
            warnings.warn(f"Invalid rule format: {rule}. Expected 'pattern: value'")
            continue

        pattern, value = rule.split(":", 1)
        pattern = pattern.strip()
        value = value.strip()

        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        parsed_rules.append((re.compile(regex_pattern), value))

    return parsed_rules


def get_rule_values(rules, name, **kwargs):
    for pattern, value in rules:
        if pattern.match(name):
            return evaluate_expression(value, **kwargs), pattern
    return None, pattern


def model_init(model, init_rules=None, lr_rules=None, **kwargs):
    if init_rules is None:
        init_rules = []
    if lr_rules is None:
        lr_rules = []

    parsed_init_rules = parse_rules(init_rules)
    parsed_lr_rules = parse_rules(lr_rules)

    for name, param in model.named_parameters():
        fan_in = get_fan_in(param)
        init_value, pattern_init = get_rule_values(
            parsed_init_rules, name, fan_in=fan_in, **kwargs
        )

        assert init_value is not None, f"No initialization rule found for {name}"

        if len(param.shape) > 1:
            nn.init.normal_(param, mean=0.0, std=init_value)
        else:
            nn.init.constant_(param, 0)

        param.init_value = init_value

        lr_value, pattern_lr = get_rule_values(
            parsed_lr_rules, name, fan_in=fan_in, **kwargs
        )
        assert lr_value is not None, f"No learning rate rule found for {name}"
        param.lr_value = lr_value

        print(f"name: {name}, Pattern init: {pattern_init}, Pattern lr: {pattern_lr}")

    return model


def optim(model, optim_class=torch.optim.AdamW, **kwargs):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append({"params": param, "lr": param.lr_value})
    return optim_class(params, **kwargs)


def _test_model_init(init_rules, lr_rules):

    import torch
    from torch import nn

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(64, 128)
            self.linear2 = nn.Linear(128, 256)
            self.conv = nn.Embedding(1000, 64)

        def forward(self, x):
            x = self.conv(x)
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    model1 = TestModel()
    model2 = TestModel()

    torch.manual_seed(42)
    model1 = model_init(model1, init_rules, lr_rules)

    torch.manual_seed(42)
    model2 = model_init(model2, init_rules, lr_rules)

    parsed_init_rules = parse_rules(init_rules)
    parsed_lr_rules = parse_rules(lr_rules)

    all_equal = True
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if not torch.allclose(param1, param2):
            print(f"Parameters {name1} not equal between runs")
            all_equal = False

    lr_applied = True
    for name, param in model1.named_parameters():
        if hasattr(param, "lr_value"):
            fan_in = get_fan_in(param)
            expected_lr, pattern_lr = get_rule_values(
                parsed_lr_rules, name, fan_in=fan_in
            )

            if param.lr_value != expected_lr:
                print(
                    f"LR multiplier for {name} is {param.lr_value}, expected {expected_lr}"
                )
                lr_applied = False
    for name, param in model1.named_parameters():
        print(
            f"Parameter {name}: lr_value = {param.lr_value}, init_value = {param.init_value}"
        )

    return all_equal and lr_applied


def _visualize_scaling_lr_init_rule():
    from falgen.diffusion.basic import SimpleDiT

    widths = [16, 32, 64, 128, 256]

    init_rules = [
        "*qkv.weight: ginit/sqrt(fan_in)",
        "*fc1.weight: ginit/sqrt(fan_in)",
        "*fc2.weight: ginit/sqrt(fan_in)",
        "*input_proj.weight: 0.1/sqrt(fan_in)",
        "*time_embed.0.weight: 0.1/sqrt(fan_in)",
        "*time_embed.2.weight: 0.1/sqrt(fan_in)",
        "*context_embedding.weight: 0.1/sqrt(fan_in)",
        "*modulation.0.weight: 0.1/sqrt(fan_in)",
        "*modulation.2.weight: 0.1/sqrt(fan_in)",
        "patch_posemb*: 1.0",
        "*.bias: 0.01",
        "*.weight: ginit/sqrt(fan_in)",
    ]

    lr_rules = ["*.weight: 1/fan_in", "patch_posemb*: 1.0", "*.bias: 0.1"]

    config_list = []
    for width in widths:
        config = {"width": width, "params": {}}
        print(f"Model width: {width}")
        model = SimpleDiT(hidden_size=width, depth=2)
        model_init(model, init_rules, lr_rules, ginit=1.1)

        for name, param in model.named_parameters():
            print(
                f"Parameter {name}: lr_value = {param.lr_value}, init_value = {param.init_value}"
            )
            config["params"][name] = {
                "lr_value": param.lr_value,
                "init_value": param.data.std(unbiased=False).item(),
            }
        config_list.append(config)

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Learning Rate Scaling", "Initialization Value Scaling"),
        horizontal_spacing=0.1,
    )

    all_param_names = set()
    for config in config_list:
        all_param_names.update(config["params"].keys())

    for param_name in all_param_names:
        widths_data = []
        lr_values = []
        for config in config_list:
            if param_name in config["params"]:
                widths_data.append(config["width"])
                lr_values.append(config["params"][param_name]["lr_value"])

        if len(lr_values) > 1:
            if "weight" in param_name:
                color = "blue"
            elif "bias" in param_name:
                color = "red"
            else:
                color = "green"

            fig.add_trace(
                go.Scatter(
                    x=widths_data,
                    y=lr_values,
                    mode="lines+markers",
                    name=param_name,
                    line=dict(color=color),
                    marker=dict(size=8),
                    hovertemplate="Width: %{x}<br>LR: %{y}<br>Param: " + param_name,
                ),
                row=1,
                col=1,
            )

    for param_name in all_param_names:
        widths_data = []
        init_values = []
        for config in config_list:
            if param_name in config["params"]:
                widths_data.append(config["width"])
                init_values.append(config["params"][param_name]["init_value"])

        if len(init_values) > 1:
            if "weight" in param_name:
                color = "blue"
            elif "bias" in param_name:
                color = "red"
            else:
                color = "green"

            fig.add_trace(
                go.Scatter(
                    x=widths_data,
                    y=init_values,
                    mode="lines+markers",
                    name=param_name,
                    line=dict(color=color),
                    marker=dict(size=8),
                    hovertemplate="Width: %{x}<br>Init: %{y}<br>Param: " + param_name,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(type="log", title_text="Width", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Width", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Learning Rate", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Initialization Value", row=1, col=2)

    fig.update_layout(
        height=600,
        width=1200,
        title_text="Parameter Scaling Behavior",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        margin=dict(r=150),
    )

    fig.write_html("_scaling_behavior_interactive.html")
    fig.write_image("_scaling_fig_2.png", scale=2)


def muP_scaling_coordinate_check():
    import torch
    from torch import nn
    from torch.nn import functional as F

    from falgen.diffusion.basic import SimpleDiT

    widths = [16, 32, 64, 128, 256]

    lr_rules = ["*.weight: 1/fan_in", "patch_posemb*: 1.0", "*.bias: 0.1"]
    init_rules = ["*.weight: ginit/sqrt(fan_in)", "*.bias: 1.0"]

    config_list = []
    step_to_track_activations = 10
    total_steps = 100
    data_batch_input = (
        torch.randn(32, 16, 64),
        torch.randint(0, 1000, (32, 10)),
        torch.randn(
            32,
        ),
    )
    data_batch_target = torch.randn(32, 16, 64)

    for width in widths:
        config = {"width": width, "params": {}}
        print(f"Model width: {width}")
        model = SimpleDiT(
            hidden_size=width, depth=2, is_discrete_context=True, vocab_size=1000
        )
        model_init(model, init_rules, lr_rules, ginit=0.1)
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append({"name": name, "param": param, "lr": param.lr_value})

        optimizer = torch.optim.AdamW(params, lr=0.001)
        for step in range(total_steps):
            model.train()
            output = model(data_batch_input)
            loss = F.mse_loss(output, data_batch_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % step_to_track_activations == 0:
                print(f"Step {step}, Loss: {loss.item()}")

                activation_stats = {}

                def hook_fn(name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            activation_stats[name] = {
                                "mean": output.mean().item(),
                                "std": output.std().item(),
                                "norm": output.norm().item(),
                                "shape": list(output.shape),
                            }
                        elif isinstance(output, tuple) and len(output) > 0:
                            if isinstance(output[0], torch.Tensor):
                                activation_stats[name] = {
                                    "mean": output[0].mean().item(),
                                    "std": output[0].std().item(),
                                    "norm": output[0].norm().item(),
                                    "shape": list(output[0].shape),
                                }

                    return hook

                hooks = []
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)) or "block" in name:
                        hooks.append(module.register_forward_hook(hook_fn(name)))

                with torch.no_grad():
                    model.eval()
                    _ = model(data_batch_input)

                for hook in hooks:
                    hook.remove()

                if step == 0:
                    config["initial_activations"] = activation_stats
                elif step == total_steps - 1:
                    config["final_activations"] = activation_stats

                grad_stats = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_stats[name] = {
                            "mean": param.grad.mean().item(),
                            "std": param.grad.std().item(),
                            "norm": param.grad.norm().item(),
                        }

                if step == 0:
                    config["initial_grads"] = grad_stats
                elif step == total_steps - 1:
                    config["final_grads"] = grad_stats


if __name__ == "__main__":
    init_rules = ["conv.weight: 2/fan_in", "*.bias: 1", "*: 1.0"]
    lr_rules = ["*.weight: 0.1", "attention.*.weight: sqrt(fan_in)", "*: 1.0"]
    _test_model_init(init_rules, lr_rules)
    _visualize_scaling_lr_init_rule()
