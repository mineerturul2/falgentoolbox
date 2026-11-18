"""Training and generation utilities for autoregressive models."""

import torch
import torch.nn.functional as F


def autoregressive_loss(model, input_ids, ignore_index=-100):
    """Compute cross-entropy loss for next-token prediction.

    Args:
        model: An autoregressive model that returns (B, T, vocab_size) logits.
        input_ids: (B, T) long tensor of token ids.

    Returns:
        loss: scalar cross-entropy loss (mean over non-ignored positions).
    """
    logits = model(input_ids)                    # (B, T, V)
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().reshape(-1, logits.size(-1)),
        input_ids[:, 1:].contiguous().reshape(-1),
        ignore_index=ignore_index,
    )
    return loss


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=64, temperature=1.0, top_k=None, eos_id=None):
    """Generate a continuation from a prompt using the autoregressive model.

    Args:
        model: An autoregressive model returning (B, T, vocab_size) logits.
        prompt_ids: (B, T_prompt) long tensor of token ids.
        max_new_tokens: maximum number of tokens to generate.
        temperature: softmax temperature (> 0). 1.0 = no scaling.
        top_k: if set, only sample from the top-k highest probability tokens.
        eos_id: if set, stop generation when this token is produced.

    Returns:
        full_ids: (B, T_prompt + generated) long tensor of token ids.
    """
    model.eval()
    device = prompt_ids.device
    batch_size = prompt_ids.shape[0]
    full_ids = prompt_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(full_ids)                       # (B, T, V)
        next_logits = logits[:, -1, :] / temperature    # (B, V)

        if top_k is not None:
            top_k_values, _ = torch.topk(next_logits, top_k, dim=-1)
            threshold = top_k_values[:, -1, None]       # (B, 1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        full_ids = torch.cat([full_ids, next_id], dim=1)

        if eos_id is not None:
            if (next_id == eos_id).any():
                break

    return full_ids