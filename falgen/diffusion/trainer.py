import torch
import torch.nn.functional as F


def batchdim_view(t_like, target):
    return t_like.view([target.shape[0], *([1] * (len(target.shape) - 1))])


def diffusion_loss(model, x, cond, t=None, device="cuda", seed=None):
    """
    model: Need to take x_noised, t, cond as input.
    x: (B, ...)
    cond: (B, ...)
    t: (B,)
    """
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    if t is None:
        t = torch.rand(x.shape[0], device=device, generator=gen)

    noise = torch.randn(x.shape, device=device, generator=gen)
    x_noised = torch.lerp(x, noise, batchdim_view(t, x))
    v = model(x_noised, t=t, cond=cond)
    loss = F.mse_loss(v, noise - x)
    return loss


def diffusion_sample(
    model, x_or_x_shape, cond, uncond=None, device="cuda", seed=None, num_inf=50
):
    """
    x_or_x_shape: (B, ...) or (B, 1, H, W)
    cond: (B, ...)
    """
    if isinstance(x_or_x_shape, torch.Tensor):
        x_or_x_shape = x_or_x_shape.shape
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    x = torch.randn(x_or_x_shape, device=device, generator=gen)

    dt = 1.0 / num_inf
    dt = batchdim_view(torch.tensor([dt] * x.shape[0]), x)
    dt = dt.to(device)
    images = [x]
    for i in range(num_inf, 0, -1):
        t = i / num_inf
        t = torch.tensor([t] * x.shape[0]).to(device)

        vc = model(x, cond=cond, t=t)
        if uncond is not None:
            vu = model(x, cond=uncond, t=t)
            vc = vu + 2.0 * (vc - vu)

        x = x - dt * vc
        images.append(x)
    return images
