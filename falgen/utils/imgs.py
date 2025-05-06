import numpy as np
from PIL import Image
from torchvision.utils import make_grid


def torch_to_gif(path, tensor_list, normalization_range=(-1, 1), clip_range=(0, 1)):
    """
    Save a list of tensors as a GIF.
    Args:
        path: Path to save the GIF
        tensor_list: List of tensors to save
        normalization_range: Range to normalize from (min, max)
        clip_range: Range to clip to (min, max)
    """
    frames = []
    for tensor in tensor_list:
        # Normalize
        norm_min, norm_max = normalization_range
        clip_min, clip_max = clip_range
        tensor = (tensor - norm_min) / (norm_max - norm_min) * (
            clip_max - clip_min
        ) + clip_min
        tensor = tensor.clamp(clip_min, clip_max)

        # Convert to grid and then to image
        grid = make_grid(tensor, nrow=int(np.sqrt(tensor.size(0))))
        img_np = grid.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        frames.append(Image.fromarray(img_np.squeeze()))

    # Save as GIF
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
