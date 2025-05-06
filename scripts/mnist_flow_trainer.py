import os
import click
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from einops import rearrange
from falgen.diffusion.basic import SimpleDiT
from falgen.diffusion.trainer import diffusion_loss, diffusion_sample
from falgen.utils.imgs import torch_to_gif
from falgen.utils.mup import model_init, optim
from falgen.utils.trainings import set_seed
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb


def train_epoch(
    model,
    optimizer,
    train_loader,
    device,
    epoch,
    ema_model_state,
    global_step,
    lr_scheduler,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, cond) in enumerate(pbar):

        data = rearrange(data, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2).to(
            device
        )
        cond = cond.view(-1, 1).to(device)
        # randomly replace 10% of cond with 10
        mask = torch.rand(cond.shape) < 0.1
        cond[mask] = 10
        cond = cond.repeat(1, 16)

        optimizer.zero_grad()
        loss = diffusion_loss(model, data, cond)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        pbar.set_description(
            f"Epoch {epoch} Loss: {loss.item():.6f}, lr: {lr_scheduler.get_last_lr()[0]:.6f}"
        )

        # Log to wandb
        wandb.log({"train/loss": loss.item(), "train/step": global_step})

        # Update EMA model
        if batch_idx % 4 == 0:
            for name, param in model.named_parameters():
                ema_model_state[name].data.mul_(0.99).add_(param.data, alpha=0.01)

        global_step += 1

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")
    return avg_loss, global_step


@torch.no_grad()
def eval_epoch(model, val_loader, device, epoch, seed=42):
    """Evaluate for one epoch with fixed seed."""
    model.eval()
    total_loss = 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    pbar = tqdm(val_loader, desc=f"Eval Epoch {epoch}")
    with torch.no_grad():
        for batch_idx, (data, cond) in enumerate(pbar):
            data = rearrange(
                data, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2
            ).to(device)
            cond = cond.view(-1, 1).to(device)
            # randomly replace 10% of cond with 10
            mask = torch.rand(cond.shape, generator=generator) < 0.1
            cond[mask] = 10
            cond = cond.repeat(1, 16)

            loss = diffusion_loss(model, data, cond, seed=batch_idx)
            total_loss += loss.item()
            pbar.set_description(f"Eval Epoch {epoch} Loss: {loss.item():.6f}")

            # Log to wandb
            wandb.log({"val/loss": loss.item()})

    avg_loss = total_loss / len(val_loader)
    print(f"Eval Epoch {epoch} Average Loss: {avg_loss:.6f}")
    return avg_loss


@click.command()
@click.option("--name", default="mnist-flow", help="Name of the run")
@click.option("--epochs", default=10, help="Number of epochs to train")
@click.option("--batch-size", default=256, help="Batch size for training")
@click.option("--eval-batch-size", default=256, help="Batch size for evaluation")
@click.option("--lr", default=0.1, help="Learning rate")
@click.option("--hidden-size", default=128, help="Hidden size for the model")
@click.option("--depth", default=4, help="Number of transformer blocks")
@click.option(
    "--ckpt-dir", default="./checkpoints", help="Directory to save checkpoints"
)
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option(
    "--wandb-project", default="mnist-flow", help="Weights & Biases project name"
)
@click.option("--save-model", default=True, help="Save model checkpoints")
@click.option("--eval-seed", default=42, help="Seed for evaluation reproducibility")
def main(
    name,
    epochs,
    batch_size,
    eval_batch_size,
    lr,
    hidden_size,
    depth,
    ckpt_dir,
    seed,
    wandb_project,
    save_model,
    eval_seed,
):
    # Set seed for reproducibility
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "lr": lr,
            "hidden_size": hidden_size,
            "depth": depth,
            "seed": seed,
            "eval_seed": eval_seed,
        },
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    val_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    model = SimpleDiT(
        in_channels=4,
        out_channels=4,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=4,
        is_discrete_context=True,
        vocab_size=20,
    ).to(device)

    # Log model architecture to wandb
    wandb.watch(model, log="all")

    init_rules = [
        f"*modulation*: {modulation_lr}/sqrt(fan_in)",
        "*modulation*: 0.0",
        "*.weight: 0.1/sqrt(fan_in)",
        "*.bias: 0.01",
        "patch_posemb: 0.0",
    ]

    lr_rules = [
        "*.weight: glr/fan_in",
        "*.bias: glr * 0.1",
        "patch_posemb: glr * 1.0",
    ]

    model = model_init(model, init_rules, lr_rules, glr=lr)
    optimizer = optim(model, optim_class=torch.optim.AdamW)

    max_steps = len(train_loader) * epochs
    warmup_steps = 100
    cooldown_steps = int(max_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        elif step < max_steps - cooldown_steps:
            return 1
        else:
            return (max_steps - step) / cooldown_steps

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(ckpt_dir, exist_ok=True)

    # EMA model state
    ema_model_state = model.state_dict()
    global_step = 0

    for epoch in range(epochs):

        model.eval()
        temp_model_state = model.state_dict()
        model.load_state_dict(ema_model_state)

        with torch.no_grad():
            n_samples = 16
            cond = torch.arange(0, n_samples).to(device) % 10
            uncond = torch.ones_like(cond) * 10
            cond = cond.view(n_samples, 1).repeat(1, 16)
            uncond = uncond.view(n_samples, 1).repeat(1, 16)

            images = diffusion_sample(
                model,
                (n_samples, 16 * 16 * 1, 4),
                cond,
                uncond=uncond,
                seed=0,
            )

            final_images = images[-1]
            # rearrange
            final_images = rearrange(
                final_images,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                p1=2,
                p2=2,
                h=16,
                w=16,
            )

            wandb.log(
                {
                    "samples": wandb.Image(
                        make_grid(
                            final_images,
                            nrow=4,
                            normalize=True,
                            scale_each=True,
                        )
                    )
                }
            )

            # Save as image
            sample_dir = os.path.join(ckpt_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            # Save the final image
            grid = make_grid(final_images, nrow=4, normalize=True, scale_each=True)
            image_path = os.path.join(sample_dir, f"samples_epoch_{epoch}.png")
            utils.save_image(grid, image_path)

            # Save the sampling process as GIF
            reshaped_images = [
                rearrange(
                    img,
                    "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                    p1=2,
                    p2=2,
                    h=16,
                    w=16,
                )
                for img in images[:: len(images) // 10]
            ]
            gif_path = os.path.join(sample_dir, f"sampling_process_epoch_{epoch}.gif")
            torch_to_gif(
                gif_path,
                reshaped_images,
                normalization_range=(-1, 1),
                clip_range=(0, 1),
            )

            # Log the GIF to wandb
            wandb.log({"sampling_process": wandb.Video(gif_path, fps=4, format="gif")})

            # Restore model state
            model.load_state_dict(temp_model_state)

        # Run evaluation on validation set
        val_loss = eval_epoch(model, val_loader, device, epoch, seed=eval_seed)
        wandb.log({"epoch": epoch, "val_loss": val_loss})

        # Save model checkpoint
        if save_model:
            checkpoint_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": ema_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

        avg_loss, global_step = train_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            ema_model_state,
            global_step,
            lr_scheduler,
        )
        wandb.log({"epoch": epoch, "epoch_avg_loss": avg_loss})

    wandb.finish()
    print("Training completed and model saved!")


if __name__ == "__main__":
    main()
