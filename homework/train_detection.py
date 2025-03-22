import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import Detector, load_model, save_model
from .datasets.drive_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    # create loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    global_step = 0
    metrics = {"train_seg_acc": [], "val_seg_acc": [], "train_depth_loss": [], "val_depth_loss": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            img, depth_gt, seg_gt = batch["image"].to(device), batch["depth"].to(device), batch["track"].to(device).long()

            # forward pass
            seg_pred, depth_pred = model(img)

            # calculate loss
            seg_loss = torch.nn.functional.cross_entropy(seg_pred, seg_gt)
            depth_loss = torch.nn.functional.mse_loss(depth_pred, depth_gt)
            total_loss = seg_loss + depth_loss

            # backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_step += 1

            # compute and store training metrics
            seg_acc = (seg_pred.argmax(dim=1) == seg_gt).float().mean()
            metrics["train_seg_acc"].append(seg_acc)
            metrics["train_depth_loss"].append(depth_loss.item())

        epoch_train_seg_acc = torch.as_tensor(metrics["train_seg_acc"]).mean()
        epoch_train_depth_loss = np.mean(metrics["train_depth_loss"])

        # validation loop
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                img, depth_gt, seg_gt = batch["image"].to(device), batch["depth"].to(device), batch["track"].to(device).long()

                # forward pass
                seg_pred, depth_pred = model(img)

                # calculate loss
                seg_loss = torch.nn.functional.cross_entropy(seg_pred, seg_gt)
                depth_loss = torch.nn.functional.mse_loss(depth_pred, depth_gt)

                # compute and store validation metrics
                seg_acc = (seg_pred.argmax(dim=1) == seg_gt).float().mean()
                metrics["val_seg_acc"].append(seg_acc)
                metrics["val_depth_loss"].append(depth_loss.item())

        epoch_val_seg_acc = torch.as_tensor(metrics["val_seg_acc"]).mean()
        epoch_val_depth_loss = np.mean(metrics["val_depth_loss"])

        # log metrics to tensorboard
        logger.add_scalar("train_seg_acc", epoch_train_seg_acc, global_step=global_step)
        logger.add_scalar("val_seg_acc", epoch_val_seg_acc, global_step=global_step)
        logger.add_scalar("train_depth_loss", epoch_train_depth_loss, global_step=global_step)
        logger.add_scalar("val_depth_loss", epoch_val_depth_loss, global_step=global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_seg_acc={epoch_train_seg_acc:.4f} "
                f"val_seg_acc={epoch_val_seg_acc:.4f} "
                f"train_depth_loss={epoch_train_depth_loss:.4f} "
                f"val_depth_loss={epoch_val_depth_loss:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))