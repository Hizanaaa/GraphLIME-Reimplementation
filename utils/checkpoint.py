"""
Checkpoint utilities.
"""

from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    history,
    path,
):
    """
    Save training checkpoint.
    """

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
):
    """
    Load checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint