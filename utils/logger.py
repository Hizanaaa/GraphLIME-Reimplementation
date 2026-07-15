"""
Training logger.
"""

from pathlib import Path

import pandas as pd


class CSVLogger:

    def __init__(self, path):

        self.path = Path(path)

        self.records = []

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
    ):

        self.records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

    def save(self):

        df = pd.DataFrame(self.records)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            self.path,
            index=False,
        )
