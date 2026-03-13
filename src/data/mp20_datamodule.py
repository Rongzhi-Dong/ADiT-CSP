"""Crystal Structure Prediction datamodule — MP20 only."""

from typing import Optional

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src.data.components.mp20_dataset import MP20
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# MP20 split sizes (same as JointDataModule)
_MP20_TRAIN_END = 27138
_MP20_VAL_END = 27138 + 9046  # 36184


class MP20DataModule(LightningDataModule):
    """`LightningDataModule` for Crystal Structure Prediction on MP20.

    Single-dataset datamodule.
    Returns a single val/test dataloader so Lightning does NOT require
    `dataloader_idx` in `validation_step`.
    """

    def __init__(
        self,
        root: str,
        batch_size: DictConfig,
        num_workers: DictConfig,
        proportion: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        mp20 = MP20(root=self.hparams.root)

        train = mp20[:_MP20_TRAIN_END]
        val   = mp20[_MP20_TRAIN_END:_MP20_VAL_END]
        test  = mp20[_MP20_VAL_END:]

        # Optionally use a subset (useful for quick smoke tests)
        p = self.hparams.proportion
        self.train_dataset = train[: int(len(train) * p)]
        self.val_dataset   = val  [: int(len(val)   * p)]
        self.test_dataset  = test [: int(len(test)  * p)]

        log.info(
            f"MP20 splits — train: {len(self.train_dataset)}, "
            f"val: {len(self.val_dataset)}, test: {len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size.train,
            num_workers=self.hparams.num_workers.train,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size.val,
            num_workers=self.hparams.num_workers.val,
            pin_memory=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size.test,
            num_workers=self.hparams.num_workers.test,
            pin_memory=False,
            shuffle=False,
        )