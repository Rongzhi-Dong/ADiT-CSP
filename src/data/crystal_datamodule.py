"""Crystal Structure Prediction datamodules — MP20, Perov5, Carbon24, MPTS52.


Hydra config _target_ values:
    src.data.crystal_datamodule.MP20DataModule
    src.data.crystal_datamodule.Perov5DataModule
    src.data.crystal_datamodule.Carbon24DataModule
    src.data.crystal_datamodule.MPTS52DataModule
"""

from typing import Optional

from lightning import LightningDataModule
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from src.data.components.all_datasets import MP20, Carbon24, MPTS52, Perov5
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# Standard CDVAE/DiffCSP split boundaries
# (index into the concatenated all.csv: train first, then val, then test)

_MP20_TRAIN_END     = 27137
_MP20_VAL_END       = 27137 + 9046      # 36183

_PEROV5_TRAIN_END   = 11356
_PEROV5_VAL_END     = 11356 + 3786      # 15142

_CARBON24_TRAIN_END = 6092
_CARBON24_VAL_END   = 6092 + 2031       # 8123

_MPTS52_TRAIN_END   = 24286
_MPTS52_VAL_END     = 24286 + 8095      # 32381


class MP20DataModule(LightningDataModule):
    """`LightningDataModule` for Crystal Structure Prediction on MP-20.

    Single-dataset datamodule — returns a single val/test dataloader so
    Lightning does NOT require `dataloader_idx` in `validation_step`.
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

class Perov5DataModule(LightningDataModule):
    """`LightningDataModule` for Crystal Structure Prediction on Perov-5.

    Single-dataset datamodule — returns a single val/test dataloader so
    Lightning does NOT require `dataloader_idx` in `validation_step`.
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
        perov5 = Perov5(root=self.hparams.root)

        train = perov5[:_PEROV5_TRAIN_END]
        val   = perov5[_PEROV5_TRAIN_END:_PEROV5_VAL_END]
        test  = perov5[_PEROV5_VAL_END:]

        p = self.hparams.proportion
        self.train_dataset = train[: int(len(train) * p)]
        self.val_dataset   = val  [: int(len(val)   * p)]
        self.test_dataset  = test [: int(len(test)  * p)]

        log.info(
            f"Perov-5 splits — train: {len(self.train_dataset)}, "
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


class Carbon24DataModule(LightningDataModule):
    """`LightningDataModule` for Crystal Structure Prediction on Carbon-24.

    Single-dataset datamodule — returns a single val/test dataloader so
    Lightning does NOT require `dataloader_idx` in `validation_step`.
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
        carbon24 = Carbon24(root=self.hparams.root)

        train = carbon24[:_CARBON24_TRAIN_END]
        val   = carbon24[_CARBON24_TRAIN_END:_CARBON24_VAL_END]
        test  = carbon24[_CARBON24_VAL_END:]

        p = self.hparams.proportion
        self.train_dataset = train[: int(len(train) * p)]
        self.val_dataset   = val  [: int(len(val)   * p)]
        self.test_dataset  = test [: int(len(test)  * p)]

        log.info(
            f"Carbon-24 splits — train: {len(self.train_dataset)}, "
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


class MPTS52DataModule(LightningDataModule):
    """`LightningDataModule` for Crystal Structure Prediction on MPTS-52.

    Single-dataset datamodule — returns a single val/test dataloader so
    Lightning does NOT require `dataloader_idx` in `validation_step`.
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
        mpts52 = MPTS52(root=self.hparams.root)

        train = mpts52[:_MPTS52_TRAIN_END]
        val   = mpts52[_MPTS52_TRAIN_END:_MPTS52_VAL_END]
        test  = mpts52[_MPTS52_VAL_END:]

        p = self.hparams.proportion
        self.train_dataset = train[: int(len(train) * p)]
        self.val_dataset   = val  [: int(len(val)   * p)]
        self.test_dataset  = test [: int(len(test)  * p)]

        log.info(
            f"MPTS-52 splits — train: {len(self.train_dataset)}, "
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