import lightning as L
import torch
from torch.utils.data import DataLoader, random_split


class DefaultDataModule(L.LightningDataModule):

    def __init__(
        self,
        dataset,
        batch_size_train: int = 32,
        batch_size_val: int = 32,
        split: list[float] | None = None,
        num_workers: int = 0,
        on_gpu: bool = False,
        seed_split: int = 42,  # fixed seed to be used for random train-val-test split
        seed_train: int = 42,  # fixed seed to be used shuffle in train dataloader
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.dataset = dataset
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.split: list[float] = [0.8, 0.1, 0.1] if split is None else split
        self.num_workers = num_workers
        if on_gpu is not None:
            self.pin_memory = True
        else:
            self.pin_memory = False
        self.seed_train = seed_train
        self.seed_split = seed_split

    def setup(self, stage=None):
        train_size, val_size, _ = self.split
        train_size = int(train_size * len(self.dataset))
        val_size = int(val_size * len(self.dataset))
        test_size = len(self.dataset) - (train_size + val_size)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed_split),
        )

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=torch.Generator().manual_seed(self.seed_train),
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, max_len=None):
        assert self.test_dataset is not None
        if max_len is not None:
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, range(min(len(self.test_dataset), max_len)))
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
