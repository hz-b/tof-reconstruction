import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class H5Dataset(Dataset):
    def __init__(
        self, path_list, input_transform=None, target_transform=None, load_max=None
    ):
        super().__init__()
        self.input_transform = input_transform
        self.target_transform = target_transform
        for path in tqdm(path_list):
            with h5py.File(path, "r") as f:
                dataset = f["x"]
                assert isinstance(dataset, h5py.Dataset)
                if load_max is None:
                    loaded_dataset = dataset[:]
                else:
                    loaded_dataset = dataset[:load_max]
                new_tensor = torch.from_numpy(loaded_dataset).float()
                if hasattr(self, "x"):
                    self.x = torch.vstack([self.x, new_tensor])
                else:
                    self.x = new_tensor

    def __getitem__(self, idx):
        input = (
            self.input_transform(self.x[idx])
            if self.input_transform is not None
            else self.x[idx]
        )
        target = (
            self.target_transform(self.x[idx])
            if self.target_transform is not None
            else self.x[idx]
        )
        return input, target

    def __len__(self):
        return self.x.shape[0]
