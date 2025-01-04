import h5py
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class H5Dataset(Dataset):
    def __init__(
        self, path_list, input_transform=None, target_transform=None, load_max=None
    ):
        super().__init__()
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        # Step 1: Compute the total number of rows
        total_rows = 0
        for path in tqdm(path_list, desc="Calculating total rows", leave=False):
            with h5py.File(path, "r") as f:
                dataset = f["x"]
                assert isinstance(dataset, h5py.Dataset)
                if load_max is None:
                    total_rows += dataset.shape[0]
                else:
                    total_rows += min(dataset.shape[0], load_max)
        
        # Step 2: Reserve a big tensor
        with h5py.File(path_list[0], "r") as f:
            dataset = f["x"]
            num_features = dataset.shape[1]  # Number of features in each row
        self.x = torch.empty((total_rows, num_features), dtype=torch.float32)
        
        # Step 3: Load data into the reserved tensor
        current_index = 0
        for path in tqdm(path_list, desc="Loading data", leave=False):
            with h5py.File(path, "r") as f:
                dataset = f["x"]
                assert isinstance(dataset, h5py.Dataset)
                if load_max is None:
                    loaded_dataset = dataset[:]
                else:
                    loaded_dataset = dataset[:load_max]
                new_tensor = torch.from_numpy(loaded_dataset).float()
                
                # Copy into the pre-allocated tensor
                rows = new_tensor.shape[0]
                self.x[current_index:current_index + rows] = new_tensor
                current_index += rows

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
