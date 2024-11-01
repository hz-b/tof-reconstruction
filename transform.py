import torch
from scipy.signal import wiener

class Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.reshape(-1, 16)
    
class DisableNeighborTOFs(torch.nn.Module):
    def __init__(self, min_disabled_tofs_count: int, max_disabled_tofs_count: int):
        super().__init__()
        self.min_disabled_tofs_count = min_disabled_tofs_count
        self.max_disabled_tofs_count = max_disabled_tofs_count
    
    def forward(self, img):
        img_copy = img.clone()
        tof_count = img.shape[-1]
        
        # Determine how many TOFs to disable
        disabled_tofs_count = torch.randint(
            self.min_disabled_tofs_count, self.max_disabled_tofs_count + 1, (1,), device=img.device
        )
        
        # Randomly pick the first TOF to disable
        initial_disabled_tof = torch.randint(0, tof_count, (1,), device=img.device)
        disabled_tofs = initial_disabled_tof  # Initialize as a single-element tensor
        
        # Available TOFs to choose from
        tof_list = torch.randperm(tof_count, device=img.device)
        tof_list = tof_list[tof_list != initial_disabled_tof]  # Remove the initial TOF from available ones

        # Disable neighboring TOFs in a circular manner
        for _ in range(int(disabled_tofs_count.item()) - 1):
            permuted_disabled_tofs = disabled_tofs[torch.randperm(len(disabled_tofs), device=img.device)]
            for current_tof in permuted_disabled_tofs:
                new_neighbor = (current_tof + 1) % tof_count
            
                if (disabled_tofs == new_neighbor).sum() == 0:  # Check if new_neighbor is not in disabled_tofs
                    disabled_tofs = torch.cat((disabled_tofs, new_neighbor.view(1)))
                    tof_list = tof_list[tof_list != new_neighbor]
                    break
        
        # Set the selected TOFs to zero in the copied image
        img_copy[:, disabled_tofs] = 0.0
        assert len(disabled_tofs) >=self.min_disabled_tofs_count and len(disabled_tofs) <= self.max_disabled_tofs_count
        return img_copy

class DisableOppositeTOFs(torch.nn.Module):
    def __init__(self, min_disabled_tofs_count: int, max_disabled_tofs_count: int):
        super().__init__()
        self.min_disabled_tofs_count = min_disabled_tofs_count
        self.max_disabled_tofs_count = max_disabled_tofs_count
    
    def forward(self, img):
        img_copy = img.clone()
        tof_count = img.shape[-1]
        
        # Determine how many TOFs to disable
        disabled_tofs_count = torch.randint(
            self.min_disabled_tofs_count, self.max_disabled_tofs_count + 1, (1,), device=img.device
        )
        
        # Randomly pick the first TOF to disable
        initial_disabled_tof = torch.randint(0, tof_count, (1,), device=img.device)
        disabled_tofs = initial_disabled_tof  # Initialize as a single-element tensor
        
        # Available TOFs to choose from
        tof_list = torch.randperm(tof_count, device=img.device)
        tof_list = tof_list[tof_list != initial_disabled_tof]  # Remove the initial TOF from available ones

        # Disable neighboring TOFs in a circular manner
        for _ in range(int(disabled_tofs_count.item()) - 1):
            permuted_disabled_tofs = disabled_tofs[torch.randperm(len(disabled_tofs), device=img.device)]
            opposite_found = False
            for current_tof in permuted_disabled_tofs:
                new_opposite = (current_tof + int(tof_count / 2)) % tof_count
            
                if (disabled_tofs == new_opposite).sum() == 0:  # Check if new_neighbor is not in disabled_tofs
                    disabled_tofs = torch.cat((disabled_tofs, new_opposite.view(1)))
                    tof_list = tof_list[tof_list != new_opposite]
                    opposite_found = True
                    break
            if not opposite_found:
                new_element = tof_list[0]
                tof_list = tof_list[tof_list != new_element]
                disabled_tofs = torch.cat((disabled_tofs, new_element.view(1)))
                    
        
        # Set the selected TOFs to zero in the copied image
        img_copy[:, disabled_tofs] = 0.0
        assert len(disabled_tofs) >=self.min_disabled_tofs_count and len(disabled_tofs) <= self.max_disabled_tofs_count
        return img_copy

class DisableRandomTOFs(torch.nn.Module):
    def __init__(
        self,
        min_disabled_tofs_count=1,
        max_disabled_tofs_count=3,
        neighbor_probability=0.5,
    ):
        super().__init__()
        self.max_disabled_tofs_count = max_disabled_tofs_count
        self.min_disabled_tofs_count = min_disabled_tofs_count
        self.neighbor_probability = neighbor_probability

    def forward(self, img):
        if self.min_disabled_tofs_count == 0 and self.max_disabled_tofs_count == 0:
            return img
        img_copy = img.clone()
        tof_count = img.shape[-1]
        self.disabled_tofs = []
        disabled_tofs_count = torch.randint(
            self.min_disabled_tofs_count, self.max_disabled_tofs_count + 1, (1,), device=img.device
        )

        # Randomly pick the first TOF to disable
        initial_disabled_tof = torch.randint(0, tof_count, (1,), device=img.device)
        disabled_tofs = initial_disabled_tof  # Initialize as a single-element tensor
        
        # Available TOFs to choose from
        tof_list = torch.randperm(tof_count, device=img.device)
        tof_list = tof_list[tof_list != initial_disabled_tof]  # Remove the initial TOF from available one

        for i in range(disabled_tofs_count - 1):
            random_variable = torch.rand(1, device=img.device)
            permuted_disabled_tofs = disabled_tofs[torch.randperm(len(disabled_tofs), device=img.device)]
            if random_variable < self.neighbor_probability:
                # neighbor or opposite
                if random_variable < self.neighbor_probability / 2:
                    # neighbor
                    for current_tof in permuted_disabled_tofs:
                        new_neighbor = (current_tof + 1) % tof_count
            
                        if (disabled_tofs == new_neighbor).sum() == 0:  # Check if new_neighbor is not in disabled_tofs
                            disabled_tofs = torch.cat((disabled_tofs, new_neighbor.view(1)))
                            tof_list = tof_list[tof_list != new_neighbor]
                            break
                else:
                    # opposite
                    opposite_found = False
                    for current_tof in permuted_disabled_tofs:
                        new_opposite = (current_tof + int(tof_count / 2)) % tof_count
            
                        if (disabled_tofs == new_opposite).sum() == 0:  # Check if new_neighbor is not in disabled_tofs
                            disabled_tofs = torch.cat((disabled_tofs, new_opposite.view(1)))
                            tof_list = tof_list[tof_list != new_opposite]
                            opposite_found = True
                            break
                    if not opposite_found:
                        new_element = tof_list[0]
                        tof_list = tof_list[tof_list != new_element]
                        disabled_tofs = torch.cat((disabled_tofs, new_element.view(1)))
            else:
                new_element = tof_list[0]
                tof_list = tof_list[tof_list != new_element]
                disabled_tofs = torch.cat((disabled_tofs, new_element.view(1)))
        assert disabled_tofs.shape[0] <= disabled_tofs_count
        assert torch.unique(disabled_tofs).shape[0] == disabled_tofs_count
        img_copy[:, disabled_tofs] = 0.0
        return img_copy


class DisableRandomChannel(torch.nn.Module):
    def __init__(self, max_disabled_channels_count=4):
        super().__init__()
        self.max_disabled_channels_count = max_disabled_channels_count

    def forward(self, img):
        img_copy = img.clone()
        channel_count = img.shape[-2]
        disabled_channels_count = torch.randint(
            1, self.max_disabled_channels_count, (1,)
        )
        disabled_channels = torch.randperm(channel_count)[:disabled_channels_count]
        img_copy[disabled_channels, :] = 0.0
        return img_copy


class DisableSpecificTOFs(torch.nn.Module):
    def __init__(self, disabled_tofs: list[int]):
        super().__init__()
        self.disabled_tofs = disabled_tofs

    def forward(self, img):
        img_copy = img.clone()
        img_copy[:, self.disabled_tofs] = 0.0
        return img_copy


class HotPeaks(torch.nn.Module):
    def __init__(self, peak_probability=0.1, distortion_factor=1.0):
        super().__init__()
        self.peak_probability = peak_probability
        self.distortion_factor = distortion_factor

    def forward(self, img):
        mask = torch.rand_like(img) > self.peak_probability
        return torch.where(
            mask, img, (img.max() - img.min()) * self.distortion_factor + img.min()
        )


class GaussianNoise(torch.nn.Module):
    def __init__(self, std=0.2):
        super().__init__()
        self.std = std

    def forward(self, img):
        return img + torch.randn_like(img) * torch.rand(1, device=img.device) * self.std


class PerImageNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img_min = img.min()
        img_max = img.max()
        if img_min == img_max:
            if img_min == 0.:
                return img
            else:
                print("Warning: Image contains only similar elements and is not 0. Cannot normalize.", img_min.item())
                return img
        return (img - img_min) / (img.max() - img_min)

class CircularPadding(torch.nn.Module):
    def __init__(self, padding: int):
        super().__init__()
        self.padding = padding

    def forward(self, img):
        return torch.nn.functional.pad(img, (self.padding,self.padding), "circular")

class PruneNegative(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return torch.where(img > 0, img, 0.0)

class Wiener(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img: torch.Tensor):
        if img.min() == img.max():
            return img
        device = img.device
        img = torch.from_numpy(wiener(img.detach().cpu().numpy())).float().to(device)
        return img
