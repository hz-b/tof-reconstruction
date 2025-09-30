import glob
import psutil
from scipy import stats
import os
import sys
import torch
import subprocess
import h5py
from torch.utils import benchmark
from datamodule import DefaultDataModule
from dataset import H5Dataset
from tof_reconstructor import TOFReconstructor
from transform import (
    DisableNeighborTOFs,
    DisableOppositeTOFs,
    DisableRandomTOFs,
    DisableSpecificTOFs,
    GaussianNoise,
    HotPeaks,
    Wiener,
    PerImageNormalize,
    Reshape,
    CircularPadding,
    ZeroTransform,
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import Compose
from tqdm import trange, tqdm
from data_generation import Job
from scipy.stats import ttest_rel
import numpy as np
import pickle
from pacman import PacMan
import time

torch.manual_seed(42)


class MeanModel(torch.nn.Module):
    def __init__(self, tof_count, device):
        super().__init__()
        self.tof_count = tof_count
        self.device = device
        self.padding = 0

    def forward(self, input):
        return torch.stack(
            [self.process_image(image.reshape(-1, self.tof_count)) for image in input],
            dim=0,
        )

    def process_image(self, image):
        image_copy = image.clone()
        disabled_tofs = self.get_disabled_tofs(image)
        for disabled_tof in disabled_tofs:
            left, right = self.get_left_right_neighbors(
                disabled_tof, disabled_tofs, image.shape[1]
            )
            image_copy[:, disabled_tof] = (
                image_copy[:, left] + image_copy[:, right]
            ) / 2
        return image_copy.flatten()

    def get_disabled_tofs(self, image):
        return torch.arange(0, image.shape[1], device=image.device)[image.sum(dim=0) == 0.0]

    def get_left_right_neighbors(self, disabled_tof, disabled_tofs, tof_count):
        for i in range(tof_count):
            left = (disabled_tof + i) % tof_count
            if left not in disabled_tofs:
                break
        for i in range(tof_count):
            right = (disabled_tof - i) % tof_count
            if right not in disabled_tofs:
                break
        return left, right

class Evaluator:
    def __init__(
        self,
        model_dict: dict,
        device: torch.device = torch.get_default_device(),
        output_dir: str = "outputs/",
        dataset = "datasets/sigmaxy_7_peaks_0_20_hot_15/shuffled_*.h5",
        load_max=None,
        pac_man=False
    ):
        self.device = device
        for key, value in model_dict.items():
            model_dict[key] = self.load_eval_model(Evaluator.load_first_ckpt_file(value))
            
        self.model_dict: dict = model_dict
        self.model_dict["Neighboring Mean"] = MeanModel(16, device=device)
        if pac_man:
            self.model_dict["Pacman"] = PacMan()

        self.initial_input_transforms = [
            Reshape(),
            HotPeaks(0.1, 1.0),
            PerImageNormalize(),
            GaussianNoise(0.1),
            PerImageNormalize(),
        ]
        self.output_dir = output_dir

        target_transform = Compose(
            [
                Reshape(),
                PerImageNormalize(),
            ]
        )
        if dataset is not None:
            self.dataset = H5Dataset(
                path_list=list(glob.iglob("datasets/sigmaxy_7_peaks_0_20_hot_15/shuffled_*.h5")),
                input_transform=None,
                target_transform=target_transform,
                load_max=load_max,
            )
            print(len(self.dataset))
    @staticmethod
    def load_first_ckpt_file(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)
        # Filter files that end with '.ckpt'
        ckpt_files = [file for file in files if file.endswith('.ckpt')]
        if len(ckpt_files) > 1:
            raise FileNotFoundError("Multiple .ckpt files, cannot decide.")
        # Sort the files alphabetically (optional, to ensure consistent order)
        ckpt_files.sort()
        # Return the first file if available
        if ckpt_files:
            return os.path.join(folder_path, ckpt_files[0])
        else:
            raise FileNotFoundError("No .ckpt files found in the specified folder.")

    def load_eval_model(self, model):
        model = TOFReconstructor.load_from_checkpoint(model, channels=60)
        model.eval()
        model.to(self.device)
        return model
    
    def evaluate_transform_normalized(self, model_keys, transform):
        output_dict = {}
        for key in model_keys:
            model = self.model_dict[key]
            input_transform = Compose(
                        self.initial_input_transforms
                        + [
                            transform,
                            PerImageNormalize(),
                            CircularPadding(model.padding),
                        ]
                    )
            output_dict[key] = self.evaluate_rmse(key, input_transform)
        
        return output_dict

    def evaluate_n_disabled_tofs(self, model_keys, disabled_tof_count=3):
        return self.evaluate_transform_normalized(model_keys, DisableRandomTOFs(disabled_tof_count, disabled_tof_count))

    def evaluate_1_n_disabled_tofs(self, model_keys, n=3):
        return self.evaluate_transform_normalized(model_keys, DisableRandomTOFs(1, n, neighbor_probability=1))

    def evaluate_specific_disabled_tofs(self, model_keys, disabled_list):
        return self.evaluate_transform_normalized(model_keys, DisableSpecificTOFs(disabled_list))

    def evaluate_neigbors(self, model_keys, min_disabled_count, max_disabled_count):
        return self.evaluate_transform_normalized(model_keys, DisableNeighborTOFs(min_disabled_count, max_disabled_count))
    
    def evaluate_opposite(self, model_keys, min_disabled_count, max_disabled_count):
        return self.evaluate_transform_normalized(model_keys, DisableOppositeTOFs(min_disabled_count, max_disabled_count))
    
    @staticmethod
    def significant_confidence_levels(group_A, group_B, confidence=0.99):
        ci = ttest_rel(group_A.flatten().cpu(), group_B.flatten().cpu()).confidence_interval(confidence_level=confidence)
        confidence_interval = (ci.low.item(), ci.high.item())
        return not (confidence_interval[0] < 0. and confidence_interval[1] > 0.), confidence_interval

    @staticmethod
    def result_dict_to_latex(result_dict, statistics_table=True):
        if len(result_dict) < 4:
            alignment = "c" * len(result_dict)
            table_environment = "tabular"
        else:
            alignment = r"""*{"""+str(len(result_dict))+r"""}{>{\centering\arraybackslash}X}"""
            table_environment = "tabularx"
        
        if table_environment =="tabularx":
            text_width =  r"""{\textwidth}"""
        else:
            text_width = ""
        if statistics_table:
            first_column_width = "1.5cm"
        else:
            first_column_width = "2.5cm"
        output_string = (
            r"""
        \begin{"""+table_environment+r"""}"""+text_width+r"""{p{"""+first_column_width+"""}|"""+
        alignment    
            + r"""}
        \hline"""
            + "\n"
        )

        keys = [k for k in result_dict.keys()]
        keys = ["Failed \\acp{TOF}"] + [entry.split()[0] for entry in list(result_dict.keys())]
        keys2 = ["scenario"] + [entry.split()[1] for entry in list(result_dict.keys())]
        for entry in [keys, keys2]:
            output_string += " & ".join(entry) + r" \\" + "\n" 
        output_string += r"\midrule" + "\n"

        model_keys = list(list(result_dict.values())[0].keys())

        for model_key in model_keys:
            model_row = [model_key]
            for scenario_value in result_dict.values():
                best_key = min(scenario_value, key=scenario_value.get)
                std_dev = scenario_value[model_key][1].std()
                if statistics_table:
                    model_row_element = f"{scenario_value[model_key][0]:.2e}".replace("e+0", "e+").replace("e-0", "e-")+f" $\\pm${std_dev:.2e}".replace("e+0", "e+").replace("e-0", "e-")
                else:
                    val_scaled = scenario_value[model_key][0] * 1e2
                    model_row_element = f"{val_scaled:.2f}"
                    if best_key == model_key:
                        model_row_element = r"\mathbf{" + model_row_element + r"}"
                    model_row_element = "$"+model_row_element+"$"
                if best_key == model_key:
                    if statistics_table:
                        model_row_element = r"\textbf{" + model_row_element + r"}"
                    else:
                        model_row_element += r" ~"
                else:
                    if statistics_table:
                        p_value = Evaluator.significant_confidence_levels(scenario_value[best_key][1], scenario_value[model_key][1])[1]
                        model_row_element += f" ({p_value[0]:.2e}, {p_value[1]:.2e})".replace("e+0", "e+").replace("e-0", "e-")
                    significant = Evaluator.significant_confidence_levels(scenario_value[best_key][1], scenario_value[model_key][1])[0]
                    if significant and not statistics_table:
                        model_row_element += " $\\dagger$"
                    if not significant and not statistics_table:
                        model_row_element += r" ~"
                model_row += [model_row_element]
            output_string += " & ".join(model_row) + r" \\" + "\n"
            if statistics_table:
                output_string += r"""\midrule""" + "\n"
        if not statistics_table:
            output_string += r"""\bottomrule""" + "\n"
        output_string += r"""\end{"""+table_environment+r"""}"""
        return output_string

    def test_with_input_transform(self, input_transform, batch_size=8192, max_len=None):
        workers = psutil.Process().cpu_affinity()
        num_workers = len(workers) if workers is not None else 0
        self.dataset.input_transform = input_transform
        datamodule = DefaultDataModule(dataset=self.dataset, batch_size_val=batch_size, num_workers=num_workers, on_gpu=(self.device.type=='cuda'))
        datamodule.setup()
        test_dataloader = datamodule.test_dataloader(max_len=max_len)
        return test_dataloader

    def evaluate_missing_tofs(self, disabled_tofs, model_label):
        model = self.model_dict[model_label]
        input_transform = Compose(
            self.initial_input_transforms
            + [
                DisableSpecificTOFs(disabled_tofs=disabled_tofs),
                PerImageNormalize(),
                CircularPadding(model.padding),
            ]
        )
        mean, _ = self.evaluate_rmse(model_label, input_transform)
        return mean

    def evaluate_rmse(self, model_label, input_transform):
        test_dataloader = self.test_with_input_transform(input_transform)
        test_loss_list = []
        tof_loss_list = []
        for x, y in tqdm(test_dataloader, leave=False):
            test_loss = self.calculate_losses(model_label, x, y)
            test_loss_list.append(test_loss)
        test_loss_tensor = torch.cat(test_loss_list)
        return test_loss_tensor.mean(), test_loss_tensor.flatten()

    def evaluate_extended_rmse(self, model_label, input_transform, pacman_limit=None):
        test_dataloader = self.test_with_input_transform(input_transform)
        
        test_diff_list = []
        z_diff_list = []
        tof_diff_list = []
        z_tof_diff_list = []
        
        for x, y in tqdm(test_dataloader, leave=False):
            if model_label == "Pacman":
                x = x[:pacman_limit]
                y = y[:pacman_limit]
            test_diff, z_diff, tof_diff, z_tof_diff = self.calculate_losses(model_label, x, y, extended_losses=True)
            
            test_diff_list.append(test_diff)
            z_diff_list.append(z_diff)
            tof_diff_list.append(tof_diff)
            z_tof_diff_list.append(z_tof_diff)
            if model_label == "Pacman":
                if pacman_limit is not None and len(test_diff_list) * x.shape[0] >= pacman_limit:
                    break
                
        
        # Concatenate all loss tensors
        test_diff_tensor = torch.cat(test_diff_list)
        z_diff_tensor = torch.cat(z_diff_list)
        tof_diff_tensor = torch.cat(tof_diff_list)
        z_tof_diff_tensor = torch.cat(z_tof_diff_list)
        
        return {
            "test_loss": test_diff_tensor,
            "z_loss": z_diff_tensor,
            "tof_loss": tof_diff_tensor,
            "z_tof_loss": z_tof_diff_tensor,
        }
    
    @staticmethod
    def z_score(x):
        # Compute mean and std per sample (dims except dim=0)
        dims = tuple(range(1, x.ndim))
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)

        # Avoid division by zero: set std to 1 where it's zero
        safe_std = std.clone()
        safe_std[safe_std == 0] = 1.

        # Z-score
        z = (x - mean) / safe_std

        # Set output to 0 where std was originally zero (i.e., constant inputs)
        zero_std_mask = (std == 0)
        z = z * (~zero_std_mask)  # this works due to broadcasting

        return z

    @staticmethod
    def min_max(x):
        # Compute min and max per sample
        min_val = x.min(dim=1, keepdim=True).values
        max_val = x.max(dim=1, keepdim=True).values
        return (x - min_val) / (max_val - min_val)


    def calculate_losses(self, model_label, x, y, extended_losses=False):
        model = self.model_dict[model_label]
        channels = x.shape[-2]
        tof_count = x.shape[-1] - 2 * model.padding
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        with torch.no_grad():
            y_hat = model(x.to(model.device))
        if model_label == "Pacman":
            y_hat = y_hat[0]

        y_hat = y_hat.to(torch.device('cpu'))

        y_hat = y_hat.reshape(-1, channels, tof_count + 2*model.padding)
        if model.padding != 0:
            y_hat = y_hat[:, :, model.padding:-model.padding]
        y_hat = y_hat.flatten(start_dim=1)
        
        y_hat_copy = y_hat.clone()
        if model_label == "Pacman":
            y_hat = Evaluator.min_max(y_hat)
        
        if extended_losses:
            test_diff = y_hat - y
            z_diff = Evaluator.z_score(y_hat_copy) - Evaluator.z_score(y)
            # now we need a mask to regain the disabled TOFs
            # Condition 1: They need all to be equal
            x_orig = x.reshape(-1, channels, tof_count)
            y_orig = y.reshape(-1, channels, tof_count)
            y_hat_orig = y_hat.reshape(-1, channels, tof_count)
            y_hat_copy_orig = y_hat_copy.reshape(-1, channels, tof_count)

            any_diff = (x_orig != y_orig).any(dim=1)  

            # Condition 2: All values in x are zero in the column
            all_y_zero = (x_orig == 0).all(dim=1) 
            # Combine both conditions
            mask = any_diff & all_y_zero


            mask_expanded = mask.unsqueeze(1).expand(-1, y_orig.shape[1], -1)
            y_selected = y_orig[mask_expanded]
            y_hat_selected = y_hat_orig[mask_expanded]
            y_hat_copy_selected = y_hat_copy_orig[mask_expanded]
            
            tof_y = y_selected
            tof_y_hat = y_hat_selected
            tof_y_hat_copy = y_hat_copy_selected
            tof_diff = (tof_y_hat - tof_y).flatten()
            z_tof_diff = (Evaluator.z_score(tof_y_hat) - Evaluator.z_score(tof_y)).flatten()
            
            tof_loss = (tof_diff.reshape(-1, channels)**2).mean(dim=-1)
            z_tof_loss = (z_tof_diff.reshape(-1, channels)**2).mean(dim=-1)
            
            test_loss = (test_diff.reshape(-1, channels, tof_count)**2).mean(dim=[-1,-2])
            z_loss = (z_diff.reshape(-1, channels, tof_count)**2).mean(dim=[-1,-2])
            
            return test_loss, z_loss, tof_loss, z_tof_loss
        else:
            return torch.sqrt(torch.nn.functional.mse_loss(y_hat, y, reduction='none').mean(dim=-1))

    def two_missing_tofs_rmse_matrix(self, model_label, job_id=None, total_job_count=None):
        model = self.model_dict[model_label]
        save_file = lambda x,y: 'outputs/rmse_matrix_'+str(x)+'_'+str(y)+'.pt'
        with torch.no_grad():
            output_matrix = torch.full((16, 16), 0., device=model.device)
            evaluation_list = []
            for i in trange(output_matrix.shape[0]):
                for j in range(output_matrix.shape[1]):
                    if i >= j:
                        continue
                    else:
                        evaluation_list.append((i,j))
            full_evaluation_list = evaluation_list.copy()

            if job_id is not None and total_job_count is not None:
                assert job_id < total_job_count
                normal_length = len(evaluation_list) // total_job_count
                last_length = len(evaluation_list) - normal_length * (total_job_count - 1)
                length = normal_length if job_id != total_job_count - 1 else last_length
                evaluation_list = evaluation_list[job_id*length:job_id*length+length]
            
            for i, j in tqdm(evaluation_list):
                new_entry = self.evaluate_missing_tofs(
                        [i, j], model_label
                )
                torch.save(new_entry, save_file(i,j))
            if all([os.path.exists(save_file(i,j)) for i, j in full_evaluation_list]):
                for i,j in full_evaluation_list:
                    output_matrix[i][j] = torch.load(save_file(i,j))
                return output_matrix
    
            return None

    def one_missing_tof_rmse_tensor(self, model_label):
        with torch.no_grad():
            outputs = []
            for i in trange(16):
                outputs.append(
                    self.evaluate_missing_tofs([i], model_label)
                )
            return torch.tensor(outputs, device=self.device)

    def plot_rmse_matrix(self, matrix, diag):
        # scale to units of 10^-2
        matrix = matrix * 100
        diag = diag * 100
        f = plt.figure(figsize=(8, 6), constrained_layout=True)
        plt.matshow((matrix+matrix.T+torch.diag(diag)).cpu(), fignum=plt.get_fignums()[-1], cmap="hot", aspect="auto")
        plt.xticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=15)
        plt.yticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=15)
        plt.grid(alpha=0.7)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        cb.ax.set_ylabel('RMSE', fontsize=20)
        plt.xlabel("TOF position [#]", fontsize=20)
        plt.ylabel("TOF position [#]", fontsize=20)
        plt.savefig(self.output_dir + "2_tof_failed.pdf")

    @staticmethod
    def retrieve_spectrogram_detector(kick_min=0, kick_max=100, peaks=5, seed=42, hot_enabled=False, ellipt=0.73, pulse=7):
        output = Job([1, kick_min, kick_max, peaks, ellipt, (90 - 22.5) / 180 * np.pi, pulse, seed, hot_enabled, False, None])
        assert output is not None
        X, Y = output
        return X, Y

    @staticmethod
    def spec_detector_image_ax(ax, X, fontsize, residual=False):
        out = ax.imshow(np.array(X), aspect=X.shape[1] / X.shape[0], cmap='hot', interpolation="none", origin="lower")
        ax.set_ylabel("Kinetic energy [eV]",fontsize=fontsize)
        ax.set_xticks(range(0, 16, 5), [str(i) for i in range(1, 17, 5)],fontsize=20)
        ax.set_xlabel("TOF position [#]",fontsize=fontsize)
        title = "Residual image" if residual else "Detector image"
        ax.set_title(title,fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_yticks(ticks=range(0, 70, 10), labels=range(280, 350, 10))
        ax.spines[['right', 'top']].set_visible(False)
        out.set_clim(vmin=0, vmax=1)
        return out

    @staticmethod
    def save_spectrogram_detector_image_plot(X, Y, output_path, Z=None):
        import matplotlib.pyplot as plt
        import numpy as np
    
        fontsize = 20
    
        # Determine how many plots we need
        num_images = 2 if Z is None else 3
    
        # Create figure and axes
        fig, ax = plt.subplots(1, num_images, figsize=(5 * num_images, 4), gridspec_kw={'wspace': 0.3})
    
        # If only one image, ax is not a list – force it into a list
        if num_images == 1:
            ax = [ax]
    
        # Plot Spectrogram (first image)
        im0 = ax[0].imshow(Y, aspect='auto', cmap='hot', interpolation="none", origin="lower")
        ax[0].set_ylabel("Photon energy [eV]", fontsize=fontsize)
        ax[0].set_title('Spectrogram', fontsize=fontsize)
        ax[0].set_xlabel('Time [steps]', fontsize=fontsize)
        ax[0].set_xticks(range(0, 100, 20), labels=range(0, 100, 20), fontsize=fontsize)
        ax[0].tick_params(axis='both', labelsize=fontsize)
        ax[0].set_yticks(ticks=list(range(0, 61, 10)) + [60], labels=list(range(1150, 1220, 10)) + [1210])
        ax[0].spines[['right', 'top']].set_visible(False)
    
        # Plot Detector Image (second image)
        out = Evaluator.spec_detector_image_ax(ax[1], X, fontsize)
    
        # If Z is provided, plot it in third subplot
        if Z is not None:
            out = Evaluator.spec_detector_image_ax(ax[2], Z, fontsize, residual=True)
    
        # Add colorbar to the last subplot (regardless of 2 or 3 images)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(out, cax=cbar_ax)
        cbar.set_label('Intensity [arb.u.]', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)
    
        # Adjust layout to leave room for colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    
    def plot_spectrogram_detector_image(self, peaks=5, seed=42, hot_enabled=False, ellipt=0.73, pulse=7):
        X, Y = self.retrieve_spectrogram_detector(peaks=peaks, seed=seed, pulse=pulse, hot_enabled=hot_enabled, ellipt=ellipt)
        label_appendix="_hot" if hot_enabled else ""
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        
        Evaluator.save_spectrogram_detector_image_plot(X, Y, self.output_dir + 'spectrogram_detector_image_'+str(peaks)+'_'+str(seed)+label_appendix+'.pdf')


    def plot_rmse_tensor(self, rmse):
        fig, ax1 = plt.subplots(figsize=(16, 4), constrained_layout=True)
        # Example vectors
        x = np.arange(16)  # Shared x-axis
        #y1 = np.random.random(16) * 100  # Vector 1 with one scale
        ang_dist = np.array([0., 0.14644661, 0.5, 0.85355339, 1., 0.85355339,
                             0.5, 0.14644661, 0., 0.14644661, 0.5, 0.85355339,
                             1., 0.85355339, 0.5, 0.14644661])  # Vector 2 with a different scale

        # Use colors from the default Matplotlib color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color1 = colors[0]  # First color in the cycle
        color2 = colors[1]  # Second color in the cycle

        # Plot the first vector (y1) with its scale on the primary y-axis
        plt.xticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=20)
        line1, = ax1.plot(x, rmse*100, color=color1, label='RMSE')  # Set zorder for line1
        ax1.set_xlabel('TOF position [#]', fontsize=20)
        ax1.set_ylabel('RMSE', color=color1, fontsize=20)
        plt.yticks(fontsize=20)
        ax1.tick_params(axis='y', labelcolor=color1)

        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot the second vector (ang_dist) with its scale on the secondary y-axis
        line2, = ax2.plot(x, ang_dist, color=color2, label='Angular Distribution', zorder=1)  # Set zorder for line2
        ax2.set_ylabel('Angular Distribution', color=color2, fontsize=20)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add a single legend and place it inside the plot
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax2.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1), frameon=True, fontsize=20)
        #plt.yscale('log')
        plt.yticks(fontsize=20)
        plt.grid(alpha=0.8)

        # Show the plot
        plt.tight_layout()

        plt.savefig(self.output_dir + "1_tof_failed.pdf")


    @staticmethod
    def detector_image_ax(ax, data, title, aspect_auto=True, additional_text=""):
        ax.set_xlabel("TOF position [#]"+additional_text)
        ax.set_title(title)
        ax.set_xticks(range(0, 16, 5), [str(i) for i in range(1, 17, 5)])
        ax.set_yticks(ticks=list(range(0, 70, 10))+[60], labels=list(range(280, 350, 10))+ [340])
        return ax.imshow(
            data,
            aspect='auto' if aspect_auto else data.shape[1] / data.shape[0],
            interpolation="none",
            cmap="hot",
            origin="lower",
        )
    
    @staticmethod
    def pacman_spectrogram_simulation(pacman, peaks, seed, its_override=None, hot_enabled=False):
        X, Y = Evaluator.retrieve_spectrogram_detector(peaks=peaks, seed=seed, hot_enabled=hot_enabled)
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        out = pacman(torch.tensor(X).unsqueeze(0), its_override=its_override)
        return X, Y, out


    @staticmethod
    def plot_detector_image_comparison(data_list, title_list, filename, output_dir, show_rmse=False, label_index=0, tof_rmse:list=None, min_max_normalize_labels=["Pacman"]):
        min_max = lambda x: (x - x.min()) / (x.max() - x.min())
        z_score = lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0
        if len(data_list) > 3:
            if len(data_list) == 4:
                columns = 2
            else:
                columns = 3
            rows = len(data_list) // columns + (len(data_list) % columns != 0)
        else:
            rows = 1
            columns = len(data_list)
        fig, ax = plt.subplots(
            rows, columns, sharex=False, sharey=True, squeeze=False, figsize=(8, 3+rows*1.2), constrained_layout=True
        )

        for i in range(len(data_list)):
            cur_row = i // columns
            cur_col = i % columns
            if cur_col == 0:
                ax[cur_row, cur_col].set_ylabel("Kinetic energy [eV]")
            ax[cur_row, cur_col].spines[['right', 'top']].set_visible(False)
            
            
            additional_text = ""
            # TOF RMSE text
            if i != label_index and show_rmse:
                data_list_entry = data_list[i]
                if title_list[i] in min_max_normalize_labels:
                    data_list_entry = min_max(data_list_entry)
                diff = data_list[label_index] - data_list_entry

                diff_z = z_score(data_list[label_index]) - z_score(data_list[i])
                rmse_z = torch.sqrt(torch.mean(diff_z**2))
                mse = torch.mean(diff ** 2)
                rmse = torch.sqrt(mse)
                additional_text += f'\nRMSE: {rmse.item()*100:.2f} Z-RMSE: {rmse_z.item():.2f}'
            if i!= label_index and tof_rmse is not None:
                diff = data_list[label_index] - data_list_entry
                diff = diff[tof_rmse]
                
                diff_z_tof = diff_z[tof_rmse]
                rmse_z_tof = (diff_z_tof ** 2).mean().sqrt()
                
                mse = torch.mean(diff ** 2)
                rmse = torch.sqrt(mse)
                add_sep = " " if columns < 3 else "\n"
                additional_text += f'\nT-RMSE: {rmse.item()*100:.2f}'+add_sep+f'TZ-RMSE: {rmse_z_tof.item():.2f}'               

            out = Evaluator.detector_image_ax(ax[cur_row, cur_col], min_max(data_list[i]), title_list[i], rows!=1, additional_text=additional_text)
            ax[cur_row, cur_col].set_yticks(ticks=range(0, 70, 10), labels=range(280, 350, 10))

        for i in range(len(data_list), rows*columns):
            cur_row = i // columns
            cur_col = i % columns
            ax[cur_row, cur_col].set_visible(False)
        out.set_clim(vmin=0, vmax=1)

        plt.tight_layout()
        fig.colorbar(out, ax=ax, shrink=0.49, label='Intensity [arb.u.]')
        plt.savefig(output_dir + filename + ".pdf", dpi=300, bbox_inches="tight")

    def plot_missing_tofs_comparison(self, disabled_tofs, batch_id=1):
        input_transform = Compose(
            self.initial_input_transforms
            + [
                DisableSpecificTOFs(disabled_tofs=disabled_tofs),
                PerImageNormalize(),
            ]
        )
        test_dataloader = self.test_with_input_transform(input_transform)
        with torch.no_grad():
            i = 0
            for x, y in tqdm(test_dataloader):
                i += 1
                if i == batch_id:
                    Evaluator.plot_detector_image_comparison(
                        [x[0], y[0]],
                        ["Sample with noise", "Sample without noise"],
                        "two_tofs_disabled",
                        self.output_dir,
                    )
                    break

    def plot_reconstructing_tofs_comparison(
    self, disabled_tofs, model_labels, batch_id=1, sample_id=0, show_rmse=True,
    tof_rmse_list=None
    ):
        if not isinstance(model_labels, list):
            raise ValueError("model_labels must be a list of model names (strings).")
        
        # We'll assume same padding across all models for simplicity
        padding = self.model_dict[model_labels[0]].padding
    
        input_transform = Compose(
            self.initial_input_transforms
            + [
                DisableSpecificTOFs(disabled_tofs=disabled_tofs),
                PerImageNormalize(),
                CircularPadding(padding),
            ]
        )
    
        test_dataloader = self.test_with_input_transform(input_transform)
    
        with torch.no_grad():
            i = 0
            for x, y in test_dataloader:
                i += 1
                if i == batch_id:
                    # Prepare input
                    input_sample = x[sample_id].unsqueeze(0).flatten(start_dim=1).to(self.device)
                    noisy_image = x[sample_id].cpu()
    
                    # Store all reconstructions
                    reconstructions = []
                    model_names = []
    
                    for idx, model_label in enumerate(model_labels):
                        model = self.model_dict[model_label]
                        model_padding = model.padding
                        
                        if model_padding != padding:
                            raise ValueError(f"All models must use the same padding for fair comparison. Mismatch found in {model_label}.")
    
                        output = self.evaluate_model(input_sample, model_label)[0]
                        output = output.reshape(-1, 16 + 2 * padding).unsqueeze(0)
    
                        if padding != 0:
                            output = output[:, :, padding:-padding]
    
                        reconstructions.append(output[0].cpu())
                        model_names.append(model_label)
    
                    # Crop noisy image if padding was applied
                    if padding != 0:
                        noisy_image = noisy_image[:, padding:-padding]
    
                    # Prepare all images to plot: [input, label, model1, model2, ...]
                    images = [noisy_image, y[sample_id].cpu()] + reconstructions
                    labels = ["With noise", "Label"] + model_names
                    label_index=1

                    model_part = "_".join(model_names)
                    filename = f"multi_model_comparison_{model_part}_b{batch_id}_s{sample_id}"
                    
                    Evaluator.plot_detector_image_comparison(
                        images,
                        labels,
                        filename,
                        self.output_dir,
                        show_rmse=show_rmse,
                        label_index=label_index,
                        tof_rmse=tof_rmse_list,
                    )
                    break

                    
    def plot_single_tof_reconstructing(
        self, model_label, batch_id=1, sample_id=0
    ):
        noisy_rmse_list = []
        model_rmse_list = []
        rec_noisy_rmse_list = []
        rec_model_rmse_list = []
        for disabled_tof in range(16):
            input_transform = Compose(
                self.initial_input_transforms
                + [
                    DisableSpecificTOFs(disabled_tofs=[disabled_tof]),
                    PerImageNormalize(),
                    CircularPadding(self.model_dict[model_label].padding),
                ]
            )
            padding = self.model_dict[model_label].padding
            test_dataloader = self.test_with_input_transform(input_transform)
            with torch.no_grad():
                i = 0
                for x, y in test_dataloader:
                    i += 1
                    if i == batch_id:
                        z = (
                            self.evaluate_model(x[sample_id].unsqueeze(0).flatten(start_dim=1).to(self.device), model_label)[0]
                            .reshape(-1, 16 + 2*padding)
                            .unsqueeze(0)
                        )
                        noisy_image = x[sample_id].cpu()
                        if padding != 0:
                            noisy_image = noisy_image[:, padding:-padding]
                            z = z[:,:,padding:-padding]
                            
                        label = y[sample_id].cpu()
                        model_output = z[0].cpu()
                        noisy_rmse_list.append(self.rmse(noisy_image, label))
                        model_rmse_list.append(self.rmse(model_output, label))
                        rec_noisy_rmse_list.append(self.rmse(noisy_image[disabled_tof], label[disabled_tof]))
                        rec_model_rmse_list.append(self.rmse(model_output[disabled_tof], label[disabled_tof]))
                        break
        return noisy_rmse_list, model_rmse_list, rec_noisy_rmse_list, rec_model_rmse_list
        
    def plot_single_tof_reconstructing_batched(
        self, model_label, max_len=10
    ):
        noisy_rmse_list = []
        model_rmse_list = []
        rec_noisy_rmse_list = []
        rec_model_rmse_list = []
        for disabled_tof in range(16):
            input_transform = Compose(
                self.initial_input_transforms
                + [
                    DisableSpecificTOFs(disabled_tofs=[disabled_tof]),
                    PerImageNormalize(),
                    CircularPadding(self.model_dict[model_label].padding),
                ]
            )
            padding = self.model_dict[model_label].padding
            test_dataloader = self.test_with_input_transform(input_transform)
            with torch.no_grad():
                noisy_image = []
                model_output = []
                label = []
                for x, y in test_dataloader:
                    z = (
                        self.evaluate_model(x.to(self.device).flatten(start_dim=1), model_label)
                        .reshape(-1, 60, 16 + 2*padding)
                    )
                    noisy_image.append(x)
                    if padding != 0:
                        noisy_image = noisy_image[:, padding:-padding]
                        z = z[:,:,padding:-padding]
                        
                    label.append(y)
                    model_output.append(z.cpu())
                label = torch.vstack(label)
                noisy_image = torch.vstack(noisy_image)
                model_output = torch.vstack(model_output)
                print(label.shape)
                
                noisy_rmse_list.append(self.rmse(noisy_image, label).item())
                model_rmse_list.append(self.rmse(model_output, label).item())
                rec_noisy_rmse_list.append(self.rmse(noisy_image[disabled_tof], label[disabled_tof]).item())
                rec_model_rmse_list.append(self.rmse(model_output[disabled_tof], label[disabled_tof]).item())

        return noisy_rmse_list, model_rmse_list, rec_noisy_rmse_list, rec_model_rmse_list
        
    @staticmethod
    def rmse(a, b):
        diff = a - b
        mse = torch.mean(diff**2)
        return torch.sqrt(mse)

    def evaluate_model(self, data, model_label):
        assert self.model_dict[model_label] is not None
        with torch.no_grad():
            return self.model_dict[model_label](data)

    @staticmethod
    def plot_gasdet_electron_int(
        data_path="datasets/210.hdf5", 
        sample_count=None, 
        hdf_attribute="gasdet_after_att_mJ"
    ):
        # Load data from HDF5
        f = h5py.File(data_path, 'r')
        imgs = f['acq_mV'][:sample_count]
        gmd = f[hdf_attribute][:sample_count]
        X = gmd[:, 0]
        Y = [np.sum(imgs[i]) for i in range(imgs.shape[0])]
    
        # Convert to numpy arrays
        x = np.array(X)
        y = np.array(Y)
    
        # Mask to filter data
        mask = (x > 0.02) & (y > 9000)
        x_cut = x[mask]
        y_cut = y[mask]
    
        slope, intercept = np.polyfit(x_cut, y_cut, 1)
        x_plot = np.concatenate(([0], x))
    
        # Define fit line for visualization
        y_fit = slope * x_plot# + intercept
    
        # Colors for scatter and fit line
        colors = plt.cm.tab10.colors
        scatter_color = colors[0]
        line_color = colors[1]
    
        # Plotting
        plt.scatter(x, y-intercept, color=scatter_color, s=0.8, alpha=0.8, label='Baseline Corrected Data')
        plt.plot(x_plot, y_fit, color=line_color, label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
        plt.xlim(0, None)
        
    
        plt.xlabel('Gas Monitor Detector [mJ]')
        plt.ylabel('Electron Intensity [arb.u.]')
        plt.savefig('outputs/saturation.png', bbox_inches='tight')
        plt.show()

    def eval_real_rec(self, sample_limit, model_label, input_transform=None, output_transform=None, tofs_to_evaluate=None):
        real_images = TOFReconstructor.get_real_data(
                0, sample_limit, "datasets/210.hdf5"
        )
        if model_label == None:
            padding = 0
        else:
            padding = self.model_dict[model_label].padding
        circular_transform = CircularPadding(padding)
        if input_transform is not None:
            composed_transform = Compose([input_transform, circular_transform])
        else:
            composed_transform = circular_transform

        if model_label == None:
            eval_func = lambda data: data
        else:
            eval_func = lambda data: self.evaluate_model(data, model_label)
        real_images, evaluated_real_data = TOFReconstructor.evaluate_real_data(
                real_images.to(self.device), eval_func, composed_transform
            )
        if padding != 0:
                real_images = real_images[...,padding:-padding]
                evaluated_real_data = evaluated_real_data[...,padding:-padding]
        if output_transform is not None:
            real_images = output_transform(real_images)
        tofs_to_evaluate = slice(None) if tofs_to_evaluate is None else tofs_to_evaluate
        
        return torch.sqrt(((real_images[:,tofs_to_evaluate]-evaluated_real_data[:, tofs_to_evaluate])**2).mean()).item()

    def eval_real_rec_comparison(self, model_label, sample_limit=None):
        results = (
            self.eval_real_rec(sample_limit, model_label, output_transform=Wiener()),
            self.eval_real_rec(sample_limit, model_label, output_transform=None),
            self.eval_real_rec(sample_limit, None, output_transform=Wiener()),
            self.eval_real_rec(sample_limit, None, output_transform=ZeroTransform())
        )
    
        return (
            f"({model_label} vs Wiener: {results[0]}, "
            f"{model_label} vs Original: {results[1]}, "
            f"Original vs Wiener: {results[2]}, "
            f"Original vs Empty: {results[3]})"
        )
        
        
    def plot_real_data(self, sample_id, data_path="datasets/210.hdf5", model_label_list=None, input_transform=None, add_to_label="", show_real_data=True, show_label=False, show_rmse=False, label_index=0, additional_transform_labels={"Wiener": Wiener()}):
        evaluated_images_list = []
        evaluated_plot_title_list = []
        if model_label_list is None:
            model_label_list = list(self.model_dict.keys())
        for model_label in model_label_list:
            evaluated_plot_title_list.append(model_label)
            real_images = TOFReconstructor.get_real_data(
                sample_id, sample_id + 1, data_path
            )
            padding = self.model_dict[model_label].padding
            circular_transform = CircularPadding(padding)
            if input_transform is not None:
                composed_transform = Compose([input_transform, circular_transform])
            else:
                composed_transform = circular_transform
            eval_func = lambda data: self.evaluate_model(data, model_label)
            if show_label:
                real_label_images, evaluated_label_real_data = TOFReconstructor.evaluate_real_data(
                    real_images.to(self.device), eval_func, circular_transform
                )
            real_images, evaluated_real_data = TOFReconstructor.evaluate_real_data(
                real_images.to(self.device), eval_func, composed_transform
            )
            real_image = real_images[0].cpu()
            eval_real_image = evaluated_real_data[0].cpu()
            if show_label:
                real_label_image = real_label_images[0].cpu()
            if padding != 0:
                real_image = real_image[:,padding:-padding]
                eval_real_image = eval_real_image[:,padding:-padding]
                if show_label:
                    real_label_image = real_label_image[:,padding:-padding]
            evaluated_images_list.append(eval_real_image)
        if add_to_label != "":
            add_to_label = "_" + add_to_label
        add_to_label = str(sample_id) + add_to_label

        if show_real_data:
            preset_list = [real_image]
            preset_label_list = ["Real data"]
        else:
            preset_list = []
            preset_label_list = []

        if show_label:
            if input_transform is not None:
                preset_list.append(real_label_image)
                preset_label_list.append("Label")
            for key, entry in additional_transform_labels.items():
                preset_list.append(entry(real_label_image))
                preset_label_list.append(key)
        Evaluator.plot_detector_image_comparison(
            preset_list+evaluated_images_list,
            preset_label_list+evaluated_plot_title_list,
            "_".join(["real_image", add_to_label]),
            self.output_dir,
            show_rmse=show_rmse,
            label_index=label_index,
        )
    def persist_var(self, save_var, filename):
        with open(os.path.join(self.output_dir, filename), 'wb') as file:
            pickle.dump(save_var, file)

    def measure_time(self, model_name, device="cpu"):
        print((subprocess.check_output("lscpu | grep 'Model name'", shell=True).strip()).decode())
        model = self.model_dict[model_name]

        data = torch.rand(1024, 60*(16+2*model.padding), device=model.device)
        repetitions=10
        t0 = benchmark.Timer(
            stmt='eval_model(model, data)',
            setup='from __main__ import eval_model',
            globals={'model': model.to(device), 'data': data.to(device)},
            num_threads=100,
            label=model_name,
            sub_label='1024 random data points')
        print(t0.timeit(repetitions))

    @staticmethod
    def warmup_gpu(device):
        a = torch.randn(3000, 3000, device=device)
        b = torch.randn(3000, 3000, device=device)
        torch.cuda.synchronize()
        _ = torch.mm(a, b)
        torch.cuda.synchronize()
    
    def eval_model_simulation(self, model_label, input_transform=[], limit=None):
        z_score = lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0
        min_max = lambda x: (x - x.min()) / (x.max() - x.min())
        ds = self.test_with_input_transform(Compose(self.initial_input_transforms+input_transform))
        model = self.model_dict[model_label]
        
        squared_error_list = []
        z_error_list = []
        time_list = []
        
        device = "cpu"
        
        if hasattr(model, "device"):
            if model.device.type == "cuda":
                self.warmup_gpu(model.device)
                device = "cuda"            
        
        for i in ds:
            if model_label == "Pacman":
                noisy = i[0].reshape(i[0].shape[0], -1, i[1].shape[-1])
            else:
                noisy = i[0].reshape(i[0].shape[0], -1)
            label = i[1]
            if limit is not None and len(time_list) >= limit:
                break
            for j in trange(noisy.shape[0], leave=False):
                start_time = time.time()
                with torch.no_grad():
                    output = model(noisy[j].unsqueeze(0).to(torch.device(device)))[0]
                if model_label != "Pacman":
                    output = output.cpu()

                output_z = z_score(output.clone())
                
                if model_label == "Pacman":
                    output = min_max(output)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                time_list.append(elapsed_time)
                squared_error = (output-label[j])**2
                squared_error_list.append(squared_error[0])
                squared_z_error = (output_z-z_score(label[j]))**2
                z_error_list.append(squared_z_error[0])
                if limit is not None and len(time_list) >= limit:
                    break
        
        squared_error_tensor = torch.stack(squared_error_list)
        z_error_tensor = torch.stack(z_error_list)
        time_tensor = torch.tensor(time_list)
        return squared_error_tensor, z_error_tensor, time_tensor

    @staticmethod
    def print_rmse_time_comparison(results_dict, sig_level=0.01):
        def format_value(val, best_val, multiply=1., digits=2):
            format_str = f"{{:.{digits}f}}"
            s = format_str.format(val * multiply)
            return f"\\mathbf{{{s}}}" if val == best_val else s


        def print_line(name, rmse_tensor, rmse_z_tensor, time_tensor, 
                       rmse_sig=False, rmse_z_sig=False, time_sig=False, 
                       best_rmse=None, best_rmse_z=None, best_time=None):
            rmse_mean = rmse_tensor.mean().item()
            rmse_std = rmse_tensor.std().item()
            rmse_z_mean = rmse_z_tensor.mean().item()
            rmse_z_std = rmse_z_tensor.std().item()
            time_mean = time_tensor.mean().item()
            time_std = time_tensor.std().item()

            rmse_str = format_value(rmse_mean, best_rmse, multiply=100)
            rmse_z_str = format_value(rmse_z_mean, best_rmse_z)
            time_str = format_value(time_mean, best_time, digits=4)

            # Add significance markers
            rmse_dagger = "^\\dagger" if rmse_sig else ""
            rmse_z_dagger = "^\\dagger" if rmse_z_sig else ""
            time_dagger = "^\\dagger" if time_sig else ""

            print(f"{name} & "
                  f"${rmse_str}\\pm{rmse_std:.2f}{rmse_dagger}$ & "
                  f"${rmse_z_str}\\pm{rmse_z_std:.2f}{rmse_z_dagger}$ & "
                  f"${time_str}\\pm{time_std:.4f}{time_dagger}$ \\\\")

        # Preprocess all data
        rmse_data = {}
        rmse_z_data = {}
        time_data = {}

        for name, (squared_errors, z_errors, times) in results_dict.items():
            rmse_per_run = torch.sqrt((squared_errors).mean(dim=(1, 2)))      # [500]
            rmse_z_per_run = torch.sqrt((z_errors).mean(dim=(1, 2)))  # [500]
            rmse_data[name] = rmse_per_run
            rmse_z_data[name] = rmse_z_per_run
            time_data[name] = times  # Already shape [500]

        # Determine best (minimum) RMSE, RMSE-Z, and time
        rmse_means = {k: v.mean().item() for k, v in rmse_data.items()}
        rmse_z_means = {k: v.mean().item() for k, v in rmse_z_data.items()}
        time_means = {k: v.mean().item() for k, v in time_data.items()}

        best_rmse = min(rmse_means.values())
        best_rmse_z = min(rmse_z_means.values())
        best_time = min(time_means.values())

        # Choose a reference model (e.g., first entry) to compare others against
        reference_name = list(results_dict.keys())[0]
        ref_rmse = rmse_data[reference_name]
        ref_rmse_z = rmse_z_data[reference_name]
        ref_time = time_data[reference_name]

        # Print LaTeX table lines
        for name in results_dict:
            rmse = rmse_data[name]
            rmse_z = rmse_z_data[name]
            time = time_data[name]

            # Paired significance tests
            rmse_p = ttest_rel(ref_rmse.cpu(), rmse.cpu()).pvalue
            rmse_z_p = ttest_rel(ref_rmse_z.cpu(), rmse_z.cpu()).pvalue
            time_p = ttest_rel(ref_time.cpu(), time.cpu()).pvalue

            rmse_sig = rmse_p <= sig_level
            rmse_z_sig = rmse_z_p <= sig_level
            time_sig = time_p <= sig_level

            print_line(name, rmse, rmse_z, time,
                       rmse_sig=rmse_sig, rmse_z_sig=rmse_z_sig, time_sig=time_sig,
                       best_rmse=best_rmse, best_rmse_z=best_rmse_z, best_time=best_time)

    @staticmethod
    def format_tof_rmse_table_with_significance(result_dict, sig_level=0.01):
        def format_val(mean, std, best, digits=2, sig=False):
            fmt = f"{{:.{digits}f}}"
            val_str = fmt.format(mean)
            val_str = f"\\mathbf{{{val_str}}}" if abs(mean - best) < 1e-6 else val_str
            dagger = "^\\dagger" if sig else "~"
            return f"${val_str}{dagger} \\pm {fmt.format(std)}$"

        # Collect means and stds for all methods
        tof_means = {}
        z_means = {}
        tof_stds = {}
        z_stds = {}

        for method in result_dict:
            tof = result_dict[method]["tof_loss"]
            z_tof = result_dict[method]["z_tof_loss"]

            tof_means[method] = tof.mean().sqrt().item() * 100
            z_means[method] = z_tof.mean().sqrt().item()
            tof_stds[method] = tof.std().sqrt().item() * 100
            z_stds[method] = z_tof.std().sqrt().item()

        # Identify best methods per column
        best_tof = min(tof_means.values())
        best_z = min(z_means.values())

        best_tof_method = min(tof_means, key=tof_means.get)
        best_z_method = min(z_means, key=z_means.get)

        # Extract reference tensors for significance tests
        ref_tof = result_dict[best_tof_method]["tof_loss"].sqrt()
        ref_z = result_dict[best_z_method]["z_tof_loss"].sqrt()

        # Start LaTeX table output
        print("\\begin{tabular}{lrr}")
        print("\\toprule")
        print("\\textbf{Method} & \\textbf{ToF RMSE [\\%]} & \\textbf{ToF RMSE\\textsubscript{z}} \\\\")
        print("\\midrule")

        for method in result_dict:
            tof = result_dict[method]["tof_loss"].sqrt()
            z_tof = result_dict[method]["z_tof_loss"].sqrt()

            # Truncate to match reference lengths to avoid shape issues
            N_tof = min(len(tof), len(ref_tof))
            N_z = min(len(z_tof), len(ref_z))

            # Determine significance for ToF RMSE
            if method != best_tof_method:
                tof_p = ttest_rel(ref_tof[:N_tof].cpu(), tof[:N_tof].cpu()).pvalue
                tof_sig = tof_p <= sig_level
            else:
                tof_sig = False

            # Determine significance for ToF RMSE_z
            if method != best_z_method:
                z_p = ttest_rel(ref_z[:N_z].cpu(), z_tof[:N_z].cpu()).pvalue
                z_sig = z_p <= sig_level

            else:
                z_sig = False

            # Format output strings with bolding and daggers
            tof_str = format_val(tof_means[method], tof_stds[method], best_tof, digits=2, sig=tof_sig)
            z_str = format_val(z_means[method], z_stds[method], best_z, digits=2, sig=z_sig)

            print(f"{method} & {tof_str} & {z_str} \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
    
    @staticmethod
    def plot_ellipt(outputs_dir):
        # Constants
        ELL_TILT = (90 - 22.5) / 180 * np.pi  # Tilt in radians

        # Generate phase values from 0 to 2π
        phase_vals = np.linspace(0, 2 * np.pi, 500)

        # Start plot
        fig, ax = plt.subplots(figsize=(9, 4.5))

        # Loop through different ellipticities
        for ELLIPT in [0.3, 0.73, 1.0]:
            # Define the ellipticity function
            def ef(phase):
                return (ELLIPT) ** 2 / (
                    (ELLIPT * np.cos(phase - ELL_TILT)) ** 2 + (np.sin(phase - ELL_TILT)) ** 2
                )
            
            # Evaluate the function
            ef_vals = ef(phase_vals)
            
            # Plot each curve
            ax.plot(phase_vals, ef_vals, label=f'$\\varepsilon = {ELLIPT}$')

        # Main axis labels
        ax.set_xlabel("Angle $\\phi$ [rad]", fontsize=16)
        ax.set_ylabel("ef($\\phi$)", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
        ax.legend(fontsize=14)

        # Add secondary x-axis (top)
        def radians_to_custom(x):
            return x / (2 * np.pi) * 80

        def custom_to_radians(x):
            return x / 80 * (2 * np.pi)

        secax = ax.secondary_xaxis('top', functions=(radians_to_custom, custom_to_radians))
        secax.set_xlabel("Time [step]", fontsize=16)
        secax.set_xticks(np.linspace(0, 80, 9))  # Optional: neat ticks
        secax.tick_params(axis='x', labelsize=14)

        # Final layout
        plt.tight_layout()
        os.makedirs(outputs_dir, exist_ok=True)
        plt.savefig(os.path.join(outputs_dir, 'ellipt.pdf'))
        plt.close()

def eval_model(model, data):
    with torch.no_grad():
        return model(data)

if __name__ == "__main__":
    job_id = None
    total_job_count = None
    rmse_tensor_file = 'rmse_tensor.pkl'
    rmse_tensor = None
    # first argument is test_case, second and third (if defined) are job_id and total_job_count to divide the matrix calculation
    if len(sys.argv) > 1:
        test_case = int(sys.argv[1])
        if len(sys.argv) > 2:
            job_id = int(sys.argv[2])
            total_job_count = int(sys.argv[3])
    else:
        test_case = 0

    if test_case == 0:
        model_dict = {"1TOF Model": "outputs/tof_reconstructor/g0ebnecw/checkpoints",
             "2TOF Model": "outputs/tof_reconstructor/j75cmjsq/checkpoints",
             "3TOF Model": "outputs/tof_reconstructor/d0ccdqnp/checkpoints",
             "General Model": "outputs/tof_reconstructor/hj69jsmh/checkpoints",
             "Spec Model": "outputs/tof_reconstructor/1qo21nap/checkpoints"}
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
        if job_id is None or job_id == 0:
            e.measure_time("General Model", device="cpu")
            e.measure_time("General Model", device="cuda")
            result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(model_dict.keys(), i) for i in range(1)}
            e.persist_var(result_dict, 'denoising.pkl')
            print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
            print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
        
            # 1.2 table RMSEs of specific models vs. General model vs. 'meaner'
            result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(model_dict.keys(), i) for i in range(1,4)}
            result_dict["2 neighbors"] = e.evaluate_neigbors(model_dict.keys(), 2, 2)
            result_dict["2 opposite"] = e.evaluate_opposite(model_dict.keys(), 2, 2)
            result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs(model_dict.keys(), [7,12])
            e.persist_var(result_dict, 'rec_comp.pkl')
            print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
            print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
        
            # 1.3 heatmap plot rmse 1 TOF missing
            rmse_tensor = e.one_missing_tof_rmse_tensor("General Model")
            e.persist_var(rmse_tensor, rmse_tensor_file)
            e.plot_rmse_tensor(rmse_tensor.cpu())
    
        # 1.4 heatmap plot rmse 2 TOFs missing
        mse_matrix = e.two_missing_tofs_rmse_matrix("General Model", job_id, total_job_count)
        if os.path.exists(os.path.join(e.output_dir, rmse_tensor_file)):
            with open(os.path.join(e.output_dir, rmse_tensor_file), 'rb') as file:
                rmse_tensor = pickle.load(file)

        if mse_matrix is not None and rmse_tensor is not None:
            e.persist_var(mse_matrix, 'rmse_matrix.pkl')
            e.plot_rmse_matrix(mse_matrix.cpu(), rmse_tensor.cpu())
    
    elif test_case == 1:
        # Appendix
        model_dict = {"$\\gamma=0.3$ CAE-64": "outputs/tof_reconstructor/c9qnv5d1/checkpoints/",
             "$\\gamma=0.7$ CAE-64": "outputs/tof_reconstructor/qhjst8f6/checkpoints/",
             "$p=0$ CAE-64": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
             "$p=1$ CAE-64": "outputs/tof_reconstructor/okht9r1i/checkpoints/",
             "$p=2$ CAE-64": "outputs/tof_reconstructor/748p94if/checkpoints/",
             "CCNN": "outputs/tof_reconstructor/8c8o7h9j/checkpoints/", 
             }
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(model_dict.keys(), i) for i in range(1,4)}
        result_dict["2 neighbors"] = e.evaluate_neigbors(model_dict.keys(), 2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(model_dict.keys(), 2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs(model_dict.keys(), [7,12])
        e.persist_var(result_dict, 'rec_comp_params.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))

    elif test_case == 2:
        # Architecture comparison
        model_dict = {
             "CAE-32": "outputs/tof_reconstructor/o6nqth09/checkpoints/",
             "CAE-64": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
             "CAE-128": "outputs/tof_reconstructor/gvd9sv1x/checkpoints/",
             "CAE-256": "outputs/tof_reconstructor/0ys8nmh7/checkpoints/",
             "CAE-512": "outputs/tof_reconstructor/o8tdxj44/checkpoints/",
             "CAE-1024": "outputs/tof_reconstructor/1fgqdg17/checkpoints/",
             "UNet-512": "outputs/tof_reconstructor/n6od82ls/checkpoints/",
             "Spec Model": "outputs/tof_reconstructor/1qo21nap/checkpoints",
             "2TOF Model": "outputs/tof_reconstructor/j75cmjsq/checkpoints",
             "4TOF": "outputs/tof_reconstructor/lxfy2zgs/checkpoints",
             "5TOF": "outputs/tof_reconstructor/5y9vu48g/checkpoints",
             "AdamW": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
             "Adam": "outputs/tof_reconstructor/7w5lfbqf/checkpoints/",
             }
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())

        # 1. spectrogram detector image
        e.plot_spectrogram_detector_image(3, 21)
        e.plot_spectrogram_detector_image(3, 21, hot_enabled=True)
        # simulated sample denoised+rec
        e.plot_reconstructing_tofs_comparison([7, 12], ["Spec Model"])
        
        # AdamW vs Adam
        e.plot_real_data(42, model_label_list=["AdamW", "Adam"], input_transform=DisableSpecificTOFs([4,5]), add_to_label="adamw", show_label=True, additional_transform_labels={})
        
        # 2. real sample
        # 2.1 real sample denoising
        keys = list(model_dict.keys())
        architecture_keys = keys[:7]
        spec_2_tof_keys = keys[7:9]
        bigger_tof_count_keys = keys[9:11]
        e.plot_real_data(42, model_label_list=architecture_keys, additional_transform_labels={"Wiener": Wiener()}, show_label=True)
        
        # 2.2 real sample disabled + denoising
        e.plot_real_data(
            42, model_label_list=architecture_keys, input_transform=DisableSpecificTOFs([7, 12]), add_to_label="disabled_2_tofs", additional_transform_labels={}, show_label=True)
        
        requested_keys = architecture_keys+["Neighboring Mean"]
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(requested_keys, i) for i in range(1,4)}
        result_dict["2 neighbors"] = e.evaluate_neigbors(requested_keys, 2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(requested_keys, 2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs(requested_keys, [7,12])
        e.persist_var(result_dict, 'rec_comp_architectures.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))

        requested_keys = ["CAE-64"] + bigger_tof_count_keys + ["Neighboring Mean"]
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(requested_keys, i) for i in range(4,6)}
        result_dict["1--4 random"] = e.evaluate_1_n_disabled_tofs(requested_keys, n=4)
        result_dict["1--5 random"] = e.evaluate_1_n_disabled_tofs(requested_keys, n=5)
        e.persist_var(result_dict, 'rec_comp_4_5.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    elif test_case == 3:
        Evaluator.plot_gasdet_electron_int(sample_count=None)
        Evaluator.plot_ellipt("outputs/")
        
    elif test_case == 4:
        model_dict = {"general": "outputs/tof_reconstructor/hj69jsmh/checkpoints/"}
        e: Evaluator = Evaluator(model_dict=model_dict, device = torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device(), load_max=1)
        model = e.model_dict['general']
        disabled_tofs_min = 1
        disabled_tofs_max = 3
        padding = 0
        batch_size = 1024
        
        disabled_tof_rmse_list = []
        disabled_tof_intensity_list = []
        for disabled_tof in trange(16):
        
            target_transform = Compose(
                [
                    Reshape(),
                    PerImageNormalize(),
                    #CircularPadding(padding),
                ]
            )
            
            input_transform = Compose(
                [
                    Reshape(),
                    HotPeaks(0.1, 1.0),
                    PerImageNormalize(),
                    GaussianNoise(0.1),
                    PerImageNormalize(),
                    #DisableRandomTOFs(disabled_tofs_min, disabled_tofs_max, 0.5),
                    DisableSpecificTOFs([disabled_tof]),
                    PerImageNormalize(),
                    #CircularPadding(padding),
                ]
            )
            
            phase_rmse_list = []
            phase_intensity_list = []
            for i in trange(80, leave=False):
                dataset = H5Dataset(
                    path_list=["datasets/sigmaxy_7_peaks_0_20_hot_15_phase_separated/N10000_peaks1_phase"+str(i)+"_seed42.h5"],
                    input_transform=input_transform,
                    target_transform=target_transform,
                    load_max=None,
                )
                workers = psutil.Process().cpu_affinity()
                num_workers = len(workers) if workers is not None else 0
                
                datamodule = DefaultDataModule(
                    dataset=dataset,
                    num_workers=num_workers,
                    on_gpu=torch.cuda.is_available(),
                    batch_size_train=batch_size,
                    batch_size_val=batch_size,
                    split=[1., 0.,0.]
                )
                datamodule.setup()
            
                rmse_list = []
                intensity_list = []
                
                for i in datamodule.train_dataloader():
                    with torch.no_grad():
                        diff = model(i[0].flatten(start_dim=1).to(model.device))[:, 0] - i[1].to(model.device)
                    rmse_list.append(torch.sqrt((diff**2).mean()))
                    intensity_list.append(torch.sqrt((i[1].to(model.device)**2)[...,disabled_tof].mean()))
                phase_rmse_list.append(torch.stack(rmse_list).mean())
                phase_intensity_list.append(torch.stack(intensity_list).mean())
            phase_rmse_list = torch.stack(phase_rmse_list)
            phase_intensity_list = torch.stack(phase_intensity_list)
            disabled_tof_rmse_list.append(phase_rmse_list)
            disabled_tof_intensity_list.append(phase_intensity_list)
        
        disabled_tof_rmse_tens = torch.stack(disabled_tof_rmse_list).T
        disabled_tof_intensity_tens = torch.stack(disabled_tof_intensity_list).T
        
        with open('outputs/disabled_tof_rmse_tens.pkl', 'wb') as handle:
            pickle.dump(disabled_tof_rmse_tens.cpu(), handle)
        with open('outputs/disabled_tof_intensity_tens.pkl', 'wb') as handle:
            pickle.dump(disabled_tof_intensity_tens.cpu(), handle)
        # Create a figure with 1 row and 2 columns for subplots
        big_font=24
        small_font=22
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot the RMSE tensor
        rmse_image = axes[0].imshow(disabled_tof_rmse_tens.cpu().detach().numpy() * 100, cmap="hot", interpolation='none', aspect='auto')
        axes[0].set_xlabel("TOF position [#]", fontsize=big_font)
        axes[0].set_ylabel("Time [steps]", fontsize=big_font)
        axes[0].set_title("RMSE", fontsize=big_font)
        axes[0].tick_params(axis='both', labelsize=small_font)
        axes[0].set_xticks(range(0, 16, 2), [str(i) for i in range(1, 17, 2)], fontsize=small_font)
        
        # Add colorbar for the RMSE plot
        cbar_rmse = plt.colorbar(rmse_image, ax=axes[0])
        cbar_rmse.ax.tick_params(labelsize=small_font)
        cbar_rmse.set_label("RMSE", fontsize=big_font)
        
        # Plot the Intensity tensor
        intensity_image = axes[1].imshow(disabled_tof_intensity_tens.cpu().detach().numpy(), cmap="hot", interpolation='none', aspect='auto')
        axes[1].set_xlabel("TOF position [#]", fontsize=big_font)
        axes[1].set_ylabel("Time [steps]", fontsize=big_font)
        axes[1].set_title("Intensity", fontsize=big_font)
        axes[1].tick_params(axis='both', labelsize=small_font)
        axes[1].set_xticks(range(0, 16, 2), [str(i) for i in range(1, 17, 2)], fontsize=small_font)
        
        # Add colorbar for the Intensity plot
        cbar_intensity = plt.colorbar(intensity_image, ax=axes[1])
        cbar_intensity.ax.tick_params(labelsize=small_font)
        cbar_intensity.set_label("Intensity [arb.u.]", fontsize=big_font)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(e.output_dir + 'phase_tof_rmse.png', dpi=300, bbox_inches="tight")
        plt.savefig(e.output_dir + 'phase_tof_rmse.pdf', dpi=300, bbox_inches="tight")
        
    elif test_case == 5:
        model_dict = {
            "CAE-64": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
            "CAE-512": "outputs/tof_reconstructor/o8tdxj44/checkpoints/",
        }
        e: Evaluator = Evaluator(model_dict=model_dict, device = torch.device('cpu'), dataset=None)
        print(e.eval_real_rec_comparison("CAE-64", None))
        print(e.eval_real_rec_comparison("CAE-512", None))
    elif test_case == 6:
        model_dict = {
            "CAE-64": "outputs/tof_reconstructor/okht9r1i/checkpoints/",
        }
        e: Evaluator = Evaluator(model_dict=model_dict, device = torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device(), load_max=None)
        dataloader = e.test_with_input_transform(None) 
        stack = []
        for i in tqdm(dataloader):
            stack.append(i[1].sum(dim=0).sum(dim=0))
        stack = torch.stack(stack).sum(dim=0)
        print(stack)
    elif test_case== 7:
        min_max = lambda x: (x-x.min())/(x.max()-x.min())
        e: Evaluator = Evaluator({"General Model": "outputs/tof_reconstructor/hj69jsmh/checkpoints",
                                  "Spec Model": "outputs/tof_reconstructor/1qo21nap/checkpoints",
                                  "2TOF Model": "outputs/tof_reconstructor/j75cmjsq/checkpoints",
                                 }, device=torch.device('cuda'), output_dir="outputs/", load_max=10000, pac_man=True)
        for i,its_override in enumerate([10, None, 100, 100]):
            hot_enabled=True if i==3 else False
            label_appendix="_hot" if hot_enabled else ""
            X, Y, out = Evaluator.pacman_spectrogram_simulation(e.model_dict["Pacman"], 3, 21, its_override=its_override, hot_enabled=hot_enabled)
            Evaluator.save_spectrogram_detector_image_plot(Evaluator.min_max(out[0][0]), Evaluator.min_max(out[1][0]), output_path=e.output_dir + "pacman_"+str(its_override)+"_steps"+label_appendix+".pdf", Z=out[2][0])
        for i in range(5):
            e.plot_real_data(42+i, model_label_list=["General Model", "Pacman"], input_transform=DisableSpecificTOFs([7,12]), add_to_label="pacman", show_label=True, additional_transform_labels={})
        for i in range(5):
            e.plot_reconstructing_tofs_comparison([7, 12], ["General Model", "Pacman"], sample_id=i, tof_rmse_list=[7,12])
            
        e.plot_real_data(
            42, model_label_list=["Neighboring Mean", "Pacman", "Spec Model"], input_transform=DisableSpecificTOFs([7, 12]), add_to_label="disabled_2_tofs_other", additional_transform_labels={"Wiener": Wiener()}, show_label=True)
        e.plot_reconstructing_tofs_comparison([7, 12], ["General Model", "Spec Model", "Pacman", "Neighboring Mean"], sample_id=0, tof_rmse_list=[7,12])
        results_dict = {}
        
        for model_label in ["General Model", "Pacman"]:
            results_dict[model_label] = e.eval_model_simulation(model_label, limit=1000)

        e.persist_var(results_dict, "pacman.pkl")
        Evaluator.print_rmse_time_comparison(results_dict)
    elif test_case == 8:
        results_dict = {}
        e: Evaluator = Evaluator({"General Model": "outputs/tof_reconstructor/hj69jsmh/checkpoints",
                                 }, device=torch.device('cpu'), output_dir="outputs/", load_max=10000, pac_man=True)
        for model_label in ["General Model", "Pacman"]:
            results_dict[model_label] = e.eval_model_simulation(model_label, limit=1000, input_transform=[DisableRandomTOFs(1, 3)])
        e.persist_var(results_dict, "pacman_rec.pkl")
        Evaluator.print_rmse_time_comparison(results_dict)
    elif test_case == 9:
        model_dict = {
            "CAE-32":      "outputs/tof_reconstructor/o6nqth09/checkpoints/",
            "CAE-64":      "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
            "CAE-128":     "outputs/tof_reconstructor/gvd9sv1x/checkpoints/",
            "CAE-256":     "outputs/tof_reconstructor/0ys8nmh7/checkpoints/",
            "CAE-512":     "outputs/tof_reconstructor/o8tdxj44/checkpoints/",
            "CAE-1024":    "outputs/tof_reconstructor/1fgqdg17/checkpoints/",
            "UNet-512":    "outputs/tof_reconstructor/n6od82ls/checkpoints/",
            "Spec Model":  "outputs/tof_reconstructor/1qo21nap/checkpoints/",
            "1TOF Model":  "outputs/tof_reconstructor/g0ebnecw/checkpoints/",
            "2TOF Model":  "outputs/tof_reconstructor/j75cmjsq/checkpoints/",
            "3TOF Model":  "outputs/tof_reconstructor/d0ccdqnp/checkpoints/",
        }
        e: Evaluator = Evaluator(model_dict, device=torch.device('cuda'), output_dir="outputs/", load_max=None, pac_man=True)
        input_transform = Compose(
                    e.initial_input_transforms
                    + [
                        DisableRandomTOFs(1,3),
                        PerImageNormalize(),
                    ]
                    )
        results_dict = {}
        for model_label in list(model_dict.keys()):
            torch.manual_seed(42)
            results_dict[model_label] = e.evaluate_extended_rmse(model_label, input_transform, pacman_limit=1000)
        e.persist_var(results_dict, "tof_rmse.pkl")
        Evaluator.format_tof_rmse_table_with_significance(results_dict)
    else:
        print("Test case not found")
