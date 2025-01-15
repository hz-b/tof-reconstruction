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
from torchvision.transforms import Compose
from tqdm import trange, tqdm
from data_generation import Job
from scipy.stats import ttest_rel
import numpy as np
import pickle


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
        load_max=None
    ):
        self.device = device
        for key, value in model_dict.items():
            model_dict[key] = self.load_eval_model(Evaluator.load_first_ckpt_file(value))
            
        self.model_dict: dict = model_dict
        self.model_dict["Mean model"] = MeanModel(16, device=device)

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
            output_dict[key] = self.evaluate_rmse(model, input_transform)
        
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
        output_string += r"\hline" + "\n"

        model_keys = list(list(result_dict.values())[0].keys())

        for model_key in model_keys:
            model_row = [model_key]
            for scenario_value in result_dict.values():
                best_key = min(scenario_value, key=scenario_value.get)
                std_dev = scenario_value[model_key][1].std()
                if statistics_table:
                    model_row_element = f"{scenario_value[model_key][0]:.2e}".replace("e+0", "e+").replace("e-0", "e-")+f" $\\pm${std_dev:.2e}".replace("e+0", "e+").replace("e-0", "e-")
                else:
                    model_row_element = f"{scenario_value[model_key][0]:.2e}".replace("e+0", "e+").replace("e-0", "e-")
                if best_key == model_key:
                    model_row_element = r"\textbf{" + model_row_element + r"}"
                else:
                    if statistics_table:
                        p_value = Evaluator.significant_confidence_levels(scenario_value[best_key][1], scenario_value[model_key][1])[1]
                        model_row_element += f" ({p_value[0]:.2e}, {p_value[1]:.2e})".replace("e+0", "e+").replace("e-0", "e-")
                    if Evaluator.significant_confidence_levels(scenario_value[best_key][1], scenario_value[model_key][1])[0] and not statistics_table:
                        model_row_element += " $\\dagger$"
                model_row += [model_row_element]
            output_string += " & ".join(model_row) + r" \\" + "\n"
            if statistics_table:
                output_string += r"""\hline""" + "\n"
        if not statistics_table:
            output_string += r"""\hline""" + "\n"
        output_string += r"""\end{"""+table_environment+r"""}"""
        return output_string

    def test_with_input_transform(self, input_transform):
        workers = psutil.Process().cpu_affinity()
        num_workers = len(workers) if workers is not None else 0
        self.dataset.input_transform = input_transform
        datamodule = DefaultDataModule(dataset=self.dataset, batch_size_val=8192, num_workers=num_workers, on_gpu=(self.device.type=='cuda'))
        datamodule.setup()
        test_dataloader = datamodule.test_dataloader(max_len=100000)
        return test_dataloader

    def evaluate_missing_tofs(self, disabled_tofs, model):
        input_transform = Compose(
            self.initial_input_transforms
            + [
                DisableSpecificTOFs(disabled_tofs=disabled_tofs),
                PerImageNormalize(),
                CircularPadding(model.padding),
            ]
        )
        mean, _ = self.evaluate_rmse(model, input_transform)
        return mean

    def evaluate_rmse(self, model, input_transform):
        with torch.no_grad():
            test_dataloader = self.test_with_input_transform(input_transform)
            test_loss_list = []
            for x, y in tqdm(test_dataloader, leave=False):
                channels = x.shape[-2]
                tof_count = x.shape[-1] - 2 * model.padding
                x = x.flatten(start_dim=1)
                y = y.flatten(start_dim=1).to(model.device)
                y_hat = model(x.to(model.device))
                y_hat = y_hat.reshape(-1, channels, tof_count + 2*model.padding)
                if model.padding != 0:
                    y_hat = y_hat[:, :, model.padding:-model.padding]
                y_hat = y_hat.flatten(start_dim=1)
                test_loss = (torch.nn.functional.mse_loss(y_hat, y, reduction='none').mean(dim=-1))
                test_loss_list.append(test_loss)
            test_loss_tensor = torch.cat(test_loss_list)
            return test_loss_tensor.mean(), test_loss_tensor.flatten()

    def two_missing_tofs_rmse_matrix(self, model):
        with torch.no_grad():
            output_matrix = torch.full((16, 16), 0., device=model.device)
            evaluation_list = []
            for i in trange(output_matrix.shape[0]):
                for j in range(output_matrix.shape[1]):
                    if i >= j:
                        continue
                    else:
                        evaluation_list.append((i,j))
            for i, j in tqdm(evaluation_list):
                output_matrix[i][j] = self.evaluate_missing_tofs(
                        [i, j], model
                    )
            return output_matrix

    def one_missing_tof_rmse_tensor(self, model):
        with torch.no_grad():
            outputs = []
            for i in trange(16):
                outputs.append(
                    self.evaluate_missing_tofs([i], model)
                )
            return torch.tensor(outputs, device=self.device)

    def plot_rmse_matrix(self, matrix, diag):
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
        plt.savefig(self.output_dir + "2_tof_failed.png")

    def retrieve_spectrogram_detector(self, kick_min=0, kick_max=100, peaks=5, seed=42):
        output = Job([1, kick_min, kick_max, peaks, 0.73, (90 - 22.5) / 180 * np.pi, 30, seed, False, None])
        assert output is not None
        X, Y = output
        return X, Y

    def plot_spectrogram_detector_image(self, peaks=5, seed=42):
        X, Y = self.retrieve_spectrogram_detector(peaks=peaks, seed=seed)
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        fig, ax = plt.subplots(1,2)
        fontsize = 12
        ax[0].imshow(np.array(Y), aspect=Y.shape[1] / Y.shape[0], cmap='hot', interpolation="none", origin="lower",)
        ax[0].set_ylabel("Photon Energy [eV]", fontsize=fontsize)
        ax[0].set_title('Spectrogram', fontsize=fontsize)
        ax[0].set_xlabel('Time [steps]', fontsize=fontsize)
        ax[0].set_xticks(range(0, 100, 20), labels=range(0, 100, 20),fontsize=20)
        ax[0].tick_params(axis='both', labelsize=fontsize)
        ax[0].set_yticks(ticks=list(range(0, 61, 10)) + [60], labels=list(range(1150, 1220, 10)) + [1210])
        ax[0].spines[['right', 'top']].set_visible(False)
        out = ax[1].imshow(np.array(X), aspect=X.shape[1] / X.shape[0], cmap='hot', interpolation="none", origin="lower")
        ax[1].set_ylabel("Kinetic Energy [eV]",fontsize=fontsize)
        ax[1].set_xticks(range(0, 16, 5), [str(i) for i in range(1, 17, 5)],fontsize=20)
        ax[1].set_xlabel("TOF position [#]",fontsize=fontsize)
        ax[1].set_title('Detector image',fontsize=fontsize)
        ax[1].tick_params(labelsize=fontsize)
        ax[1].set_yticks(ticks=range(0, 70, 10), labels=range(280, 350, 10))
        ax[1].spines[['right', 'top']].set_visible(False)
        plt.tight_layout()
        out.set_clim(vmin=0, vmax=1)
        fig.colorbar(out, ax=ax, shrink=0.49, label='Intensity [arb.u.]')
        plt.savefig(self.output_dir + 'spectrogram_detector_image_'+str(peaks)+'_'+str(seed)+'.png', dpi=300, bbox_inches="tight")

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
        line1, = ax1.plot(x, rmse, color=color1, label='RMSE')  # Set zorder for line1
        ax1.set_xlabel('TOF position [#]', fontsize=20)
        ax1.set_ylabel('RMSE [arb.u.]', color=color1, fontsize=20)
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

        plt.savefig(self.output_dir + "1_tof_failed.png")


    @staticmethod
    def detector_image_ax(ax, data, title):
        ax.set_xlabel("TOF position [#]")
        ax.set_title(title)
        ax.set_xticks(range(0, 16, 5), [str(i) for i in range(1, 17, 5)])
        ax.set_yticks(ticks=list(range(0, 70, 10))+[60], labels=list(range(280, 350, 10))+ [340])
        return ax.imshow(
            data,
            aspect=data.shape[1] / data.shape[0],
            interpolation="none",
            cmap="hot",
            origin="lower",
        )

    @staticmethod
    def plot_detector_image_comparison(data_list, title_list, filename, output_dir):
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
            rows, columns, sharex=True, sharey=True, squeeze=False, figsize=(8, 3+rows*1)# (5+1, rows*(1.7+0.3)) #(8, 6)
        )
        for i in range(len(data_list)):
            cur_row = i // columns
            cur_col = i % columns
            if cur_col == 0:
                ax[cur_row, cur_col].set_ylabel("Kinetic Energy [eV]")
            ax[cur_row, cur_col].spines[['right', 'top']].set_visible(False)
            out = Evaluator.detector_image_ax(ax[cur_row, cur_col], data_list[i], title_list[i])
            ax[cur_row, cur_col].set_yticks(ticks=range(0, 70, 10), labels=range(280, 350, 10))
        for i in range(len(data_list), rows*columns):
            cur_row = i // columns
            cur_col = i % columns
            ax[cur_row, cur_col].set_visible(False)
        out.set_clim(vmin=0, vmax=1)
        plt.tight_layout()
        fig.colorbar(out, ax=ax, shrink=0.49, label='Intensity [arb.u.]')
        plt.savefig(output_dir + filename + ".png", dpi=300, bbox_inches="tight")

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
        self, disabled_tofs, model_label, batch_id=1, sample_id=0
    ):
        input_transform = Compose(
            self.initial_input_transforms
            + [
                DisableSpecificTOFs(disabled_tofs=disabled_tofs),
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
                    Evaluator.plot_detector_image_comparison(
                        [noisy_image, y[sample_id].cpu(), z[0].cpu()],
                        ["With noise", "Label", model_label],
                        "two_tofs_disabled",
                        self.output_dir,
                    )
                    break

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

    def eval_real_rec(self, sample_limit, model_label, input_transform=None, output_transform=None):
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
        return torch.sqrt(((real_images-evaluated_real_data)**2).mean()).item()

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
        
        
    def plot_real_data(self, sample_id, data_path="datasets/210.hdf5", model_label_list=None, input_transform=None, add_to_label="", show_label=False, additional_transform_labels={"Wiener": Wiener()}):
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

        preset_list = [real_image]
        preset_label_list = ["Real data"]
        if show_label:
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
        )
    def persist_var(self, save_var, filename):
        with open(os.path.join(self.output_dir, filename), 'wb') as file:
            pickle.dump(save_var, file)

    def measure_time(self, model_name):
        print((subprocess.check_output("lscpu | grep 'Model name'", shell=True).strip()).decode())
        model = self.model_dict[model_name]

        data = torch.rand(1024, 60*(16+2*model.padding), device=model.device)
        repetitions=10
        t0 = benchmark.Timer(
            stmt='eval_model(model, data)',
            setup='from __main__ import eval_model',
            globals={'model': model.to('cpu'), 'data': data.to('cpu')},
            num_threads=100,
            label=model_name,
            sub_label='1024 random data points')
        print(t0.timeit(repetitions))
       

def eval_model(model, data):
    with torch.no_grad():
        return model(data)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_case = int(sys.argv[1])
    else:
        test_case = 0

    if test_case == 0:
        model_dict = {"1TOF model": "outputs/tof_reconstructor/g0ebnecw/checkpoints",
             "2TOF model": "outputs/tof_reconstructor/j75cmjsq/checkpoints",
             "3TOF model": "outputs/tof_reconstructor/d0ccdqnp/checkpoints",
             "General model": "outputs/tof_reconstructor/hj69jsmh/checkpoints",
             "Spec model": "outputs/tof_reconstructor/1qo21nap/checkpoints"}
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
        e.measure_time("General model")
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(model_dict.keys(), i) for i in range(1)}
        e.persist_var(result_dict, 'denoising.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    
        # 1.2 table RMSEs of specific models vs. General model vs. 'meaner'
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(model_dict.keys(), i) for i in range(1,4)}
        result_dict["1--3 random"] = e.evaluate_1_n_disabled_tofs(model_dict.keys(), n=3)
        result_dict["2 neighbors"] = e.evaluate_neigbors(model_dict.keys(), 2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(model_dict.keys(), 2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs(model_dict.keys(), [7,12])
        e.persist_var(result_dict, 'rec_comp.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    
        # 1.3 heatmap plot rmse 1 TOF missing
        rmse_tensor = e.one_missing_tof_rmse_tensor(e.model_dict["General model"])
        e.persist_var(rmse_tensor, 'rmse_tensor.pkl')
        e.plot_rmse_tensor(rmse_tensor.cpu())
    
        # 1.4 heatmap plot rmse 2 TOFs missing
        mse_matrix = e.two_missing_tofs_rmse_matrix(e.model_dict["General model"])
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
        result_dict["1--3 random"] = e.evaluate_1_n_disabled_tofs(model_dict.keys(), n=3)
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
             "Spec model": "outputs/tof_reconstructor/1qo21nap/checkpoints",
             "2TOF model": "outputs/tof_reconstructor/j75cmjsq/checkpoints",
             "1-4TOF": "outputs/tof_reconstructor/lxfy2zgs/checkpoints",
             "1-5TOF": "outputs/tof_reconstructor/5y9vu48g/checkpoints",
             "AdamW": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
             "Adam": "outputs/tof_reconstructor/7w5lfbqf/checkpoints/",
             }
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())

        # 1. spectrogram detector image
        e.plot_spectrogram_detector_image(3, 57)
        # simulated sample denoised+rec
        e.plot_reconstructing_tofs_comparison([7, 12], "Spec model")
        
        # AdamW vs Adam
        e.plot_real_data(42, model_label_list=["AdamW", "Adam"], input_transform=DisableSpecificTOFs([4,5]), add_to_label="adamw", show_label=True, additional_transform_labels={})
        
        # 2. real sample
        # 2.1 real sample denoising
        keys = list(model_dict.keys())
        architecture_keys = keys[:5]
        spec_2_tof_keys = keys[5:7]
        bigger_tof_count_keys = keys[7:9]
        e.plot_real_data(42, model_label_list=architecture_keys, additional_transform_labels={})
        
        # 2.2 real sample disabled + denoising
        e.plot_real_data(
                    42, model_label_list=architecture_keys+spec_2_tof_keys+["Mean model"], input_transform=DisableSpecificTOFs([7, 12]), add_to_label="disabled_2_tofs", additional_transform_labels={})
        
        requested_keys = architecture_keys+["Mean model"]
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(requested_keys, i) for i in range(1,4)}
        result_dict["1--3 random"] = e.evaluate_1_n_disabled_tofs(requested_keys, n=3)
        result_dict["2 neighbors"] = e.evaluate_neigbors(requested_keys, 2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(requested_keys, 2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs(requested_keys, [7,12])
        e.persist_var(result_dict, 'rec_comp_architectures.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))

        requested_keys = ["CAE-64"] + bigger_tof_count_keys + ["Mean model"]
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(requested_keys, i) for i in range(4,6)}
        result_dict["1--4 random"] = e.evaluate_1_n_disabled_tofs(requested_keys, n=4)
        result_dict["1--5 random"] = e.evaluate_1_n_disabled_tofs(requested_keys, n=5)
        e.persist_var(result_dict, 'rec_comp_4_5.pkl')
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    elif test_case == 3:
        Evaluator.plot_gasdet_electron_int(sample_count=None)
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
        big_font=18
        small_font=14
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot the RMSE tensor
        rmse_image = axes[0].imshow(disabled_tof_rmse_tens.cpu().detach().numpy(), cmap="hot", interpolation='none', aspect='auto')
        axes[0].set_xlabel("TOF position [#]", fontsize=big_font)
        axes[0].set_ylabel("Time [steps]", fontsize=big_font)
        axes[0].set_title("RMSE", fontsize=big_font)
        axes[0].tick_params(axis='both', labelsize=small_font)
        axes[0].set_xticks(range(0, 16, 2), [str(i) for i in range(1, 17, 2)], fontsize=small_font)
        
        # Add colorbar for the RMSE plot
        cbar_rmse = plt.colorbar(rmse_image, ax=axes[0])
        cbar_rmse.ax.tick_params(labelsize=small_font)
        cbar_rmse.set_label("RMSE [arb.u.]", fontsize=big_font)
        
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
        plt.savefig(self.output_dir + 'phase_tof_rmse.png', dpi=300, bbox_inches="tight")
        plt.savefig(self.output_dir + 'phase_tof_rmse.pdf', dpi=300, bbox_inches="tight")
        
    elif test_case == 5:
        model_dict = {
            "CAE-64": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
            "CAE-512": "outputs/tof_reconstructor/o8tdxj44/checkpoints/",
        }
        e: Evaluator = Evaluator(model_dict=model_dict, device = torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device(), dataset=None)
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
    else:
        print("Test case not found")
