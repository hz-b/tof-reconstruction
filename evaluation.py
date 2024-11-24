import glob
import psutil
from scipy import stats
import os
import sys
import torch
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
    PerImageNormalize,
    Reshape,
    CircularPadding,
)
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from tqdm.auto import trange, tqdm
from data_generation import Job
from scipy.stats import ttest_ind
import numpy as np


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
    ):
        self.device = device
        for key, value in model_dict.items():
            model_dict[key] = self.load_eval_model(Evaluator.load_first_ckpt_file(value))
            
        self.model_dict: dict = model_dict
        self.model_dict["mean model"] = MeanModel(16, device=device)

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

        self.dataset = H5Dataset(
            path_list=list(glob.iglob("datasets/sigmaxy_7_peaks_0_20_hot_15/shuffled_*.h5")),
            input_transform=None,
            target_transform=target_transform,
            load_max=None,
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
    
    def evaluate_transform_normalized(self, transform):
        output_dict = {}
        for label, model in self.model_dict.items():
            input_transform = Compose(
                        self.initial_input_transforms
                        + [
                            transform,
                            PerImageNormalize(),
                            CircularPadding(model.padding),
                        ]
                    )
            output_dict[label] = self.evaluate_rmse(model, input_transform)
        
        return output_dict

    def evaluate_n_disabled_tofs(self, disabled_tof_count=3):
        return self.evaluate_transform_normalized(DisableRandomTOFs(disabled_tof_count, disabled_tof_count))

    def evaluate_1_3_disabled_tofs(self):
        return self.evaluate_transform_normalized(DisableRandomTOFs(1, 3, neighbor_probability=1))

    def evaluate_specific_disabled_tofs(self, disabled_list):
        return self.evaluate_transform_normalized(DisableSpecificTOFs(disabled_list))

    def evaluate_neigbors(self, min_disabled_count, max_disabled_count):
        return self.evaluate_transform_normalized(DisableNeighborTOFs(min_disabled_count, max_disabled_count))
    
    def evaluate_opposite(self, min_disabled_count, max_disabled_count):
        return self.evaluate_transform_normalized(DisableOppositeTOFs(min_disabled_count, max_disabled_count))
    
    @staticmethod
    def significant_confidence_levels(group_A, group_B, confidence=0.99):
        ci = ttest_ind(group_A.flatten().cpu(), group_B.flatten().cpu(), equal_var=False).confidence_interval(confidence_level=confidence)
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
        f = plt.figure(figsize=(19, 15), constrained_layout=True)
        plt.matshow((matrix+matrix.T+torch.diag(diag)).cpu(), fignum=plt.get_fignums()[-1], cmap="hot", aspect="auto")
        plt.xticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=20)
        plt.yticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=20)
        plt.grid(alpha=0.8)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
        cb.ax.set_ylabel('RMSE', fontsize=20)
        plt.xlabel("TOF position [#]", fontsize=20)
        plt.ylabel("TOF position [#]", fontsize=20)
        plt.savefig(self.output_dir + "2_tof_failed.png")

    def retrieve_spectrogram_detector(self, kick_min=0, kick_max=100, peaks=5, seed=42):
        output = Job([1, kick_min, kick_max, peaks, 0.73, (90 - 22.5) / 180 * np.pi, 30, seed, False])
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

    def plot_rmse_tensor(self, rmse_list, labels=None):
        f = plt.figure(figsize=(16, 4), constrained_layout=True)
        for i,entry in enumerate(rmse_list):
            label = labels[i] if labels is not None else None
            plt.plot(entry.cpu(), label=label)
        plt.legend()
        plt.xticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=20)
        plt.yscale('log')
        plt.yticks(fontsize=20)
        plt.grid(alpha=0.8)
        plt.xlabel("TOF position [#]", fontsize=20)
        plt.ylabel("RMSE", fontsize=20)
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
            rows = len(data_list) // 2 + len(data_list) % 2
            columns = 2
        else:
            rows = 1
            columns = len(data_list)
        fig, ax = plt.subplots(
            rows, columns, sharex=True, sharey=True, squeeze=False
        )
        for i in range(len(data_list)):
            cur_row = i // columns
            cur_col = i % columns
            if cur_col == 0:
                ax[cur_row, cur_col].set_ylabel("Kinetic Energy [eV]")
            ax[cur_row, cur_col].spines[['right', 'top']].set_visible(False)
            out = Evaluator.detector_image_ax(ax[cur_row, cur_col], data_list[i], title_list[i])
            ax[cur_row, cur_col].set_yticks(ticks=range(0, 70, 10), labels=range(280, 350, 10))
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
            for x, y in tqdm(test_dataloader):
                i += 1
                if i == batch_id:
                    z = (
                        self.evaluate_model(x[sample_id].unsqueeze(0).flatten(start_dim=1).to(self.device), model_label)[0]
                        .reshape(-1, 16 + 2*padding)
                        .unsqueeze(0)
                    )
                    if padding != 0:
                        z = z[:,:,padding:-padding]
                    Evaluator.plot_detector_image_comparison(
                        [x[sample_id].cpu(), y[sample_id].cpu(), z[0].cpu()],
                        ["With noise", "Label", "Reconstructed"],
                        "two_tofs_disabled",
                        self.output_dir,
                    )
                    break

    def evaluate_model(self, data, model_label):
        assert self.model_dict[model_label] is not None
        with torch.no_grad():
            return self.model_dict[model_label](data)

    def plot_real_data(self, sample_id, model_label_list, evaluated_plot_title_list, input_transform=None, add_to_label=""):
        evaluated_images_list = []
        for model_label in model_label_list:
            real_images = TOFReconstructor.get_real_data(
                sample_id, sample_id + 1, "datasets/210.hdf5"
            )
            padding = self.model_dict[model_label].padding
            circular_transform = CircularPadding(padding)
            if input_transform is not None:
                composed_transform = Compose([input_transform, circular_transform])
            else:
                composed_transform = circular_transform
            eval_func = lambda data: self.evaluate_model(data, model_label)
            real_images, evaluated_real_data = TOFReconstructor.evaluate_real_data(
                real_images.to(self.device), eval_func, composed_transform
            )
            real_image = real_images[0].cpu()
            eval_real_image = evaluated_real_data[0].cpu()
            if padding != 0:
                real_image = real_image[:,padding:-padding]
                eval_real_image = eval_real_image[:,padding:-padding]
            evaluated_images_list.append(eval_real_image)
        if add_to_label != "":
            add_to_label = "_" + add_to_label
        add_to_label = str(sample_id) + add_to_label
        Evaluator.plot_detector_image_comparison(
            [real_image]+evaluated_images_list,
            ["Real data"]+evaluated_plot_title_list,
            "_".join(["real_image", add_to_label]),
            self.output_dir,
        )

    def measure_time(self, model_name):
        model = self.model_dict[model_name]

        data = torch.rand(1024, 60*(16+2*model.padding), device=model.device)
        repetitions=10
        t0 = benchmark.Timer(
            stmt='eval_model(model, data)',
            setup='from __main__ import eval_model',
            globals={'model': model, 'data': data},
            num_threads=1,
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
        model_dict = {"1TOF model": "outputs/tof_reconstructor/szero50e/checkpoints",
             "2TOF model": "outputs/tof_reconstructor/szero50e/checkpoints",
             "3TOF model": "outputs/tof_reconstructor/szero50e/checkpoints",
             "general model": "outputs/tof_reconstructor/szero50e/checkpoints",
             "spec model": "outputs/tof_reconstructor/szero50e/checkpoints"}
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
        e.measure_time("general model") 
        e.plot_spectrogram_detector_image(3, 57)
        # 2. real sample
        # 2.1 real sample denoising
        e.plot_real_data(3, ["spec model"], evaluated_plot_title_list=["Denoised"])
        # 2.2 real sample disabled + denoising
        e.plot_real_data(
            3, ["spec model"], input_transform=DisableSpecificTOFs([3, 4]), add_to_label="disabled_2_tofs", evaluated_plot_title_list= ["Reconstructed"])
        # 1.1 graphic noisy+disabled vs. clear
        e.plot_missing_tofs_comparison([7, 12])
    
        # 1.1.2 plot
        e.plot_reconstructing_tofs_comparison([7, 12], "spec model")
    
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(i) for i in range(1)}
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    
        # 1.2 table RMSEs of specific models vs. general model vs. 'meaner'
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(i) for i in range(1,4)}
        result_dict["1--3 random"] = e.evaluate_1_3_disabled_tofs()
        result_dict["2 neighbors"] = e.evaluate_neigbors(2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs([7,12])
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    
        # 1.3 heatmap plot rmse 1 TOF missing
        rmse_tensor = e.one_missing_tof_rmse_tensor(e.model_dict["general model"])
        e.plot_rmse_tensor([rmse_tensor])
    
        # 1.4 heatmap plot rmse 2 TOFs missing
        mse_matrix = e.two_missing_tofs_rmse_matrix(e.model_dict["general model"])
        e.plot_rmse_matrix(mse_matrix, rmse_tensor)
    
    elif test_case == 1:
        # Appendix
        model_dict = {"$\\gamma=0.3$ general": "outputs/tof_reconstructor/s2s49jhj/checkpoints/",
             "$\\gamma=0.7$ general": "outputs/tof_reconstructor/41tg6fkf/checkpoints/",
             "padding=0 general": "outputs/tof_reconstructor/hj69jsmh/checkpoints/",
             "padding=2 general": "outputs/tof_reconstructor/748p94if/checkpoints/",
             "reference": "outputs/tof_reconstructor/okht9r1i/checkpoints/",
             #"1-4TOF": "outputs/tof_reconstructor/9ycv6lmg/checkpoints/",
             #"1-5TOF": "outputs/tof_reconstructor/9ycv6lmg/checkpoints/",      
             }
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(i) for i in range(1,4)}
        result_dict["1--3 random"] = e.evaluate_1_3_disabled_tofs()
        result_dict["2 neighbors"] = e.evaluate_neigbors(2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs([7,12])
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))

elif test_case == 2:
        # Architecture comparison
        model_dict = {
             "CAE-32": "outputs/tof_reconstructor/wzhx8vig/checkpoints/",
             "CAE-64": "outputs/tof_reconstructor/okht9r1i/checkpoints/",
             "CAE-128": "outputs/tof_reconstructor/9ycv6lmg/checkpoints/",
             "CAE-256": "outputs/tof_reconstructor/b1cl83sg/checkpoints/",
             "CAE-512": "outputs/tof_reconstructor/xxwm25nj/checkpoints/",
             "CAE-1024": "outputs/tof_reconstructor/67476x40/checkpoints/",   
             }
        e: Evaluator = Evaluator(model_dict, torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
        result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(i) for i in range(1,4)}
        result_dict["1--3 random"] = e.evaluate_1_3_disabled_tofs()
        result_dict["2 neighbors"] = e.evaluate_neigbors(2, 2)
        result_dict["2 opposite"] = e.evaluate_opposite(2, 2)
        result_dict["\\#8,\\#13 position"] = e.evaluate_specific_disabled_tofs([7,12])
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=False))
        print(Evaluator.result_dict_to_latex(result_dict, statistics_table=True))
    else:
        print("Test case not found")
