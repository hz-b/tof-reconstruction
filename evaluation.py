import glob
import psutil
from scipy import stats
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
)
import matplotlib.pyplot as plt
from torchvision.transforms import Compose
from tqdm import trange, tqdm
from data_generation import Job
import numpy as np


torch.manual_seed(42)


class MeanModel(torch.nn.Module):
    def __init__(self, tof_count, device):
        super().__init__()
        self.tof_count = tof_count
        self.device = device

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
        device: torch.device = torch.get_default_device(),
        model_path: str = "outputs/tof_reconstructor/xxwm25nj/checkpoints/epoch=49-step=2343750.ckpt",
        one_tof_model_path: str = "outputs/tof_reconstructor/cmgsrmfi/checkpoints/epoch=49-step=62500000.ckpt",
        two_tof_model_path: str = "outputs/tof_reconstructor/13jcugos/checkpoints/epoch=49-step=62500000.ckpt",
        three_tof_model_path: str = "outputs/tof_reconstructor/adxzti5m/checkpoints/epoch=49-step=62500000.ckpt",
        special_model_path: str = "outputs/tof_reconstructor/wh8p2u9b/checkpoints/epoch=49-step=62500000.ckpt",
        output_dir: str = "outputs/",
    ):
        self.device = device
        self.model_dict: dict[str, torch.nn.Module] = {
            "1TOF model": self.load_eval_model(one_tof_model_path),
            "2TOF model": self.load_eval_model(two_tof_model_path),
            "3TOF model": self.load_eval_model(three_tof_model_path),
            "general model": self.load_eval_model(model_path),
            "spec model": self.load_eval_model(special_model_path),
        }
        self.model_dict["mean model"] = MeanModel(16, device=self.model_dict["general model"].device)

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
            path_list=list(glob.iglob("datasets/shuffled_*.h5")),
            input_transform=None,
            target_transform=target_transform,
            load_max=None,
        )

    def load_eval_model(self, model):
        model = TOFReconstructor.load_from_checkpoint(model, channels=60)
        model.eval()
        model.to(self.device)
        return model
    
    def evaluate_transform_normalized(self, transform):
        input_transform = Compose(
            self.initial_input_transforms
            + [
                transform,
                PerImageNormalize(),
            ]
        )
        return {
            label: self.evaluate_rmse(model, input_transform)
            for label, model in self.model_dict.items()
        }

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
    def significant_confidence_levels(group1, group2, confidence=0.95):
        mean1, mean2 = torch.mean(group1), torch.mean(group2)
        std1, std2 = torch.std(group1, unbiased=True), torch.std(group2, unbiased=True)
        n1, n2 = group1.size(0), group2.size(0)

        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)

        # Calculate CI for each group
        ci1 = (mean1 - z_score * (std1 / torch.sqrt(torch.tensor(n1, dtype=torch.float))),
               mean1 + z_score * (std1 / torch.sqrt(torch.tensor(n1, dtype=torch.float))))
        ci2 = (mean2 - z_score * (std2 / torch.sqrt(torch.tensor(n2, dtype=torch.float))),
               mean2 + z_score * (std2 / torch.sqrt(torch.tensor(n2, dtype=torch.float))))
        
        # Calculate CI for the difference in means
        diff_mean = mean2 - mean1
        std_diff = torch.sqrt((std1**2 / n1) + (std2**2 / n2))
        ci_diff = (diff_mean - z_score * std_diff, diff_mean + z_score * std_diff)
        return not (ci_diff[0] < 0. and ci_diff[1] > 0.)

    @staticmethod
    def result_dict_to_latex(result_dict):
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

        output_string = (
            r"""
        \begin{"""+table_environment+r"""}"""+text_width+r"""{p{2.5cm}|"""+
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
                model_row_element = f"{scenario_value[model_key][0]:.{4}f}"
                if best_key == model_key:
                    model_row_element = r"\mathbf{" + model_row_element + r"}"
                else:
                    if Evaluator.significant_confidence_levels(scenario_value[best_key][1], scenario_value[model_key][1]):
                        model_row_element += " \\dagger"
                model_row += ["$"+model_row_element+"$"]
            output_string += " & ".join(model_row) + r" \\" + "\n"

        output_string += r"""\hline
        \end{"""+table_environment+r"""}"""
        return output_string

    def test_with_input_transform(self, input_transform):
        workers = psutil.Process().cpu_affinity()
        num_workers = len(workers) if workers is not None else 0
        self.dataset.input_transform = input_transform
        datamodule = DefaultDataModule(dataset=self.dataset, batch_size_val=8192, num_workers=num_workers, on_gpu=(self.device.type=='cuda'))
        datamodule.setup()
        test_dataloader = datamodule.test_dataloader()
        return test_dataloader

    def evaluate_missing_tofs(self, missing_tofs, model):
        return self.evaluate_rmse(model, DisableSpecificTOFs(missing_tofs))

    def evaluate_rmse(self, model, input_transform):
        with torch.no_grad():
            test_dataloader = self.test_with_input_transform(input_transform)
            test_loss_list = []
            for x, y in tqdm(test_dataloader, leave=False):
                x = x.flatten(start_dim=1)
                y = y.flatten(start_dim=1).to(model.device)
                y_hat = model(x.to(model.device))
                test_loss = torch.sqrt(torch.nn.functional.mse_loss(y_hat, y, reduction='none').mean(dim=-1))
                test_loss_list.append(test_loss)
            test_loss_tensor = torch.stack(test_loss_list)
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
                output_matrix[i][j], _ = self.evaluate_missing_tofs(
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
        X, Y = Job([1, kick_min, kick_max, peaks, 0.73, (90 - 22.5) / 180 * np.pi, 30, seed, False])
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
        f = plt.figure(figsize=(16, 4), constrained_layout=True)
        plt.plot(rmse.cpu())
        plt.xticks(range(0, 16), [str(i) for i in range(1, 17)], fontsize=20)
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
        fig, ax = plt.subplots(
            1, len(data_list), sharex=True, sharey=True, squeeze=False
        )
        for i in range(len(data_list)):
            if i == 0:
                ax[0, i].set_ylabel("Kinetic Energy [eV]")
            ax[0, i].spines[['right', 'top']].set_visible(False)
            out = Evaluator.detector_image_ax(ax[0, i], data_list[i], title_list[i])
            ax[0, i].set_yticks(ticks=range(0, 70, 10), labels=range(280, 350, 10))
        out.set_clim(vmin=0, vmax=1)
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
        self, disabled_tofs, model_label, batch_id=1
    ):
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
                    z = (
                        self.evaluate_spec(x[0].unsqueeze(0).flatten(start_dim=1).to(self.device))[0]
                        .reshape(-1, 16)
                        .unsqueeze(0)
                    )
                    Evaluator.plot_detector_image_comparison(
                        [x[0].cpu(), y[0].cpu(), z[0].cpu()],
                        ["With noise", "Label", "Reconstructed"],
                        "two_tofs_disabled",
                        self.output_dir,
                    )
                    break

    def evaluate_spec(self, data):
        assert self.model_dict["spec model"] is not None
        with torch.no_grad():
            return self.model_dict["spec model"](data)

    def plot_real_data(self, sample_id, input_transform=None, add_to_label="", evaluated_plot_title=""):
        real_images = TOFReconstructor.get_real_data(
            sample_id, sample_id + 1, "datasets/210.hdf5"
        )
        real_images, evaluated_real_data = TOFReconstructor.evaluate_real_data(
            real_images.to(self.device), self.evaluate_spec, input_transform
        )
        if add_to_label != "":
            add_to_label = "_" + add_to_label
        add_to_label = str(sample_id) + add_to_label
        Evaluator.plot_detector_image_comparison(
            [real_images[0].cpu(), evaluated_real_data[0].cpu()],
            ["Real data", evaluated_plot_title],
            "_".join(["real_image", add_to_label]),
            self.output_dir,
        )

    def measure_time(self, model_name):
        model = self.model_dict[model_name]
        data = torch.rand(1024, 60*16, device=model.device)
        for label, model in self.model_dict.items():
            repetitions=10
            t0 = benchmark.Timer(
                stmt='eval_model(model, data)',
                setup='from __main__ import eval_model',
                globals={'model': model, 'data': data},
                num_threads=1,
                label=label,
                sub_label='1024 random data points')
            print(t0.timeit(repetitions))
       

def eval_model(model, data):
    with torch.no_grad():
        return model(data)

if __name__ == "__main__":
    e: Evaluator = Evaluator(torch.device('cuda') if torch.cuda.is_available() else torch.get_default_device())
    e.measure_time("general model") 
    e.plot_spectrogram_detector_image(3, 57)
    # 2. real sample
    # 2.1 real sample denoising
    e.plot_real_data(3, evaluated_plot_title="Denoised")
    # 2.2 real sample disabled + denoising
    e.plot_real_data(
        3, input_transform=DisableSpecificTOFs([7, 15]), add_to_label="disabled_2_tofs", evaluated_plot_title="Reconstructed"
    )
    # 1.1 graphic noisy+disabled vs. clear
    e.plot_missing_tofs_comparison([5, 13])

    # 1.1.2 plot
    e.plot_reconstructing_tofs_comparison([7, 15], "spec model")

    result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(i) for i in range(1)}
    print(Evaluator.result_dict_to_latex(result_dict))
          
    # 1.2 table RMSEs of specific models vs. general model vs. 'meaner'
    result_dict = {str(i)+" random": e.evaluate_n_disabled_tofs(i) for i in range(1,4)}
    result_dict["1--3 random"] = e.evaluate_1_3_disabled_tofs()
    result_dict["1--3 neighbors"] = e.evaluate_neigbors(1, 3)
    result_dict["1--3 opposite"] = e.evaluate_opposite(1, 3)
    result_dict["\\#8,\\#16 position"] = e.evaluate_specific_disabled_tofs([7,15])
    print(Evaluator.result_dict_to_latex(result_dict))

    # 1.3 heatmap plot rmse 1 TOF missing
    rmse_tensor = e.one_missing_tof_rmse_tensor(e.model_dict["general model"])
    e.plot_rmse_tensor(rmse_tensor)

    # 1.4 heatmap plot rmse 2 TOFs missing
    mse_matrix = e.two_missing_tofs_rmse_matrix(e.model_dict["general model"])
    e.plot_rmse_matrix(mse_matrix, rmse_tensor)
