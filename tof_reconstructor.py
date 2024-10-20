import glob
import math
import psutil
from pathlib import Path
import torch
from torch import optim, nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from lightning.pytorch.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from torch.nn import Module
import wandb
from torchvision.transforms import Compose
from torchvision import models

from datamodule import DefaultDataModule
from dataset import H5Dataset
from transform import (
    CircularPadding,
    DisableRandomTOFs,
    DisableSpecificTOFs,

    GaussianNoise,
    HotPeaks,
    PerImageNormalize,
    PruneNegative,
    Reshape,
)
import h5py


class TOFReconstructor(L.LightningModule):
    def __init__(
        self,
        channels=60,
        layer_size: int = 4,
        blow=2.0,
        shrink_factor: str = "lin",
        learning_rate: float = 1e-4,
        optimizer: str = "adam_w",
        last_activation=nn.Sigmoid(),
        lr_scheduler: str | None = "plateau",
        outputs_dir="outputs/",
        cae=False,
        disabled_tofs_min=1,
        disabled_tofs_max=3,
        dropout_rate: float = 0.0,
        padding=0
    ):
        super(TOFReconstructor, self).__init__()
        self.save_hyperparameters(ignore=["last_activation"])
        self.channels = channels
        self.padding = padding
        self.tof_count = 16 + 2 * self.padding
        if cae:
            self.net = TOFReconstructor.create_cae()
        else:
            self.net = TOFReconstructor.create_sequential(
                self.channels * self.tof_count,
                100,
                layer_size,
                blow=blow,
                shrink_factor=shrink_factor,
                activation_function=nn.Mish(),
                last_activation=last_activation,
                mirror_for_autoencoder=True,
                dropout_rate=dropout_rate,
            )
        self.cae = cae
        self.validation_plot_len = 5
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.outputs_dir = outputs_dir
        self.optimizer = optimizer
        self.disabled_tofs_min = disabled_tofs_min
        self.disabled_tofs_max = disabled_tofs_max

        self.real_images = TOFReconstructor.get_real_data(
            108, 108 + 5, "datasets/210.hdf5"
        )

        Path(outputs_dir).mkdir(parents=True, exist_ok=True)
        self.register_buffer("validation_x_plot_data", torch.tensor([]))
        self.register_buffer("validation_y_plot_data", torch.tensor([]))
        self.register_buffer("validation_y_hat_plot_data", torch.tensor([]))
        self.register_buffer("train_x_plot_data", torch.tensor([]))
        self.register_buffer("train_y_plot_data", torch.tensor([]))
        self.register_buffer("train_y_hat_plot_data", torch.tensor([]))
        print(self.net)

    @staticmethod
    def create_sequential(
        input_length,
        output_length,
        layer_size,
        blow: float = 0.0,
        shrink_factor="log",
        activation_function: Module = nn.ReLU(),
        last_activation: Module | None = None,
        mirror_for_autoencoder: bool = False,
        dropout_rate: float = 0.0,
    ):
        layers = [input_length]
        blow_disabled = blow == 1.0 or blow == 0.0
        if not blow_disabled:
            layers.append(input_length * blow)

        if shrink_factor == "log":
            add_layers = torch.logspace(
                math.log(layers[-1], 10),
                math.log(output_length, 10),
                steps=layer_size + 2 - len(layers),
                base=10,
            ).long()
            # make sure the first and last element is correct, even though rounding
            if blow_disabled:
                add_layers[0] = input_length
            add_layers[-1] = output_length
        elif shrink_factor == "lin":
            add_layers = torch.linspace(
                layers[-1], output_length, steps=layer_size + 2 - len(layers)
            ).long()
        else:
            shrink_factor = float(shrink_factor)
            new_length = layer_size + 1 - len(layers)
            add_layers = (
                torch.ones(new_length)
                * layers[-1]
                * ((torch.ones(new_length) * shrink_factor) ** torch.arange(new_length))
            ).long()
            layers = torch.cat((torch.tensor([input_length]), add_layers))
            layers = torch.cat((layers, torch.tensor([output_length])))

        if not blow_disabled:
            layers = torch.tensor([layers[0]])
            layers = torch.cat((layers, add_layers))
        else:
            layers = add_layers
        if mirror_for_autoencoder:
            layers = torch.cat([layers, layers.flip(0)[1:]])
        nn_layers = []
        for i in range(len(layers) - 1):
            nn_layers.append(
                nn.Linear(int(layers[i].item()), int(layers[i + 1].item()))
            )
            if not i == len(layers) - 2:
                nn_layers.append(activation_function)
                if dropout_rate > 0.0:
                    nn_layers.append(nn.Dropout(p=dropout_rate))
            if i == len(layers) - 2 and last_activation is not None:
                nn_layers.append(last_activation)
        return nn.Sequential(*nn_layers)

    @staticmethod
    def create_cae(hidden_dims=(32, 64, 128, 256)):
        hidden_dims = list(hidden_dims)
        modules = []
        in_channels = 1
        for hdim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=hdim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.Mish(),
                    nn.BatchNorm2d(hdim),
                )
            )
            in_channels = hdim

        # Decoder
        hidden_dims.reverse()
        hidden_dims.append(1)
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.Mish(),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                )
            )
        #modules.append(nn.ConstantPad2d((0, 0, 0, -5), 0))

                # Final adjustment layer to ensure output of shape [batch_size, 1, 60, 20]
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    hidden_dims[-1],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Upsample(size=(60, 20), mode='bilinear', align_corners=False)  # Reshape to (60, 20)
            )
        )

        return nn.Sequential(*modules)

    class ResNetUNet(nn.Module):
        def __init__(self, n_class):
            super().__init__()

            self.base_model = models.resnet18(pretrained=True)
            self.base_layers = list(self.base_model.children())

            self.layer0 = nn.Sequential(
                *self.base_layers[:3]
            )  # size=(N, 64, x.H/2, x.W/2)
            self.layer0_1x1 = self.convrelu(64, 64, 1, 0)
            self.layer1 = nn.Sequential(
                *self.base_layers[3:5]
            )  # size=(N, 64, x.H/4, x.W/4)
            self.layer1_1x1 = self.convrelu(64, 64, 1, 0)
            self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
            self.layer2_1x1 = self.convrelu(128, 128, 1, 0)
            self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
            self.layer3_1x1 = self.convrelu(256, 256, 1, 0)
            self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
            self.layer4_1x1 = self.convrelu(512, 512, 1, 0)

            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )

            self.conv_up3 = self.convrelu(256 + 512, 512, 3, 1)
            self.conv_up2 = self.convrelu(128 + 512, 256, 3, 1)
            self.conv_up1 = self.convrelu(64 + 256, 256, 3, 1)
            self.conv_up0 = self.convrelu(64 + 256, 128, 3, 1)

            self.conv_original_size0 = self.convrelu(3, 64, 3, 1)
            self.conv_original_size1 = self.convrelu(64, 64, 3, 1)
            self.conv_original_size2 = self.convrelu(64 + 128, 64, 3, 1)

            self.conv_last = nn.Conv2d(64, n_class, 1)

        @staticmethod
        def convrelu(in_channels, out_channels, kernel, padding):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.ReLU(inplace=True),
            )

        def forward(self, input):
            x_original = self.conv_original_size0(input)
            x_original = self.conv_original_size1(x_original)

            layer0 = self.layer0(input)
            layer1 = self.layer1(layer0)
            layer2 = self.layer2(layer1)
            layer3 = self.layer3(layer2)
            layer4 = self.layer4(layer3)

            layer4 = self.layer4_1x1(layer4)
            x = self.upsample(layer4)
            layer3 = self.layer3_1x1(layer3)
            x = torch.cat([x, layer3], dim=1)
            x = self.conv_up3(x)

            x = self.upsample(x)
            layer2 = self.layer2_1x1(layer2)
            x = torch.cat([x, layer2], dim=1)
            x = self.conv_up2(x)

            x = self.upsample(x)
            layer1 = self.layer1_1x1(layer1)
            x = torch.cat([x, layer1], dim=1)
            x = self.conv_up1(x)

            x = self.upsample(x)
            layer0 = self.layer0_1x1(layer0)
            x = torch.cat([x, layer0], dim=1)
            x = self.conv_up0(x)

            x = self.upsample(x)
            x = torch.cat([x, x_original], dim=1)
            x = self.conv_original_size2(x)

            out = self.conv_last(x)

            return out

    def training_step(self, batch):
        x, y = batch
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        if self.cae:
            x = x.unflatten(1, (-1, self.tof_count)).unflatten(0, (-1, 1))
        y_hat = self.net(x)
        if self.cae:
            y_hat = y_hat.flatten(start_dim=1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        if self.train_y_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.train_y_plot_data.shape[0]
            self.train_x_plot_data = torch.cat([self.train_x_plot_data, x[:append_len]])
            self.train_y_plot_data = torch.cat([self.train_y_plot_data, y[:append_len]])
            self.train_y_hat_plot_data = torch.cat(
                [self.train_y_hat_plot_data, y_hat[:append_len]]
            )
        return loss

    def validation_step(self, batch):
        x, y = batch
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        if self.cae:
            x = x.unflatten(1, (-1, self.tof_count)).unflatten(0, (-1, 1))
        y_hat = self.net(x)
        if self.cae:
            y_hat = y_hat.flatten(start_dim=1)
        if self.validation_y_plot_data.shape[0] < self.validation_plot_len:
            append_len = self.validation_plot_len - self.validation_y_plot_data.shape[0]
            self.validation_x_plot_data = torch.cat(
                [self.validation_x_plot_data, x[:append_len]]
            )
            self.validation_y_plot_data = torch.cat(
                [self.validation_y_plot_data, y[:append_len]]
            )
            self.validation_y_hat_plot_data = torch.cat(
                [self.validation_y_hat_plot_data, y_hat[:append_len]]
            )
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch):
        return self.validation_step(batch)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def forward(self, x):
        if self.cae:
            x = x.unflatten(1, (-1, self.tof_count)).unflatten(0, (-1, 1))
        return self.net(x)

    @staticmethod
    def plot_data(tensor_list, label_list):
        fig, ax = plt.subplots(
            tensor_list[0].shape[0],
            len(tensor_list),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for i in range(tensor_list[0].shape[0]):
            for j in range(len(tensor_list)):
                ax[i, j].imshow(
                    tensor_list[j][i], aspect="auto", interpolation="none", cmap="hot"
                )
        for j, label in enumerate(label_list):
            ax[0, j].set_title(label)
        plt.tight_layout()
        return fig

    def create_plot(self, label: str, x, y_hat, y, labels=["input", "prediction", "label"]):
        if len(y) > 0:
            plt.clf()
            tensor_list = [
                tensor.reshape(-1, self.channels, self.tof_count).cpu().detach().numpy()
                for tensor in [x, y_hat, y]
            ]
            fig = TOFReconstructor.plot_data(
                tensor_list, labels
            )
            wandb.log({label: wandb.Image(fig)})
            plt.close(fig)

    @staticmethod
    def get_real_data(
        lower_idx,
        upper_idx,
        file_path="datasets/210.hdf5",
        energy_lower_bound_ev=280,
        energy_steps_ev=60,
    ):
        f = h5py.File(file_path, "r")
        acq_estimate_ev = f["acq_estimate_eV"]
        assert isinstance(acq_estimate_ev, h5py.Dataset)
        ev_scale = acq_estimate_ev[:]

        ev_scale = torch.Tensor(ev_scale)
        acq_mv_dataset = f["acq_mV"]
        assert isinstance(acq_mv_dataset, h5py.Dataset)
        acq_mv = torch.Tensor(acq_mv_dataset[lower_idx:upper_idx])

        ang_list = []
        empty_ang_list = []
        for ang in range(
            energy_lower_bound_ev, energy_lower_bound_ev + energy_steps_ev
        ):
            cur_ang_mask = ev_scale.round() == ang
            if cur_ang_mask.sum() == 0.0:
                empty_ang_list.append(ang - energy_lower_bound_ev)
            cur_tof_ang_sum_list = []
            for tof_nr in range(cur_ang_mask.shape[0]):
                cur_tof_ang_sum = acq_mv[:, tof_nr, cur_ang_mask[tof_nr]].sum(1)
                cur_tof_ang_sum_list.append(cur_tof_ang_sum)
            cur_ang = torch.stack(cur_tof_ang_sum_list, dim=-1)
            ang_list.append(cur_ang)
        output = torch.stack(ang_list, dim=1)
        output_copy = output.clone()
        for empty_ang in empty_ang_list:
            if empty_ang - 1 >= 0:
                if empty_ang - 2 in empty_ang_list:
                    shift_factor = 1.0 / 3.0
                else:
                    shift_factor = 1.0 / 2.0
                output_copy[:, empty_ang] += output[:, empty_ang - 1] * shift_factor
                output_copy[:, empty_ang - 1] *= shift_factor
            if empty_ang + 1 < output.shape[1]:
                if empty_ang + 2 in empty_ang_list:
                    shift_factor = 1.0 / 3.0
                else:
                    shift_factor = 1.0 / 2.0
                output_copy[:, empty_ang] += output[:, empty_ang + 1] * shift_factor
                output_copy[:, empty_ang + 1] *= shift_factor
        return output_copy

    @staticmethod
    def evaluate_real_data(real_images, evaluation_function, input_transform=None):
        real_image_transform = Compose(
            [
                PruneNegative(),
                PerImageNormalize(),
            ]
        )
        real_images = torch.stack(
            [real_image_transform(real_image) for real_image in real_images]
        )
        if input_transform is not None:
            real_images = torch.stack(
                [input_transform(real_image) for real_image in real_images]
            )

        evaluated_real_data = evaluation_function(real_images.flatten(start_dim=1))
        evaluated_real_data = evaluated_real_data.reshape(
            -1, real_images.shape[1], real_images.shape[2]
        )
        evaluated_data_transform = Compose([PruneNegative(), PerImageNormalize()])
        evaluated_real_data = torch.stack(
            [evaluated_data_transform(evaluated) for evaluated in evaluated_real_data]
        )

        return real_images, evaluated_real_data

    def on_validation_epoch_end(self):
        self.create_plot(
            "validation",
            self.validation_x_plot_data,
            self.validation_y_hat_plot_data,
            self.validation_y_plot_data,
        )
        self.create_plot(
            "train",
            self.train_x_plot_data,
            self.train_y_hat_plot_data,
            self.train_y_plot_data,
        )
        with torch.no_grad():
            self.real_images = self.real_images.to(self.device)
            real_images, evaluated_real_data = TOFReconstructor.evaluate_real_data(
                self.real_images, self.forward, CircularPadding(self.padding)
            )
            _, evaluated_real_data_2_tof = TOFReconstructor.evaluate_real_data(
                self.real_images, self.forward, Compose([DisableSpecificTOFs([7,15]), CircularPadding(self.padding)])
                )
            self.create_plot("real", real_images, evaluated_real_data, evaluated_real_data_2_tof, labels=["input", "prediction", "pred_-2_tof"])

        self.train_y_hat_plot_data = torch.tensor([]).to(self.train_y_hat_plot_data)
        self.train_y_plot_data = torch.tensor([]).to(self.train_y_plot_data)
        self.train_x_plot_data = torch.tensor([]).to(self.train_x_plot_data)
        self.validation_y_hat_plot_data = torch.tensor([]).to(
            self.validation_y_hat_plot_data
        )
        self.validation_y_plot_data = torch.tensor([]).to(self.validation_y_plot_data)
        self.validation_x_plot_data = torch.tensor([]).to(self.validation_x_plot_data)

    def configure_optimizers(self):
        if self.optimizer == "adam_w":
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_scheduler == "exp":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ExponentialLR(optimizer, gamma=0.895),
                    "frequency": 1,
                },
            }
        if self.lr_scheduler == "plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                    "monitor": "val_loss",
                    "frequency": 1,
                },
            }
        if self.lr_scheduler is not None:
            raise Exception("Defined LR scheduler not found.")

        return optimizer


if __name__ == "__main__":
    disabled_tofs_min = 1
    disabled_tofs_max = 3
    padding = 2

    target_transform = Compose(
        [
            Reshape(),
            PerImageNormalize(),
            CircularPadding(padding),
        ]
    )

    input_transform = Compose(
        [
            Reshape(),
            HotPeaks(0.1, 1.0),
            PerImageNormalize(),
            GaussianNoise(0.1),
            PerImageNormalize(),
            DisableRandomTOFs(disabled_tofs_min, disabled_tofs_max, 0.5),
            #DisableSpecificTOFs([3,11]),
            PerImageNormalize(),
            CircularPadding(padding),
        ]
    )
    h5_files = list(glob.iglob("datasets/sigmaxy_7_peaks_0_20_hot_15/shuffled_*.h5"))

    dataset = H5Dataset(
        path_list=h5_files,
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
    )
    datamodule.prepare_data()
    model = TOFReconstructor(
        disabled_tofs_min=disabled_tofs_min, disabled_tofs_max=disabled_tofs_max, padding=padding
    )
    #model = TOFReconstructor.load_from_checkpoint("outputs/tof_reconstructor/i2z5a29w/checkpoints/epoch=49-step=75000000.ckpt")
    wandb_logger = WandbLogger(
        name="ref2_60_20h15_7p_general", project="tof_reconstructor", save_dir=model.outputs_dir
    )
    datamodule.setup(stage="fit")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = L.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        log_every_n_steps=100000,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor],
    )
    trainer.init_module()

    trainer.fit(model=model, datamodule=datamodule)
