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
import torch.nn.functional as F

from ccnn import CConv2d, CConvTranspose2d

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
        last_activation=None,
        lr_scheduler: str | None = "plateau",
        outputs_dir="outputs/",
        architecture="mlp",
        disabled_tofs_min=1,
        disabled_tofs_max=3,
        dropout_rate: float = 0.0,
        padding=0,
        batch_size: int = 32,
        cae_hidden_dims=[32, 64, 128, 256, 512],
        padding_mode: str | None = None,
        normalize_minimum = None,
        normalize_maximum = None
    ):
        super(TOFReconstructor, self).__init__()
        self.save_hyperparameters(ignore=["last_activation"])
        self.channels = channels
        self.padding = padding
        self.normalize_minimum = normalize_minimum
        self.normalize_maximum = normalize_maximum
        self.tof_count = 16 + 2 * self.padding
        self.padding_mode = padding_mode
        self.cae_hidden_dims = cae_hidden_dims
        if architecture == "cae":
            self.net = TOFReconstructor.create_cae(dim_1_out=self.channels, dim_2_out=self.tof_count, hidden_dims=cae_hidden_dims, padding_mode=self.padding_mode, last_activation=last_activation)
        elif architecture == "unet":
            self.net = UNet2()
        elif architecture == "ccae":
            self.net = TOFReconstructor.create_cae(dim_1_out=self.channels, dim_2_out=self.tof_count, ccnn=True, hidden_dims=cae_hidden_dims, padding_mode=self.padding_mode, last_activation=last_activation)
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
        self.validation_plot_len = 5
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.outputs_dir = outputs_dir
        self.optimizer = optimizer
        self.disabled_tofs_min = disabled_tofs_min
        self.disabled_tofs_max = disabled_tofs_max
        self.architecture = architecture
        self.batch_size = batch_size

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
    def create_cae(dim_1_out, dim_2_out, hidden_dims=(32, 64, 128, 256, 512), ccnn=False, padding_mode=None, last_activation=None):
        if padding_mode is None:
            if ccnn:
                padding_mode = 'same'
            else:
                padding_mode = 'zeros'

        hidden_dims = list(hidden_dims)
        modules = []
        in_channels = 1
        for hdim in hidden_dims:
            if ccnn:    
                conv = CConv2d(
                    in_channels,
                    out_channels=hdim,
                    kernel_size=(3,3),
                    padding=padding_mode,
                )
            else:
                conv = nn.Conv2d(
                        in_channels,
                        out_channels=hdim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode=padding_mode,
                    )
            modules.append(
                nn.Sequential(
                    conv,
                    nn.Mish(),
                    nn.BatchNorm2d(hdim),
                )
            )
            in_channels = hdim

        # Decoder
        hidden_dims.reverse()
        hidden_dims.append(1)

        for i in range(len(hidden_dims) - 1):
            if ccnn:
                deconv = CConvTranspose2d(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i+1],
                    kernel_size=(3,3),
                    padding=padding_mode,
                )
            else:
                deconv = nn.ConvTranspose2d(
                            hidden_dims[i],
                            hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=1,
                        )
            dec_seq_list:list[nn.Module] = list([deconv])
            if i != len(hidden_dims) - 2 or not ccnn:
                dec_seq_list.append(nn.Mish())
                dec_seq_list.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            modules.append(
                nn.Sequential(
                    *dec_seq_list
                )
            )

                # Final adjustment layer to ensure output of shape [batch_size, 1, dim_1_out, dim_2_out]
        if ccnn:    
            conv = CConv2d(
                hidden_dims[-1],
                out_channels=1,
                kernel_size=(3,3),
                stride=(1,1),
                padding=padding_mode,
            )
        else:
            conv = nn.Conv2d(
                    hidden_dims[-1],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                )
        modules.append(
            nn.Sequential(
                conv,
                nn.Upsample(size=(dim_1_out, dim_2_out), mode='bilinear', align_corners=False) 
            )
        )
        if last_activation is not None:
            modules.append(last_activation)

        return nn.Sequential(*modules)
    @staticmethod
    def denormalize(a, minimum, maximum):
        return a * (maximum - minimum) + minimum

    def training_step(self, batch):
        x, y = batch
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        if self.architecture != 'mlp':
            x = x.unflatten(1, (-1, self.tof_count)).unflatten(0, (-1, 1))
        y_hat = self.net(x)
        if self.architecture != 'mlp':
            y_hat = y_hat.flatten(start_dim=1)
        if self.normalize_minimum is not None and self.normalize_maximum is not None:
            loss = nn.functional.mse_loss(self.denormalize(y_hat, self.normalize_minimum, self.normalize_maximum), self.denormalize(y, self.normalize_minimum, self.normalize_maximum))
        else:
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
        if self.architecture != 'mlp':
            x = x.unflatten(1, (-1, self.tof_count)).unflatten(0, (-1, 1))
        y_hat = self.net(x)
        if self.architecture != 'mlp':
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
        if self.normalize_minimum is not None and self.normalize_maximum is not None:
            val_loss = nn.functional.mse_loss(self.denormalize(y_hat, self.normalize_minimum, self.normalize_maximum), self.denormalize(y, self.normalize_minimum, self.normalize_maximum))
        else:
            val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch):
        return self.validation_step(batch)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def forward(self, x):
        if self.architecture != 'mlp':
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
    def evaluate_real_data(real_images, evaluation_function, input_transform=None, normalize_minimum=None, normalize_maximum=None):
        real_image_transform = Compose(
            [
                PerImageNormalize(normalize_minimum, normalize_maximum),
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
        evaluated_data_transform = Compose([PerImageNormalize(normalize_minimum, normalize_maximum)])
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
                self.real_images, self.forward, CircularPadding(self.padding), self.normalize_minimum, self.normalize_maximum
            )
            _, evaluated_real_data_2_tof = TOFReconstructor.evaluate_real_data(
                self.real_images, self.forward, Compose([DisableSpecificTOFs([7,15]), CircularPadding(self.padding)]), self.normalize_minimum, self.normalize_maximum
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

class UNet2(nn.Module):
    def __init__(self, encoder_channels = [1, 32, 64, 128, 256, 512]):
        super(UNet2, self).__init__()

        decoder_channels = encoder_channels[::-1]

        # Encoder
        self.encoders = nn.ModuleList([
            self.up(encoder_channels[i], encoder_channels[i+1])
            for i in range(len(encoder_channels) - 1)
        ])

        # Decoder
        decoders = [(0,1)]
        for i in range(len(decoder_channels) - 2):
            decoders.append((i, i+2))
        
        self.decoders = nn.ModuleList([
            self.down(decoder_channels[a], decoder_channels[b],
                      apply_activation=(i != len(decoders) - 1))
            for i, (a,b) in enumerate(decoders)
        ])

    def up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.BatchNorm2d(out_channels),
        )

    def down(self, in_channels, out_channels, apply_activation=True):
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        ]
        if apply_activation:
            layers += [nn.Mish(), nn.BatchNorm2d(out_channels)]
        return nn.Sequential(*layers)
        
    def prep_dec(self, input_dec, input_enc):
        target_size = input_enc.shape[2:]  # (H_dec, W_dec)
        input_dec_resized = F.interpolate(input_dec, size=target_size, mode='bilinear', align_corners=False)
        return input_dec_resized

    def forward(self, x):
        x0 = x
        enc_features = []

        # Encoder forward
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)

        # Decoder forward
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            if i < len(self.decoders) - 1:
                skip = enc_features[-(i + 2)]  # skip connections in reverse
                x = torch.cat((self.prep_dec(x, skip), skip), dim=1)
        x = self.prep_dec(x, x0)  # align final output with input
        return x

if __name__ == "__main__":
    disabled_tofs_min = 1
    disabled_tofs_max = 3
    padding = 0
    batch_size = 1024
    normalize_minimum = 0.0#0.1
    normalize_maximum = 1.0#0.9

    target_transform = Compose(
        [
            Reshape(),
            PerImageNormalize(normalize_minimum, normalize_maximum),
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
            PerImageNormalize(normalize_minimum, normalize_maximum),
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
        batch_size_train=batch_size,
        batch_size_val=batch_size
    )
    datamodule.prepare_data()
    model = TOFReconstructor(
        disabled_tofs_min=disabled_tofs_min, disabled_tofs_max=disabled_tofs_max, padding=padding, architecture='unet', batch_size=batch_size, cae_hidden_dims=[32, 64, 128, 256, 512, 64], padding_mode=None, last_activation=nn.Sigmoid(), normalize_minimum=normalize_minimum, normalize_maximum=normalize_maximum
    )
    #model = TOFReconstructor.load_from_checkpoint("outputs/tof_reconstructor/i2z5a29w/checkpoints/epoch=49-step=75000000.ckpt")
    wandb_logger = WandbLogger(
        name="ref3_general_unet", project="tof_reconstructor", save_dir=model.outputs_dir
    )
    datamodule.setup(stage="fit")

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = L.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        log_every_n_steps=1000,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor],
    )
    trainer.init_module()

    trainer.fit(model=model, datamodule=datamodule)
