import torch
import torch.nn as nn
import torch.nn.functional as F

class CConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding='valid'):
        super(CConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We'll handle custom padding separately
        )

    def forward(self, x):
        in_height, in_width = x.shape[2], x.shape[3]

        pad_along_height = max(self.kernel_size[0] - self.stride[0], 0) if in_height % self.stride[0] == 0 else max(self.kernel_size[0] - (in_height % self.stride[0]), 0)
        pad_along_width = max(self.kernel_size[1] - self.stride[1], 0) if in_width % self.stride[1] == 0 else max(self.kernel_size[1] - (in_width % self.stride[1]), 0)
        
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        if self.padding == 'same':
            # Reflective-style padding using `F.pad`
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='circular')
        elif self.padding == 'circ_width':
            # Circular padding along width
            x = F.pad(x, (0, 0, pad_top, pad_bottom), mode='constant')
            x = F.pad(x, (pad_left, pad_right, 0, 0), mode='circular')
        elif self.padding == 'circ_height':
            # Circular padding along height
            x = F.pad(x, (0, 0, pad_top, pad_bottom), mode='circular')
            x = F.pad(x, (pad_left, pad_right, 0, 0), mode='constant')
        elif self.padding != 'valid':
            raise ValueError(f"Padding '{self.padding}' is not supported.")
        
        return self.conv(x)
        
class CConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding='valid'):
        super(CConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_height = (kernel_size[0] - 1) // 2
        self.pad_width = (kernel_size[1] - 1) // 2
        if padding == "circ_width":
            self.padding_layer = nn.Sequential(nn.CircularPad2d((self.pad_width, self.pad_width, 0, 0)),
            nn.ZeroPad2d((0, 0, self.pad_height, self.pad_height)))
        elif padding == "circ_height":
            self.padding_layer = nn.Sequential(nn.CircularPad2d((0, 0, self.pad_height, self.pad_height)),
            nn.ZeroPad2d((self.pad_width, self.pad_width, 0, 0)))
        else:
            self.padding_layer = nn.CircularPad2d((self.pad_width, self.pad_width, self.pad_height, self.pad_height))
        # Initialize transposed convolution layer
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,  # We'll handle padding separately
            output_padding=0,
        )
        
    def forward(self, inp):
        x = self.padding_layer(inp)
        x= self.conv_transpose(x)
        crop_bottom = (x.shape[-2] - inp.shape[-2]) // 2
        crop_top = x.shape[-2] - inp.shape[-2] - crop_bottom

        crop_left = (x.shape[-1] - inp.shape[-1]) // 2
        crop_right = x.shape[-1] - inp.shape[-1] - crop_left
        return x[:, :, crop_left:-crop_right, crop_bottom:-crop_top]
