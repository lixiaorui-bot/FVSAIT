# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from torch.nn import init
from .networks import ConvBlock, GaussianConv2d, ResNetBlock


class ChannelAttention(nn.Module):
    def __init__(self, input_channel, ratio=4):  # (b,c,h,w)
        super(ChannelAttention, self).__init__()
        self.avg_out = nn.AdaptiveAvgPool2d(output_size=1)  # 等价于torch.mean(x, dim=(-1, -2))
        self.max_out = nn.AdaptiveMaxPool2d(output_size=1)

        self.fc1 = nn.Linear(input_channel, input_channel // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_channel // ratio, input_channel, bias=False)

    def forward(self, inputs):  # (b,c,h,w)
        avg_out = self.avg_out(inputs).squeeze(-1).squeeze(-1)  
        max_out = self.max_out(inputs).squeeze(-1).squeeze(-1)  
        out = torch.stack([avg_out, max_out], dim=1)  
        out = self.fc2(self.relu(self.fc1(out)))  
        out = torch.sum(out, dim=1)  
        #out = torch.sum(out*self.pre_computed_dct_weights, dim = [2,3]) 
        out = torch.sigmoid(out)  
        out = out.view(-1, out.shape[1], 1, 1)  
        out = torch.multiply(inputs, out)  
        return out


# 
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class Generator(torch.nn.Module):
    """Base class for generators"""

    def __init__(self):
        super(Generator, self).__init__()

    def _set_init_defaults(self, init_type, kwargs):
        # set defaults for different init_types
        defaults = {
            "normal_": {"mean": 0.0, "std": 0.2},
            "xavier_normal_": {"gain": 0.2},
            "xavier_uniform_": {"gain": 1.0},
            "kaiming_normal_": {"a": 0.0, "mode": "fan_in"},
            "orthogonal_": {"gain": 0.2},
        }
        for k, v in defaults.get(init_type, {}).items():
            kwargs.setdefault(k, v)
        return kwargs

    def init_weights(self, init_type="xavier_normal_", **kwargs):
        # get defaults based on init_type
        kwargs = self._set_init_defaults(init_type, kwargs)

        # define initializer function
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find("BatchNorm2d") != -1:
                if hasattr(m, "weight") and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, kwargs.get("std", 0.2))
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "weight") and (
                    classname.startswith("Conv") or classname.startswith("Linear")
            ):
                if init_type.lower() == "default":  # uses pytorch's default init method
                    m.reset_parameters()
                elif hasattr(init, init_type):
                    getattr(init, init_type)(m.weight.data, **kwargs)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, **kwargs)

    def build_net(self):
        pass

    def forward(self, x):
        """
        Pass input through network

        Args:
            x: torch.Tensor input to network
        """
        return self.net(x)


class ResNetEncoder(Generator):
    """
    ResNet encoder network

    Args:
        in_channels: number channels in input image
        n_channels: number of filters in first conv layer [default: 64]
        res_blocks: number of resnet blocks [default: 5]
        n_downsample: number of downsample (and upsample) layers [default: 2]
        normalization: normalization function to use after each conv layer
            [default: nn.InstanceNorm2d]
        activation: activation function to use after each normalization
            [default: nn.ReLU()]
        bias: True/False learn bias for each conv layer [default: True]
        gaussian_downsample: True/False use GaussianConv2d for downsampling,
            otherwise use `stride=2` [default: True]

    Note:
        This approach uses `nn.Conv2d` layers with `stride=2` for
        downsampling, respectively. Also, `normalization` should be the
        callable function (e.g., `nn.InstanceNorm2d` not `nn.InstanceNorm2d(64)`). The
        `activation` should be the module itself (e.g., `nn.ReLU()` not `nn.ReLU`).
    """

    def __init__(
            self,
            in_channels,
            n_channels=64,
            res_blocks=5,
            n_downsample=2,
            normalization=nn.InstanceNorm2d,
            activation=None,
            bias=True,
            gaussian_downsample=True,
    ):
        super(ResNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.res_blocks = res_blocks
        self.n_downsample = n_downsample
        self.normalization = normalization
        self.activation = nn.ReLU() if activation is None else activation
        self.bias = bias
        self.gaussian_downsample = gaussian_downsample

        self.net = self.build_net(
            self.in_channels,
            self.n_channels,
            self.res_blocks,
            self.n_downsample,
            self.normalization,
            self.activation,
            self.bias,
            self.gaussian_downsample,
        )
        self.init_weights()

    def build_net(
            self,
            in_channels,
            n_channels,
            res_blocks,
            n_downsample,
            normalization,
            activation,
            bias,
            gaussian_downsample,
    ):
        net = []

        # in conv
        net.append(
            ConvBlock(
                in_channels,
                n_channels,
                normalization=normalization,
                activation=activation,
                kernel_size=7,
                padding=3,
                pad_type="reflect",
                bias=bias,
            )
        )

        # downsample
        for i in range(n_downsample):
            mult = 2 ** i
            ##下边注释这里可以去除该注意力
            if i == n_downsample // 2:
                net.append(ChannelAttention(n_channels * mult, ratio=4))# 通道注意力机制在中间层引入
            ##上边注释这里可以去除该注意力
            net.append(
                ConvBlock(
                    n_channels * mult,
                    n_channels * mult * 2,
                    normalization=normalization,
                    activation=activation,
                    kernel_size=3,
                    stride=1 if gaussian_downsample else 2,
                    padding=1,
                    pad_type="reflect",
                    bias=bias,
                )
            )
            if gaussian_downsample:
                net.append(
                    GaussianConv2d(n_channels * mult * 2, kernel_size=3, stride=2)
                )

        # residual blocks
        mult = 2 ** n_downsample
        for _ in range(res_blocks):
            ##下边注释这里可以去除该注意力
            if _ == res_blocks - 1:
                net.append(Self_Attn(n_channels * mult, activation=activation))  # 根据大部分论文的经验性，在生成器的倒数第二层引入自注意力机制
            ##上边注释这里可以去除该注意力
            net.append(
                ResNetBlock(
                    n_channels * mult,
                    normalization=normalization,
                    activation=activation,
                    bias=bias,
                    kernel_size=3,
                    padding=1,
                    pad_type="reflect",
                )
            )

        self.out_channels = n_channels * mult
        return nn.Sequential(*net)


class ResNetDecoder(Generator):
    """
    ResNet Decoder network

    Args:
        in_channels: number channels in input
        out_channels: number of channels in output image
        n_channels: number of filters in first conv layer [default: 64]
        res_blocks: number of resnet blocks [default: 4]
        n_upsample: number of upsample layers [default: 2]
        normalization: normalization function to use after each conv layer (except last)
            [default: nn.InstanceNorm2d]
        activation: activation function to use after each normalization (except last)
            [default: nn.ReLU()]
        bias: True/False learn bias for each conv layer (except last) [default: True]
        gaussian_upsample: True/False use GaussianConv2d for upsampling, otherwise use
            `nn.Upsample(scale_factor=2)` [default: True]

    Note:
        This approach uses `nn.Upsample(scale_factor=2)` for upsampling. Also, `normalization`
        should be the callable function (e.g., `nn.InstanceNorm2d` not `nn.InstanceNorm2d(64)`).
        The `activation` should be the module itself (e.g., `nn.ReLU()` not `nn.ReLU`).
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_channels=64,
            res_blocks=4,
            n_upsample=2,
            normalization=nn.InstanceNorm2d,
            activation=None,
            bias=True,
            gaussian_upsample=True,
    ):
        super(ResNetDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        self.res_blocks = res_blocks
        self.n_upsample = n_upsample
        self.normalization = normalization
        self.activation = nn.ReLU() if activation is None else activation
        self.bias = bias
        self.gaussian_upsample = gaussian_upsample
        # ensure in_channels, n_channels, and n_upsample are appropriate
        assert (
                in_channels == n_channels * 2 ** n_upsample
        ), "`n_channels * 2 ** n_upsample` must equal `in_channels`"

        self.net = self.build_net(
            self.in_channels,
            self.out_channels,
            self.n_channels,
            self.res_blocks,
            self.n_upsample,
            self.normalization,
            self.activation,
            self.bias,
            self.gaussian_upsample,
        )
        self.init_weights()

    def build_net(
            self,
            in_channels,
            out_channels,
            n_channels,
            res_blocks,
            n_upsample,
            normalization,
            activation,
            bias,
            gaussian_upsample,
    ):
        net = []

        # residual blocks
        for _ in range(res_blocks):
            net.append(
                ResNetBlock(
                    in_channels,
                    normalization=normalization,
                    activation=activation,
                    bias=bias,
                    kernel_size=3,
                    padding=1,
                    pad_type="reflect",
                )
            )

        # upsample
        for i in range(n_upsample):
            mult = 2 ** (n_upsample - i)
            if gaussian_upsample:
                net.append(
                    GaussianConv2d(
                        n_channels * mult, kernel_size=4, stride=2, deconv=True
                    )
                )
            else:
                net.append(nn.Upsample(scale_factor=2))

            net.append(
                ConvBlock(
                    n_channels * mult,
                    n_channels * mult // 2,
                    normalization=normalization,
                    activation=activation,
                    kernel_size=3,
                    padding=1,
                    pad_type="reflect",
                    bias=bias,
                )
            )

        # out conv
        net.append(
            ConvBlock(
                n_channels,
                out_channels,
                normalization=None,
                activation=torch.nn.Tanh(),
                kernel_size=7,
                padding=3,
                pad_type="reflect",
            )
        )
        return nn.Sequential(*net)


class ResNetGenerator(Generator):
    """
    ResNet Generator network

    Args:
        in_channels: number channels in input image
        out_channels: number of channels in output image
        n_channels: number of filters in first conv layer [default: 64]
        res_blocks: number of resnet blocks [default: 9]
        n_downsample: number of downsample (and upsample) layers [default: 2]
        normalization: normalization function to use after each conv layer (except last)
            [default: nn.InstanceNorm2d]
        activation: activation function to use after each normalization (except last)
            [default: nn.ReLU()]
        bias: True/False learn bias for each conv layer (except last) [default: True]
        gaussian_downsample: True/False use GaussianConv2d for downsampling (and upsampling),
             otherwise use `stride=2` and `nn.Upsample(scale_factor=2)` [default: True]
        init_kwargs: kwargs passed to `init_weights` function

    Note:
        This approach uses `GaussianConv2d` layers with `stride=2` (see layers.py) for
        downsampling and upsampling. Also, `normalization` should be the
        callable function (e.g., `nn.InstanceNorm2d` not `nn.InstanceNorm2d(64)`). The
        `activation` should be the module itself (e.g., `nn.ReLU()` not `nn.ReLU`).
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_channels=64,
            res_blocks=9,
            n_downsample=2,
            normalization=nn.InstanceNorm2d,
            activation=None,
            bias=True,
            gaussian_downsample=True,
            init_kwargs=None,
    ):
        super(ResNetGenerator, self).__init__()

        # set encoder
        enc_res_blocks = res_blocks - res_blocks // 2
        self.encoder = ResNetEncoder(
            in_channels,
            n_channels,
            enc_res_blocks,
            n_downsample,
            normalization,
            nn.ReLU() if activation is None else activation,
            bias,
            gaussian_downsample=gaussian_downsample,
        )
        self.encode_channels = self.encoder.out_channels

        # set decoder
        dec_res_blocks = res_blocks - enc_res_blocks
        self.decoder = ResNetDecoder(
            self.encode_channels,
            out_channels,
            n_channels,
            dec_res_blocks,
            n_downsample,
            normalization,
            nn.ReLU() if activation is None else activation,
            bias,
            gaussian_upsample=gaussian_downsample,
        )
        self.init_weights(**(init_kwargs or {}))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x, **kwargs):
        return self.decoder(x)

    def forward(self, x, **kwargs):  # (b,c,h,w)
        x = x
        return self.decode(self.encode(x))
