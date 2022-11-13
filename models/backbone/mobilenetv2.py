from torch import nn
try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Tuple, Union

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _make_ntuple(x, n):
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size= 3,
        stride= 1,
        padding= None,
        groups = 1,
        norm_layer = nn.BatchNorm2d,
        activation_layer = nn.ReLU,
        dilation= 1,
        inplace = True,
        bias = None,
        conv_layer = nn.Conv2d,
    ):

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)

        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``nn.BatchNorm2d``
        activation_layer (Callable[..., nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        padding = None,
        groups = 1,
        norm_layer = nn.BatchNorm2d,
        activation_layer = nn.ReLU,
        dilation = 1,
        inplace = True,
        bias = None,
    ):

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            nn.Conv2d,
        )

class InvertedResidual(nn.Module):
    def __init__(
        self, inp, oup, stride, expand_ratio, norm_layer = None) :
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes= 1000,
        width_mult= 1.0,
        inverted_residual_setting= None,
        round_nearest= 8,
        block= None,
        norm_layer= None,
        dropout= 0.2,
    ):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features= [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
#         #padding = (kernel_size - 1) // 2
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU6(inplace=True)
#         )

# def fixed_padding(kernel_size, dilation):
#     kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
#     pad_total = kernel_size_effective - 1
#     pad_beg = pad_total // 2
#     pad_end = pad_total - pad_beg
#     return (pad_beg, pad_end, pad_beg, pad_end) 

# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, dilation, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]

#         hidden_dim = int(round(inp * expand_ratio))
#         self.use_res_connect = self.stride == 1 and inp == oup

#         layers = []
#         if expand_ratio != 1:
#             # pw
#             layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

#         layers.extend([
#             # dw
#             ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
#             # pw-linear
#             nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(oup),
#         ])
#         self.conv = nn.Sequential(*layers)

#         self.input_padding = fixed_padding( 3, dilation )
#         print(self.input_padding)

#     def forward(self, x):
#         x_pad = F.pad(x, self.input_padding)
#         if self.use_res_connect:
#             return x + self.conv(x_pad)
#         else:
#             return self.conv(x_pad)

# class MobileNetV2(nn.Module):
#     def __init__(self, num_classes=1000, output_stride=8, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
#         """
#         MobileNet V2 main class
#         Args:
#             num_classes (int): Number of classes
#             width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
#             inverted_residual_setting: Network structure
#             round_nearest (int): Round the number of channels in each layer to be a multiple of this number
#             Set to 1 to turn off rounding
#         """
#         super(MobileNetV2, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#         self.output_stride = output_stride
#         current_stride = 1
#         if inverted_residual_setting is None:
#             inverted_residual_setting = [
#                 # t, c, n, s
#                 [1, 16, 1, 1],
#                 [6, 24, 2, 2],
#                 [6, 32, 3, 2],
#                 [6, 64, 4, 2],
#                 [6, 96, 3, 1],
#                 [6, 160, 3, 2],
#                 [6, 320, 1, 1],
#             ]

#         # only check the first element, assuming user knows t,c,n,s are required
#         if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
#             raise ValueError("inverted_residual_setting should be non-empty "
#                              "or a 4-element list, got {}".format(inverted_residual_setting))

#         # building first layer
#         input_channel = _make_divisible(input_channel * width_mult, round_nearest)
#         self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
#         features = [ConvBNReLU(3, input_channel, stride=2)]
#         current_stride *= 2
#         dilation=1
#         previous_dilation = 1

#         # building inverted residual blocks
#         for t, c, n, s in inverted_residual_setting:
#             output_channel = _make_divisible(c * width_mult, round_nearest)
#             previous_dilation = dilation
#             if current_stride == output_stride:
#                 stride = 1
#                 dilation *= s
#             else:
#                 stride = s
#                 current_stride *= s
#             output_channel = int(c * width_mult)

#             for i in range(n):
#                 if i==0:
#                     features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
#                 else:
#                     features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*features)

#         # building classifier
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.last_channel, num_classes),
#         )

#         # weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.mean([2, 3])
#         x = self.classifier(x)
#         return x