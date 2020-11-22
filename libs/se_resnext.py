"""
SEResNet implementation from Cadene's pretrained models
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py
Additional credit to https://github.com/creafz

Original model: https://github.com/hujie-frank/SENet

ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from collections import OrderedDict
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.utils.model_zoo as model_zoo

__all__ = ['SENet', 'seresnext26_32x4d', 'seresnext50_32x4d', 'seresnext101_32x4d']

def load_pretrained(model, default_cfg, num_classes=1000, in_chans=3, model_path=None, filter_fn=None):
    state_dict = torch.load(model_path)
    if in_chans == 1:
        conv1_name = default_cfg['first_conv']
        print('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        assert False, "Invalid in_chans for pretrained weights"

    strict = True
    classifier_name = default_cfg['classifier']
    if num_classes == 1000 and default_cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != default_cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    model.load_state_dict(state_dict, strict=strict)
def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': [0, 0, 0], 'std': [1, 1, 1],
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }


default_cfgs = {
    'senet154':
        _cfg(url='http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth'),
    'seresnet18': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet18-4bb0ce65.pth',
        interpolation='bicubic'),
    'seresnet34': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnet34-a4004e63.pth'),
    'seresnet50': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet50-ce0d4300.pth'),
    'seresnet101': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet101-7e38fcc6.pth'),
    'seresnet152': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/se_resnet152-d17c99b7.pth'),
    'seresnext26_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/seresnext26_32x4d-65ebdb501.pth',
        interpolation='bicubic'),
    'seresnext50_32x4d':
        _cfg(url='http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'),
    'seresnext101_32x4d':
        _cfg(url='http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth'),
}


def _weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        # x = self.avg_pool(x)
        x = F.adaptive_avg_pool2d(x, 1)
        # x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(
            planes * 2, planes * 4, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(
            planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(
            inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck_disout(nn.Module):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4,
                 dist_prob=0.05,block_size=6,alpha=30,nr_steps=5e3):
        super(SEResNeXtBottleneck_disout, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(
            inplanes, width, kernel_size=1, bias=False, stride=1)
        self.disout1=LinearScheduler(Disout(dist_prob=dist_prob,block_size=block_size,alpha=alpha),
                                     start_value=0.,stop_value=dist_prob,nr_steps=nr_steps)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.disout2=LinearScheduler(Disout(dist_prob=dist_prob, block_size=block_size, alpha=alpha),
                                     start_value=0., stop_value=dist_prob, nr_steps=nr_steps)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.disout3=LinearScheduler(Disout(dist_prob=dist_prob, block_size=block_size, alpha=alpha),
                                     start_value=0., stop_value=dist_prob, nr_steps=nr_steps)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.disout1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.disout2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.disout3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out

class SEResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, drop_rate=0.2,
                 in_chans=3, inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, global_pool='avg'):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(
                    in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        # self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(3)
        self.drop_rate = drop_rate
        self.num_features = 512 * block.expansion
        self.last_linear = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(
            self.inplanes, planes, groups, reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.last_linear

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        del self.last_linear
        if num_classes:
            self.last_linear = nn.Linear(self.num_features, num_classes)
        else:
            self.last_linear = None

    def forward_features(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4

    def logits(self, x):
        x1 = self.avg_pool(x)
        if self.drop_rate > 0.:
            x1 = F.dropout(x1, p=self.drop_rate, training=self.training)
        x2 = x1.view(x1.size(0), -1)
        x3 = self.last_linear(x2)
        return x3

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.logits(x)
        return x


def seresnet18(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnet18']
    model = SENet(SEResNetBlock, [2, 2, 2, 2], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def seresnet34(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnet34']
    model = SENet(SEResNetBlock, [3, 4, 6, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def seresnet50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnet50']
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def seresnet101(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnet101']
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def seresnet152(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnet152']
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def senet154(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['senet154']
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def seresnext26_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnext26_32x4d']
    model = SENet(SEResNeXtBottleneck, [2, 2, 2, 2], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


def seresnext50_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnext50_32x4d']
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    import os
    if not os.path.exists('./se_resnext50_32x4d-a260b3a4.pth'):
        os.system('wget https://www.flyai.com/m/se_resnext50_32x4d-a260b3a4.pth')
        model_path = './se_resnext50_32x4d-a260b3a4.pth'
        load_pretrained(model, default_cfg, num_classes, in_chans, model_path)
    return model


def seresnext101_32x4d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    default_cfg = default_cfgs['seresnext101_32x4d']
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes, in_chans=in_chans, **kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(model, default_cfg, num_classes, in_chans)
    return model


class Disout(nn.Module):
    """
    Beyond Dropout: Feature Map Distortion to Regularize Deep Neural Networks
    https://arxiv.org/abs/2002.11022

    Args:
        dist_prob (float): probability of an element to be distorted.
        block_size (int): size of the block to be distorted.
        alpha: the intensity of distortion.
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    """

    def __init__(self, dist_prob, block_size=6, alpha=1.0):
        super(Disout, self).__init__()

        self.dist_prob = dist_prob
        self.weight_behind = None

        self.alpha = alpha
        self.block_size = block_size

    def forward(self, x):

        if not self.training:
            return x
        else:
            x = x.clone()
            if x.dim() == 4:
                width = x.size(2)
                height = x.size(3)

                seed_drop_rate = self.dist_prob * (width * height) / self.block_size ** 2 / (
                            (width - self.block_size + 1) * (height - self.block_size + 1))

                valid_block_center = torch.zeros(width, height, device=x.device).float()
                valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),
                int(self.block_size // 2):(height - (self.block_size - 1) // 2)] = 1.0

                valid_block_center = valid_block_center.unsqueeze(0).unsqueeze(0)

                randdist = torch.rand(x.shape, device=x.device)

                block_pattern = ((1 - valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()

                if self.block_size == width and self.block_size == height:
                    block_pattern = torch.min(block_pattern.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)[
                        0].unsqueeze(-1).unsqueeze(-1)
                else:
                    block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size),
                                                  stride=(1, 1), padding=self.block_size // 2)

                if self.block_size % 2 == 0:
                    block_pattern = block_pattern[:, :, :-1, :-1]
                percent_ones = block_pattern.sum() / float(block_pattern.numel())

                if not (self.weight_behind is None) and not (len(self.weight_behind) == 0):
                    wtsize = self.weight_behind.size(3)
                    weight_max = self.weight_behind.max(dim=0, keepdim=True)[0]
                    sig = torch.ones(weight_max.size(), device=weight_max.device)
                    sig[torch.rand(weight_max.size(), device=sig.device) < 0.5] = -1
                    weight_max = weight_max * sig
                    weight_mean = weight_max.mean(dim=(2, 3), keepdim=True)
                    if wtsize == 1:
                        weight_mean = 0.1 * weight_mean
                    # print(weight_mean)
                mean = torch.mean(x).clone().detach()
                var = torch.var(x).clone().detach()

                if not (self.weight_behind is None) and not (len(self.weight_behind) == 0):
                    dist = self.alpha * weight_mean * (var ** 0.5) * torch.randn(*x.shape, device=x.device)
                else:
                    dist = self.alpha * 0.01 * (var ** 0.5) * torch.randn(*x.shape, device=x.device)

            x = x * block_pattern
            dist = dist * (1 - block_pattern)
            x = x + dist
            x = x / percent_ones
            return x


class LinearScheduler(nn.Module):
    def __init__(self, disout, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.disout = disout
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.disout(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.disout.dist_prob = self.drop_values[self.i]
        self.i += 1


if __name__=="__main__":
    model = seresnet34()
    print(model)