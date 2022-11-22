# We reuse the code of InterHand2.6M and adapt the network architecture to input heatmaps.
# Thanks to Gyeongsik Moon for excellent work.

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/InterHand2.6M/blob/main/LICENSE
#

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, input_size=3, addition_channel=42*64, features=True, early_features=False, return_inter=False
    ):
        self.inplanes = 64
        self.early_features = early_features
        self.return_inter = return_inter
        self.features = features
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, addition_planes=addition_channel)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1,addition_planes=0):
        downsample = None
        if stride != 1 or (self.inplanes+addition_planes) != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes+addition_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes+addition_planes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, heatmap):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.return_inter:
            intermediates = OrderedDict()
        x = self.relu(x)
        if self.return_inter:
            intermediates["res_conv1_relu"] = x#128

        x = self.maxpool(x)

        x = self.layer1(x)
        if self.return_inter:
            intermediates["res_layer1"] = x#64
        x = self.layer2(x)
        if self.return_inter:
            intermediates["res_layer2"] = x#32
        x = self.layer3(torch.cat([x,heatmap],dim=1))
        if self.return_inter:
            intermediates["res_layer3"] = x#16 for attention map
        x = self.layer4(x)
        if self.return_inter:
            intermediates["res_layer4"] = x#8 output feature
        if self.early_features:
            return x
        x = x.mean(3).mean(2)

        x = x.view(x.size(0), -1)
        if self.return_inter:
            return x, intermediates
        else:
            return x, {}

    def init_weights(self, name):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        cur_resnet = self.state_dict()
        target_resnet = {}
        for k in cur_resnet:
            if k not in org_resnet:
                if k.find("num_batches_tracked"):continue
                p = cur_resnet[k].clone()
                print("%s not in org_resnet"%k)
                nn.init.normal_(p,std=0.01)
                target_resnet[k] = p
            elif org_resnet[k].shape == cur_resnet[k].shape:
                target_resnet[k] = org_resnet[k]
            else:
                print("%s.shape is changed"%k, org_resnet[k].shape, cur_resnet[k].shape)
                p = cur_resnet[k].clone()
                nn.init.normal_(p,std=0.01)
                p[:,:org_resnet[k].shape[1]] = org_resnet[k]
                target_resnet[k] = p
        self.load_state_dict(target_resnet)
        print("Initialize resnet from model zoo")



def resnet18(input_size=3, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], input_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(input_size=3, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(input_size=3, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], input_size, **kwargs)
    if pretrained:
        org_resnet = torch.utils.model_zoo.load_url(model_urls["resnet50"])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        cur_resnet = model.state_dict()
        target_resnet = {}
        for k in cur_resnet:
            if torch.all(org_resnet[k].shape == cur_resnet[k].shape):
                target_resnet[k] = org_resnet[k]
            else:
                p = cur_resnet[k].copy()
                nn.init.normal_(p,std=0.01)
                p[:,:org_resnet[k].shape[1]] = org_resnet[k]
                target_resnet[k] = p
        model.load_state_dict(target_resnet)
    return model


def resnet101(input_size=3, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], input_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(input_size=3, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], input_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model

