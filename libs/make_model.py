import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.parameter import Parameter
from libs.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from libs.se_resnet_ibn_a import se_resnet101_ibn_a
from libs.efficientnet import *
from libs.metric_learning import *
from libs.resnest import resnest50,resnest101,resnest200, resnest269
from libs.se_resnext import seresnext50_32x4d

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                                     (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
            '(' + 'p=' + '{:.4f}'.format(p) +\
            ', ' + 'eps=' + str(self.eps) + ')'

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        self.model_name = model_name
        self.cfg = cfg
        if model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(
                last_stride, pretrained=True, batchcrop=cfg.MODEL.BATCHDROP)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.base = resnet101_ibn_a(
                last_stride, frozen_stages=cfg.MODEL.FROZEN, batchcrop=cfg.MODEL.BATCHDROP)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.base = se_resnet101_ibn_a(last_stride)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'efficientnet_b0': # 1280
            self.base = EfficientNet.from_pretrained('efficientnet-b0')
        elif model_name == 'efficientnet_b1':
            self.base = EfficientNet.from_pretrained('efficientnet-b1')
        elif model_name == 'efficientnet_b2':
            self.base = EfficientNet.from_pretrained('efficientnet-b2')
        elif model_name == 'efficientnet_b3':
            self.base = EfficientNet.from_pretrained('efficientnet-b3')
        elif model_name == 'efficientnet_b4':
            self.base = EfficientNet.from_pretrained('efficientnet-b4')
        elif model_name == 'efficientnet_b5': # 456
            self.base = EfficientNet.from_pretrained('efficientnet-b5')
        elif model_name == 'efficientnet_b6':
            self.base = EfficientNet.from_pretrained('efficientnet-b6')
        elif model_name == 'efficientnet_b7': # 600
            self.base = EfficientNet.from_pretrained('efficientnet-b7')
        elif model_name == 'resnest50':
            self.base = resnest50()
        elif model_name == 'resnest101':
            self.base = resnest101()
        elif model_name == 'resnest200': # 320
            self.base = resnest200()
        elif model_name == 'resnest269': #460
            self.base = resnest269()
        elif model_name == 'seresnext50':
            self.base = seresnext50_32x4d() #224
        else:
            print('unsupported backbone! but got {}'.format(model_name))
        # self.base.apply(weights_init_kaiming)
        self.in_planes = cfg.MODEL.FEAT_SIZE
        # if 'resnet' in self.model_name:
        #     self.base.load_param(model_path)
        #     print(
        #         'Loading pretrained ImageNet model......from {}'.format(model_path))
        # elif 'efficient' in self.model_name:
        #     pass

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,
                                                        cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'curricularface':
            print('using {} loss with s:{}, m: {}'.format(
                self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CurricularFace(
                self.in_planes, self.num_classes, s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)

        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,
                                                        cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,
                                                        cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,
                                                        cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                            s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(
                self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def DualPath_fun(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.cfg.MODEL.FEAT_SIZE, x.size(2) ** 2)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) /
            28 ** 2).view(batch_size, -1)
        x = torch.nn.functional.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10))
        return x

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = self.gap(x)
        global_feat = global_feat.view(
            global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = global_feat
        if self.cfg.MODEL.DAULPATH:
            feat = self.DualPath_fun(feat)
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle', 'curricularface'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, feat
        else:
            return self.classifier(feat)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model