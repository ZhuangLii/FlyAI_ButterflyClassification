import torch.nn.functional as F
import logging
import torch
from libs.softmaxloss import CrossEntropyLabelSmooth
from libs.centerloss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = cfg.MODEL.FEAT_SIZE
    center_criterion = CenterLoss(
        num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(
            num_classes=num_classes, use_gpu=True)
    else:
        id_loss_func = F.cross_entropy

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'ce_center':
            id_loss = id_loss_func(score, target)
            cen_loss = center_criterion(feat, target)
            total_loss = cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * cen_loss
            return (total_loss, id_loss, cen_loss)
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'ce':
            id_loss = id_loss_func(score, target)
            return (id_loss, id_loss, 0)
        else:
            print('unexpected loss type')
    return loss_func, center_criterion