import torch
from .ranger import Ranger
from torch.optim import SGD, Adam

def make_optimizer(cfg, model, center_criterion):
    use_gc = cfg.SOLVER.GRADCENTER
    gc_conv_only = False
    params = []
    momentum = 0.9
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key:
                lr = cfg.SOLVER.BASE_LR * 2
        if "gap" in key:
            lr = cfg.SOLVER.BASE_LR * 10
            weight_decay = 0
        params += [{"params": [value], "lr": lr,
                    "weight_decay": weight_decay, "momentum": momentum}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = SGD(params, momentum=momentum, nesterov=True)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Ranger':
        optimizer = Ranger(params, use_gc=use_gc, gc_conv_only=gc_conv_only)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'ADAM':
        optimizer = Adam(params, betas=(0.9, 0.999), eps=1e-8)
    else:
        raise ValueError('Unfited optimizer name')
    optimizer_center = torch.optim.SGD(
        center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center