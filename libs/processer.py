import os
import time
import logging
import sys
import torch
import os.path as osp
import numpy as np
from libs.make_dataloader import GridMask
from torch.autograd import Variable
from flyai.utils.log_helper import train_log

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, feat, y_a, y_b, lam):
    all_loss_a, id_loss_a, cen_loss_a = [lam * x for x in criterion(pred, feat, y_a)]
    all_loss_b, id_loss_b, cen_loss_b = [
        (1-lam) * x for x in criterion(pred, feat, y_b)]
    all_loss = all_loss_a + all_loss_b
    id_loss = id_loss_a + id_loss_b
    cen_loss = cen_loss_a + cen_loss_b
    return all_loss, id_loss, cen_loss

def train_epoch(cfg, model, loader, optimizer, optimizer_center, center_criterion, loss_fun, epoch, n_epochs, grid, writer, logger, print_freq=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_id = AverageMeter()
    losses_center = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()
    grid.set_prob(epoch, cfg.SOLVER.MAX_EPOCHS)
    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        if cfg.INPUT.GRID_PRO > 0:
            input = grid(input)
        # compute output
        if not cfg.INPUT.MIXUP:
            output, feat = model(input, target)
            all_loss, id_loss, cen_loss = loss_fun(output, feat, target)
        else:
            input, targets_a, targets_b, lam = mixup_data(input, target, 0.5, use_cuda=True)
            input, targets_a, targets_b = map(Variable, (input,targets_a, targets_b))
            output, feat = model(input, target)
            all_loss, id_loss, cen_loss = mixup_criterion(
                loss_fun, output, feat, targets_a, targets_b, lam)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()
                              ).float().sum().item() / batch_size, batch_size)
        losses.update(all_loss.item(), batch_size)
        losses_id.update(id_loss.item(), batch_size)
        if isinstance(cen_loss, int) or isinstance(cen_loss, float):
            losses_center.update(0, batch_size)
        else:
            losses_center.update(cen_loss.item(), batch_size)
        writer.add_scalar(
            'data/loss',  losses.avg, (epoch-1)*len(loader) + batch_idx)
        writer.add_scalar(
            'data/loss_id',  losses_id.avg, (epoch-1)*len(loader) + batch_idx)
        writer.add_scalar(
            'data/loss_center',  losses_center.avg, (epoch-1)*len(loader) + batch_idx)
        writer.add_scalar(
            'data/train_error',  error.avg, (epoch-1)*len(loader) + batch_idx)
        # compute gradient and do SGD step
        all_loss.backward()
        optimizer.step()
        if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            for param in center_criterion.parameters():
                param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
            optimizer_center.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        train_log(train_loss=losses.val, train_acc=error.val)
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            logger.info(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, writer, epoch, logger, print_freq=20):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output, feat = model(input, target)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()
                                  ).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                logger.info(res)
            writer.add_scalar(
                'data/val_loss', losses.avg, (epoch-1)*len(loader) + batch_idx)
    writer.add_scalar(
        'data/val_error', error.avg, epoch)
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train_val_fun(cfg, model, train_loader, valid_loader,
                  loss_func, center_criterion, scheduler, optimizer, optimizer_center, writer, logger, val):
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    grid = GridMask(30, 50, rotate=1, ratio=0.5,
                    mode=1, prob=cfg.INPUT.GRID_PRO)
    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Train model
    best_error = 1
    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        _, train_loss, train_error = train_epoch(
            cfg=cfg,
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            center_criterion=center_criterion,
            loss_fun=loss_func,
            epoch=epoch,
            n_epochs=cfg.SOLVER.MAX_EPOCHS,
            grid=grid,
            logger= logger,
            writer=writer
        )
        scheduler.step()
        if val:
            _, valid_loss, valid_error = test_epoch(
                model=model_wrapper,
                loader=valid_loader,
                writer=writer,
                epoch=epoch,
                logger=logger
            )

        # Determine if model is the best
        if val:
            if valid_loader:
                if valid_error < best_error:
                    best_error = valid_error
                    print('New best error: %.4f' % best_error)
                    torch.save(model.state_dict(), os.path.join(
                        cfg.OUTPUT_DIR, 'model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(
                    cfg.OUTPUT_DIR, 'model.pth'))
    if not val:
        torch.save(model.state_dict(), os.path.join(
                    cfg.OUTPUT_DIR, 'model.pth'))