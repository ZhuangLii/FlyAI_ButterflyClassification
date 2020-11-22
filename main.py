# -*- coding: utf-8 -*-
import argparse
import os
import torch
import random
import sys
import numpy as np
import logging
import os.path as osp

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from path import DataID

from path import MODEL_PATH
from config import cfg
from libs.processer import train_val_fun
from libs.make_dataloader import make_dataloader
from libs.make_loss import make_loss
from libs.make_optimizer import make_optimizer
from libs.make_model import make_model
from torch.optim import lr_scheduler
from libs.warmup import WarmupMultiStepLR, GradualWarmupScheduler, WarmupCosineLR
from tensorboardX import SummaryWriter

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(
            save_dir, "train_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def __init__(self, args, logger):
        print('main func init')
        set_seed(cfg.SOLVER.SEED)
        cfg.OUTPUT_DIR = MODEL_PATH
        cfg.DATASETS.ROOT_DIR = os.path.join(sys.path[0], 'data', 'input', DataID)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
        self.cfg = cfg
        self.args = args
        self.logger = logger

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("ButterflyClassification")

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        config_files = ['./configs/'+args.CONFIG]
        for configs in config_files:
            self.cfg.merge_from_file(configs)
            cfg.merge_from_list(self.args.opts)
            logger.info("Running with config:\n{}".format(self.cfg))
            train_loader, valid_loader, num_class = make_dataloader(self.cfg)
            model = make_model(self.cfg, num_class)
            loss_func, center_criterion = make_loss(self.cfg, num_class)
            optimizer, optimizer_center = make_optimizer(
                cfg, model, center_criterion)
            if cfg.SOLVER.TYPE == 'warmup':
                scheduler = WarmupMultiStepLR(optimizer, self.cfg.SOLVER.STEPS, self.cfg.SOLVER.GAMMA,
                                    self.cfg.SOLVER.WARMUP_FACTOR,
                                    self.cfg.SOLVER.WARMUP_EPOCHS, self.cfg.SOLVER.WARMUP_METHOD)
            elif cfg.SOLVER.TYPE == 'warmup_exp':
                scheduler_exp = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
                scheduler = GradualWarmupScheduler(
                    optimizer, multiplier=10, total_epoch=self.cfg.SOLVER.WARMUP_EPOCHS, after_scheduler=scheduler_exp)
            elif cfg.SOLVER.TYPE == 'warmup_cos':
                scheduler = WarmupCosineLR(optimizer,
                                            max_iters=cfg.SOLVER.MAX_EPOCHS,
                                            warmup_factor= 0.001,
                                            warmup_iters=cfg.SOLVER.WARMUP_EPOCHS
                                        )
            if not os.path.exists(self.cfg.TBOUTPUT_DIR):
                os.mkdir(self.cfg.TBOUTPUT_DIR)
            writer = SummaryWriter(self.cfg.TBOUTPUT_DIR, filename_suffix=cfg.MODEL.NAME)
            self.logger.info('start training')
            train_val_fun(self.cfg, model, train_loader, valid_loader,loss_func, center_criterion, scheduler, optimizer, optimizer_center, writer, self.logger, val=cfg.IF_VAL) 
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Butterfly_CLS Training")
    parser.add_argument("-e", "--EPOCHS", default=20, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=2, type=int, help="batch size")
    parser.add_argument("-c", "--CONFIG", default='efficientb5.yaml', type=str, help="batch size")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    logger = setup_logger("butterfly", './log')
    # cfg.freeze()
    main = Main(args, logger)
    main.download_data()
    main.train()