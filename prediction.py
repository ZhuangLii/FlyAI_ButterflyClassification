# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from libs.make_model import make_model
from libs.dataset import butterfly_dataset
from config import cfg
import torch
from PIL import Image
import torchvision.transforms as T
from path import MODEL_PATH
import cv2
from os.path import join
import sys
import os
from path import DataID
from pytorch_toolbelt.inference import tta

class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        cfg.OUTPUT_DIR = MODEL_PATH
        cfg.DATASETS.ROOT_DIR = os.path.join(sys.path[0], 'data', 'input', DataID)
        self.models = []
        config_files = ['./configs/efficientnetb7.yaml', './configs/resnest.yaml', './configs/se_resnext.yaml']
        for config_file in config_files:
            cfg.merge_from_file(config_file)
            model = make_model(cfg, 200)
            state_dict = torch.load(join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '.pth'))
            model.load_state_dict(state_dict)
            model.cuda()
            model.eval()
            self.models.append(model)
        dataset = butterfly_dataset('./data/input/ButterflyClassification/')
        self.id_name_dict = dataset.id_name_dict
        self.val_transforms = []
        for config_file in config_files:
            cfg.merge_from_file(config_file)
            transform = T.Compose([
                        T.Resize(cfg.INPUT.SIZE_TEST),
                        T.ToTensor(),
                        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
                    ])
            self.val_transforms.append(transform)

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":"image\/172691.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":"Loxura_atymnus"}
        '''
        outputs = []
        for submodel, transform in zip(self.models, self.val_transforms):
            img = Image.open(image_path).convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0)
            img = img.cuda()
            with torch.no_grad():
                output = tta.d4_image2label(submodel, img)
                outputs.append(output)
        final = torch.mean(torch.stack(outputs, 0), 0)
        _,pred = torch.max(final,1)
        pred_name = self.id_name_dict[pred.detach().cpu().item()]
        return {"label": pred_name}

if __name__ == "__main__":
    p = Prediction()
    p.load_model()
    from glob import glob
    from os.path import join
    from tqdm import tqdm
    with open('./data/input/ButterflyClassification/train.csv', 'r') as f:
        lines = f.readlines()
        correct = 0
        for line in tqdm(lines):
            if 'label' in line:
                continue
            img, name = line.rstrip().split(',')
            img = img.split('/')[1]
            img = join('./input/ButterflyClassification/image', img)
            pred_name = p.predict(img)
            if pred_name == name:
                correct += 1
        print('score: {}'.format(correct / len(lines)))
