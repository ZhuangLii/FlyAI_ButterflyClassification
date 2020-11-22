import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from os.path import join


class butterfly_dataset(Dataset):
    def __init__(self, img_root, transform=None, gary_ratio=0.0):
        with open('./label.txt', 'r') as f:
            lines = f.readlines()
        self.name_id_dict = dict()
        self.id_name_dict = dict()
        for i, line in enumerate(lines):
            self.name_id_dict[line.rstrip()] = i
            if i == 199:
                self.name_id_dict[line.rstrip()] = 197
            elif i == 198:
                self.name_id_dict[line.rstrip()] = 196
            self.id_name_dict[i] = line.rstrip()
        with open(join(img_root, 'train.csv'), 'r') as f:
            lines = f.readlines()
        self.img_pid = dict()
        label_set = set()

        for line in lines:
            if 'label' in line:
                continue
            img, name = line.rstrip().split(',')
            img = img.split('/')[1]
            self.img_pid[img] = self.name_id_dict[name]
            label_set.add(name)

        self.imgs = list(self.img_pid.keys())
        self.transform = transform
        self.img_root = img_root
        self.id_num = len(label_set)
        self.gary_ratio = gary_ratio

    def __getitem__(self, index):
        img_name = self.imgs[index]
        if random.uniform(0, 1) < self.gary_ratio:
            img = Image.open(join(self.img_root,'image', img_name)
                             ).convert('L').convert('RGB')
        else:
            img = Image.open(join(self.img_root, 'image', img_name)).convert('RGB')
        img = self.transform(img)
        target = self.img_pid[img_name]
        return img, target

    def __len__(self):
        return len(self.imgs)