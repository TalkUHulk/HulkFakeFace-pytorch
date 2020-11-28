import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import cv2 as cv
import math
import numpy as np
import random
from PIL import Image


random.seed(1215)
# 82758

class DataLoader(Dataset):
    def __init__(self, img_path, input_size=112, mean=None, std=None):
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        self.img_path = img_path
        self.imgs = []
        self._get_file_list_from_dir(self.img_path)
        self.transform = transforms.Compose([transforms.Resize(input_size),
                                             transforms.RandomCrop(input_size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std, inplace=True)])

    def __getitem__(self, index):
        img_path, gender = self.imgs[index]
        img = cv.imread(img_path)
        while img is None:
            print("Error open ", img_path)
            img_path, gender = self.imgs[random.randint(0, self.__len__()-1)]
            img = cv.imread(img_path)

        if len(img.shape) != 3:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        return self.transform(Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))), \
               torch.tensor(gender)

    def __len__(self):
        return len(self.imgs)

    def _get_file_list_from_dir(self, img_path):
        file_path = os.path.join(img_path, "AFAD-Full.txt")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img = line.rstrip('\n').lstrip('.')
                if os.path.isfile((self.img_path + img).replace('//', '/')):
                    self.imgs.append([(self.img_path + img).replace('//', '/'), 1 if img.split('/')[2] == '111' else 0])
                else:
                    print((self.img_path + img).replace('//', '/'))


train_dh = DataLoader("./tarball-master/AFAD-Full/")
train_loader = Data.DataLoader(
            train_dh, batch_size=2, shuffle=True)
print(len(train_loader))
#
# for img, label in train_loader:
#     print(img.shape, label)

