import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import random
from PIL import Image

random.seed(1215)
input_shape = (112, 112)


class DataHander(Dataset):
    def __init__(self, data_path):
        self._data_path = data_path
        self._dir = []
        self.imgs = []
        self.trans = transforms.Compose([transforms.CenterCrop(input_shape),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

    def __getitem__(self, index):
        path = self.imgs[index]
        image = Image.open(path)
        data = self.trans(image)
        label = torch.tensor(int(path.split('/')[-2]))
        return data, label

    def __len__(self):
        return len(self.imgs)

    def __num__(self):
        return len(self._dir)

    def _get_file_list_from_dir(self):
        self._dir = [x.path for x in os.scandir(self._data_path)]
        for dir in self._dir:
            self.imgs += [x.path for x in os.scandir(dir) if x.name.endswith('jpg')]

import torch.utils.data as Data

train_dh = DataHander("../faces_glintasia/")
train_loader = Data.DataLoader(
            train_dh,
            batch_size=self.config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True
        )
