import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
import random

random.seed(1215)


class ScutFBPDatasets(Dataset):
    def __init__(self, img_path, txt_path, ext=('.jpg',), fine_tune=False, test=False):
        self.extensions = ext
        self.preproc = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5),
                                    transforms.ColorJitter(contrast=0.5),
                                    transforms.ColorJitter(saturation=0.3)
                                    ])
            , transforms.RandomHorizontalFlip()
            , transforms.ToTensor()
            , transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        if test:
            self.preproc = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.img_path = img_path
        self.txt_path = txt_path
        self.imgs = []
        self._fine_tune = fine_tune
        self._test = test
        self._get_file_list_from_dir()

    def __getitem__(self, index):
        """
        label ->[0, 1, 2, 3, 4]
        :param index:
        :return:
        """
        img = Image.open(os.path.join(self.img_path, self.imgs[index][0]))
        label = float(self.imgs[index][1]) if self._fine_tune or self._test else int(float(self.imgs[index][1])) - 1
        return self.preproc(img), torch.tensor(label)

    def __len__(self):
        return len(self.imgs)

    def _get_file_list_from_dir(self):
        with open(self.txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.imgs.append(line.rstrip('\n').split(' '))


