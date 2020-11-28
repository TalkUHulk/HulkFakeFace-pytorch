from __future__ import print_function
import torch
from torchvision import transforms
import numpy as np
from tool import timer
from facebase import FaceBase
from PIL import Image
import cv2 as cv
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Beauty.models import *
from ArcFace.data import *


class FaceBeauty(FaceBase):
    def __init__(self, weight_path, cuda=False, *, backbone='resnet50', target_size=224):
        super().__init__(weight_path, cuda)
        if backbone == 'resnet18':
            self.model = resnet_face18()
        elif backbone == 'resnet34':
            self.model = resnet_face34()
        elif backbone == 'resnet50':
            self.model = se_resnet_face50(out_planes=5)

        self.trans = transforms.Compose([transforms.Resize(target_size),
                                         transforms.CenterCrop(target_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.initialize()


    @timer
    @torch.no_grad()
    def inference(self, faces):
        print("faces number:", len(faces))
        scores = []
        print(type(faces))
        if isinstance(faces, np.ndarray):
            pass
        elif isinstance(faces, list):
            if isinstance(faces[0], np.ndarray):
                faces = list(map(lambda x: Image.fromarray(cv.cvtColor(x, cv.COLOR_BGR2RGB)), faces))
            faces = list(map(self.trans, faces))
            faces = torch.stack(faces)
            if self._cuda:
                faces = faces.cuda()
            outputs = self.model(faces)
            logits = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            scores = np.sum(logits * [1, 2, 3, 4, 5], axis=-1).tolist()
            print(scores)
        else:
            pass
        return scores


