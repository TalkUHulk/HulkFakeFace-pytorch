from __future__ import print_function
import torch
from torchvision import transforms
import numpy as np
from tool import timer
from facebase import FaceBase
from PIL import Image
import cv2 as cv

from ArcFace.models import *


class FaceFeatures(FaceBase):
    def __init__(self, weight_path, cuda=False, *, backbone='resnet50', target_size=112):
        super().__init__(weight_path, cuda)
        if backbone == 'resnet18':
            self.model = resnet_face18()
        elif backbone == 'resnet34':
            self.model = resnet_face34()
        elif backbone == 'resnet50':
            self.model = se_resnet50_ir()

        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                              std=[0.5, 0.5, 0.5],
                                                              inplace=True)])
        self.initialize()

    @timer
    @torch.no_grad()
    def inference(self, faces):
        features = []
        if isinstance(faces, np.ndarray):
            pass
        elif isinstance(faces, list):
            if isinstance(faces[0], np.ndarray):
                faces = list(map(lambda x: Image.fromarray(cv.cvtColor(x, cv.COLOR_BGR2RGB)), faces))
            faces = list(map(self.trans, faces))
            faces = torch.stack(faces)
            if self._cuda:
                faces = faces.cuda()
            features = self.model(faces).cpu().numpy().tolist()

        else:
            pass

        return features
