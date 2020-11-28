import torch
import torch.nn as nn
from model import ResNet
from torchvision import transforms
import os
import cv2 as cv
import csv
from PIL import Image
from tqdm import tqdm
import csv
import numpy as np
import requests

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
input_size = 112
transform = transforms.Compose([transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std, inplace=True)])

model = ResNet()
model.load_state_dict(torch.load('weight/gender_epoch_99.pkl', map_location='cpu')['model_state_dict'])
model.eval()

with torch.no_grad():
    img = "./data/head.png"
    bgr = cv.imread(img)
    if len(bgr.shape) != 3:
        bgr = cv.cvtColor(bgr, cv.COLOR_GRAY2BGR)

    tensor = transform(Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))).unsqueeze(0)

    _, output = model(tensor)
    logits = torch.softmax(output, dim=1)
    print(logits)
    scores = logits.numpy()[0]
    prediction = torch.argmax(logits, 1).numpy()[0]
    print(prediction, "男" if prediction == 1 else "女")


