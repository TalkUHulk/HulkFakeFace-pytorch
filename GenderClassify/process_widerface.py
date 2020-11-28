import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from model import ResNet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]
input_size = 112
transform = transforms.Compose([transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std, inplace=True)])

# 4135
model = ResNet()
model.load_state_dict(torch.load('weight/gender_epoch_99.pkl', map_location='cpu')['model_state_dict'])
model.eval()

txt_path = "/data/datasets/widerface/train/label.txt"
imgs_path = []
words = []
f = open(txt_path, 'r')
lines = f.readlines()
isFirst = True
labels = []
for line in lines:
    line = line.rstrip()
    if line.startswith('#'):
        if isFirst is True:
            isFirst = False
        else:
            labels_copy = labels.copy()
            words.append(labels_copy)
            labels.clear()
        path = line[2:]
        path = txt_path.replace('label.txt', 'images/') + path
        imgs_path.append(path)
    else:
        line = line.split(' ')
        label = [x for x in line]
        labels.append(label)

words.append(labels)

with open("./label_with_gender.txt", "w") as fw, torch.no_grad():
    for index in tqdm(range(len(imgs_path))):
        try:
            fw.write("# " + imgs_path[index].split('images/')[-1] + '\n')
            fw.flush()
            #print("~~~~~", "# " + imgs_path[index].split('images/')[-1] + '\n')
            img = cv.imread(imgs_path[index])
            height, width, _ = img.shape

            if len(img.shape) != 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            labels = words[index]
            annotations = np.zeros((0, 15))
            if len(labels) == 0:
                continue
            for idx, label in enumerate(labels):
                #print(int(label[1]), int(label[1]) + int(label[3]), int(label[0]), int(label[0]) + int(label[2]))
                face_roi = img[int(label[1]): int(label[1]) + int(label[3]), int(label[0]): int(label[0]) + int(label[2]), :]
                #cv.imwrite("face.png", face_roi)
                label.append('0')

                tensor = transform(Image.fromarray(cv.cvtColor(face_roi, cv.COLOR_BGR2RGB))).unsqueeze(0)
                _, output = model(tensor)
                logits = torch.softmax(output, dim=1)
                scores = logits.numpy()[0]
                prediction = torch.argmax(logits, 1).numpy()[0]
                gender = 1 if prediction == 1 else 0

                if float(label[4]) < 0:
                    label.append("-1")
                else:
                    label.append(str(gender))

                #print(' '.join(label))
                fw.write(' '.join(label) + '\n')
                fw.flush()
        except Exception as e:
            print(e)
