import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import csv
import torch
from models import *
from torchvision import transforms
import json
input_shape = 112


def get_img_pairs_list(pairs_txt_path):
    pair_list = []
    with open(pairs_txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            pair_list.append(line.rstrip('\n').split(' '))
    return pair_list


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def threshold_search(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    return best_acc, best_th


@torch.no_grad()
def compute_accuracy(pair_list, _model, _preprocess, img_path):
    _model.eval()
    similarities = []
    labels = []
    with open("lfw_embedding.csv", "w") as f:
        writer = csv.writer(f)
        for pair in tqdm(pair_list):
            img0 = Image.open(os.path.join(img_path, pair[0]))
            img1 = Image.open(os.path.join(img_path, pair[1]))
            label = min(0, int(pair[2]))
            embedding0 = _model(_preprocess(img0).unsqueeze(0).cuda())[0].data.cpu().numpy()
            embedding1 = _model(_preprocess(img1).unsqueeze(0).cuda())[0].data.cpu().numpy()
            writer.writerow([embedding0.tolist(), embedding1.tolist(), label])

            similarity = cosin_metric(embedding0, embedding1)
            similarities.append(similarity)
            labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold


def compute_accuracy_csv():
    similarities = []
    labels = []
    with open("lfw_embedding.csv", "r") as f:
        reader = csv.reader(f)
        for line in tqdm(reader):
            label = int(line[2])
            embedding0 = json.loads(line[0])
            embedding1 = json.loads(line[1])

            similarity = cosin_metric(embedding0, embedding1)
            print(embedding0)
            print(embedding1)
            similarities.append(similarity)
            labels.append(label)

    accuracy, threshold = threshold_search(similarities, labels)
    return accuracy, threshold

# model = se_resnet_face50().cuda()
# model.load_state_dict(torch.load("./weight/arcface_resnet50_epoch_50.pth", map_location='cuda'))
# trans = transforms.Compose([transforms.CenterCrop(input_shape),
#                             transforms.ToTensor(),
#                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
#
# pair_list = get_img_pairs_list("../validation/lfw_pair.txt")
# accuracy, threshold = compute_accuracy(pair_list, model, trans, "../validation/lfw/")
# print(accuracy, threshold)

# compute_accuracy_csv()

model = se_resnet_face50()
model.eval()
# model.load_state_dict(torch.load("./arcface_resnet50_epoch_50.pth", map_location='cpu'))
a = torch.zeros([1, 3, 112, 112])
b = torch.ones([1, 3, 112, 112])
print(a.shape)
print(model(a))
print(model(b))