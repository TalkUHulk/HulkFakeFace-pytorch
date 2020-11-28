import torch
from torch import nn
from torchviz import make_dot

from models.retinaface import RetinaFace
from data import cfg_mnetv2, cfg_re50
net = RetinaFace(cfg=cfg_re50)

x = torch.randn(1, 3, 224, 224).requires_grad_(True)
y = net(x)
vis_graph = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
vis_graph.view(filename="viz/resnet50.gv")
