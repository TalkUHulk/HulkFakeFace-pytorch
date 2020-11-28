import torch
import torch.utils.data as Data
import torch.nn as nn
from data import ScutFBPDatasets
import logging
import os
import datetime
from tqdm import tqdm
from models import *
import numpy as np


class Tester:
    def __init__(
        self,
        config
    ):
        self.config = config
        if config.backbone == 'resnet18':
            self.model = resnet_face18()
        elif config.backbone == 'resnet34':
            self.model = resnet_face34()
        elif config.backbone == 'resnet50':
            self.model = se_resnet_face50(out_planes=5)

        self.metric_fc = nn.Softmax(dim=1)

        self.cuda = config.cuda and torch.cuda.is_available()

        self.test_dh = ScutFBPDatasets(img_path=config.img_path, txt_path=config.test_txt, test=True)

        self.test_loader = Data.DataLoader(
            self.test_dh, batch_size=config.test_batch_size, shuffle=False)

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                            filename=os.path.join(self.config.log_dir, 'Tester.log'),
                            filemode='a',
                            format=
                            '%(asctime)s - %(message)s'
                            )

        self.logger = logging.getLogger(__name__)

        if self.cuda:
            self.model.cuda()
            self.metric_fc.cuda()

        checkpoint = torch.load(config.pretrained_model, map_location='cuda' if self.cuda else 'cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def run(self):
        self.model.eval()
        with torch.no_grad():
            avg_dis = .0
            cnt = 0
            with tqdm(desc='Testing', total=len(self.test_loader)) as pbar:
                for i, data_batch in enumerate(self.test_loader):
                    data, label = data_batch
                    if self.cuda:
                        data = data.cuda()
                        label = label.cuda()

                    feature = self.model(data)
                    output = self.metric_fc(feature)
                    if self.cuda:
                        score = torch.sum(output * torch.Tensor([1, 2, 3, 4, 5]).cuda(), dim=1, keepdim=True)
                    else:
                        score = torch.sum(output * torch.Tensor([1, 2, 3, 4, 5]), dim=1, keepdim=True)
                    label = label.data.cpu().numpy()
                    score = score.data.cpu().numpy()
                    dis = np.mean((np.abs(score - label)).astype(float))

                    cnt += 1
                    avg_dis += dis
                    pbar.update()

            avg_dis /= cnt
            self.logger.info("Model:{} Total Image:[{}] average distance:{}".format(
                self.config.pretrained_model.split('/')[-1],
                len(self.test_loader),
                avg_dis,
            ))



