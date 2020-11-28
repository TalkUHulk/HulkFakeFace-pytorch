import torch
import torch.utils.data as Data
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.optim as optim
from tensorboardX import SummaryWriter
import logging
import os
import datetime
from tqdm import tqdm
from models import *
import utils
from utils import *
from data import *
import numpy as np
import pprint
from warmup_scheduler import GradualWarmupScheduler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except:
    print("not support apex")


class Trainer:
    def __init__(
            self,
            config

    ):
        self.config = config
        pprint.pprint(config.__dict__)

        if config.backbone == 'resnet18':
            self.model = resnet_face18()
        elif config.backbone == 'resnet34':
            self.model = resnet_face34()
        elif config.backbone == 'resnet50':
            self.model = se_resnet50_ir()

        if config.pretrained:
            checkpoint = torch.load(config.pretrained, map_location="cpu")
            self.model.load_state_dict(checkpoint)
            print("Load pretrained model!")

        self.train_counter = 0
        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        if not os.path.exists(self.config.weight_dir):
            os.mkdir(self.config.weight_dir)

        logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                            filename=os.path.join(self.config.log_dir, 'Trainer_{}.log'.format(
                                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                            )),
                            filemode='w',
                            format=
                            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )

        self.train_dh = ImageFolder(config.train_datasets,
                                    transform=trans,
                                    target_transform=target_trans,
                                    is_valid_file=lambda x: x.endswith(('.jpg', '.png', 'jpeg'))
                                    )
        # print("ImageFolder.__class__", self.train_dh.class_to_idx)
        assert len(self.train_dh.classes) == config.num_classes
        self.train_loader = Data.DataLoader(
            self.train_dh,
            batch_size=self.config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=True
        )
        print("Load datasets!")
        self.lfw_list = get_lfw_list(config.lfw_test_list)
        self.lfw_paths = [os.path.join(config.lfw_root, each) for each in self.lfw_list]

        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(logdir=self.config.log_dir)

        if config.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma=2)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if config.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(512, config.num_classes, s=30, m=0.35)
        elif config.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512, config.num_classes, s=64, m=0.5, easy_margin=config.easy_margin, fp16=config.fp16)
        elif config.metric == 'sphere':
            self.metric_fc = SphereProduct(512, config.num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(512, config.num_classes)

        if config.optimizer == 'sgd':
            # self.optimizer = torch.optim.SGD(
            #     [{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
            #     lr=config.lr, weight_decay=config.weight_decay)
            self.optimizer = torch.optim.SGD(
                self.add_weight_decay(self.model) + self.add_weight_decay(self.metric_fc),
                lr=config.lr, momentum=config.momentum)
        else:
            self.optimizer = torch.optim.Adam(
                [{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
                lr=config.lr, weight_decay=config.weight_decay)

        # lr decay

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_step, gamma=0.1)
        self.scheduler_warmup = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1,
            total_epoch=3,
            after_scheduler=self.scheduler)

        self.cuda = config.cuda and torch.cuda.is_available()

        if self.cuda:
            self.criterion.cuda()
            self.model.cuda()
            self.metric_fc.cuda()
        if not self.config.fp16 and config.num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.metric_fc = torch.nn.DataParallel(self.metric_fc).cuda()

        if self.config.fp16:
            amp.register_float_function(utils, 'ArcMarginProduct')
            [self.model, self.metric_fc], self.optimizer = amp.initialize([self.model, self.metric_fc],
                                                                          self.optimizer,
                                                                          opt_level=config.opt_level
                                                                          )
            if config.num_gpu > 1:
                self.model = DDP(self.model)

    def add_weight_decay(self, model, weight_decay=5e-4, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]

    def save_weights(self, epoch):

        # torch.save({'epoch': epoch,
        #             'model_state_dict': self.model.state_dict(),
        #             'optimizer_state_dict': self.optimizer.state_dict()},
        #            os.path.join(self.config.weight_dir, 'spam_epoch_{}.pkl'.format(epoch)))

        if epoch == self.config.epochs:
            torch.save(self.model.state_dict(),
                       os.path.join(self.config.weight_dir,
                                    'arcface_{}_Final.pth'.format(self.config.backbone)))
            torch.save(self.metric_fc.state_dict(),
                       os.path.join(self.config.weight_dir,
                                    'arcface_{}_Final.pth'.format(self.config.metric)))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.config.weight_dir,
                                    'arcface_{}_Final.pth'.format(self.config.optimizer)))
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(self.config.weight_dir,
                                    'arcface_{}_epoch_{}.pth'.format(self.config.backbone, epoch)))
            torch.save(self.metric_fc.state_dict(),
                       os.path.join(self.config.weight_dir,
                                    'arcface_{}_epoch_{}.pth'.format(self.config.metric, epoch)))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(self.config.weight_dir,
                                    'arcface_{}_epoch_{}.pth'.format(self.config.optimizer, epoch)))

    @torch.no_grad()
    def validation(self, epoch, log=True):
        self.model.eval()
        acc = lfw_test(self.model, self.lfw_paths, self.lfw_list, self.config.lfw_test_list,
                       self.config.test_batch_size)
        if log:
            self.writer.add_scalar('Accuracy/lfw_acc', acc, epoch)

    def train(self, epoch):
        with tqdm(desc='epoch %d' % epoch, total=len(self.train_loader)) as pbar:
            self.model.train()
            for i, data_batch in enumerate(self.train_loader):
                data, label = data_batch

                if self.cuda:
                    label = label.cuda(non_blocking=True)
                    data = data.cuda(non_blocking=True)

                feature = self.model(data)
                output = self.metric_fc(feature, label)
                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                if self.config.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss_model:
                        scaled_loss_model.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                if i % self.config.log_steps == 0:
                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    label = label.data.cpu().numpy()
                    acc = np.mean((output == label).astype(int))

                    self.writer.add_scalar('Loss/train_loss',
                                           loss.item(),
                                           self.train_counter)

                    self.writer.add_scalar('Accuracy/train_acc',
                                           acc,
                                           self.train_counter)

                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar('LR',
                                               param_group['lr'],
                                               self.train_counter)

                    self.train_counter += 1

                    self.logger.debug(
                        "Train Epoch {} [{}/{}]"
                        "Loss: {}, Acc: {}' ".format(epoch, i, len(self.train_loader),
                                                     loss.item(), acc)
                    )
                    print(
                        "Train Epoch {} [{}/{}]"
                        "Loss: {}, Acc: {}' ".format(epoch, i, len(self.train_loader),
                                                     loss.item(), acc)
                    )

                pbar.update()

    def run(self):
        # self.validation(0, False)
        for epoch in range(self.config.epochs):
            self.train(epoch)
            self.validation(epoch)
            # self.scheduler.step()
            self.scheduler_warmup.step()
            # save
            if (epoch + 1) % self.config.save_epoch == 0:
                self.save_weights(epoch + 1)
