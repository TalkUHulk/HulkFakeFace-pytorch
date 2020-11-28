import torch
import torch.utils.data as Data
import torch.nn as nn
from data import ScutFBPDatasets
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import datetime
from tqdm import tqdm
from pipeline.utils import *
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import heapq

class Trainer:
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

        if config.fine_tune:
            checkpoint = torch.load(config.pretrained_model, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if not os.path.exists(self.config.log_dir):
            os.mkdir(self.config.log_dir)

        if not os.path.exists(self.config.weight_dir):
            os.mkdir(self.config.weight_dir)

        self.loss_record = []

        logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                            filename=os.path.join(self.config.log_dir, 'Trainer_{}.log'.format(
                                datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                            )),
                            filemode='w',
                            format=
                            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                            )

        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(logdir=self.config.log_dir)

        if config.fine_tune:
            self.criterion1 = torch.nn.MarginRankingLoss()
            self.criterion2 = torch.nn.SmoothL1Loss()
        elif config.loss == 'focal_loss':
            self.criterion = FocalLoss(gamma=2)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if config.fine_tune:
            self.metric_fc = nn.Softmax(dim=1)
        elif config.metric == 'add_margin':
            self.metric_fc = AddMarginProduct(5, config.num_classes, s=30, m=0.35)
        elif config.metric == 'arc_margin':
            self.metric_fc = ArcMarginProduct(5, config.num_classes, s=30, m=0.5, easy_margin=config.easy_margin)
        elif config.metric == 'sphere':
            self.metric_fc = SphereProduct(5, config.num_classes, m=4)
        else:
            self.metric_fc = nn.Linear(5, config.num_classes)

        if config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                [{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
                lr=config.lr, weight_decay=config.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                [{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
                lr=config.lr, weight_decay=config.weight_decay)

        self.cuda = config.cuda and torch.cuda.is_available()

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_step, gamma=0.1)
        self.scheduler_warmup = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1,
            total_epoch=5,
            after_scheduler=self.scheduler)

        self.train_dh = ScutFBPDatasets(img_path=config.img_path, txt_path=config.train_txt)

        self.val_dh = ScutFBPDatasets(img_path=config.img_path, txt_path=config.val_txt)

        self.train_loader = Data.DataLoader(
            self.train_dh, batch_size=config.batch_size, shuffle=True)
        self.val_loader = Data.DataLoader(
            self.val_dh, batch_size=config.val_batch_size, shuffle=False)

        if self.cuda:
            self.model.cuda()
            if config.fine_tune:
                self.criterion1.cuda()
                self.criterion2.cuda()
            else:
                self.criterion.cuda()
            self.metric_fc.cuda()

    def lr_decay(self, epoch):
        self.scheduler_warmup.step(epoch)

    def epoch_n_from_weights_name(self, w_name):
        """
        Extracts the last epoch number from the standardized weights name.
            :discriminatorR_epoch_6.pkl
        """
        try:
            starting_epoch = int(w_name.split('_')[-1].rstrip('.pkl'))
        except Exception as e:
            self.logger.warning(
                'Could not retrieve starting epoch from the weights name: \n{}'.format(w_name)
            )
            self.logger.error(e)
            starting_epoch = 0
        return starting_epoch

    def remove_old_weights(self, max_n_weights):
        """
        Scans the weights folder and removes all but:
            - max_n_weights most recent 'others' weights.
        """
        top_n_index = list(map(self.loss_record.index, heapq.nsmallest(max_n_weights, self.loss_record)))

        self.logger.debug(
            "####### Min weight index:{}".format(top_n_index)
        )

        w_list = [w for w in os.scandir(self.weight_dir) if w.name.endswith('.pkl')]

        for w in w_list:
            if self.epoch_n_from_weights_name(w.name) not in top_n_index:
                os.remove(w.path)

    def save_weights(self, epoch):

        if self.config.fine_tune:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       os.path.join(self.config.weight_dir, 'beauty_ft_epoch_{}.pth'.format(epoch)))
        else:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()},
                       os.path.join(self.config.weight_dir, 'beauty_epoch_{}.pth'.format(epoch)))

        # try:
        #     self.remove_old_weights(self.max_n_weights)
        # except Exception as e:
        #     self.logger.warning('Could not remove weights: {}'.format(e))

    def validation_fine_tune(self, epoch):
        self.model.eval()
        with torch.no_grad():
            cnt = 0
            total_l1 = 0
            total_rank = 0
            with tqdm(desc='validation %d' % epoch, total=len(self.val_loader)) as pbar:
                for i, data_batch in enumerate(self.val_loader):
                    data, label = data_batch
                    if data.shape[0] % 2:
                        need_index = torch.randperm(data.shape[0])[:data.shape[0] - 1]
                        data = data[need_index]
                        label = label[need_index]
                    if self.cuda:
                        data = data.cuda()
                        label = label.cuda()

                    feature = self.model(data)
                    output = self.metric_fc(feature)

                    if self.cuda:
                        output = torch.sum(output * torch.Tensor([1, 2, 3, 4, 5]).cuda(), dim=1, keepdim=True)
                        index = torch.Tensor([True for i in range(data.shape[0])]).cuda()
                        a_index = torch.from_numpy(
                            np.random.choice(
                                [i for i in range(data.shape[0])],
                                data.shape[0] // 2,
                                replace=False)
                        ).cuda()
                    else:
                        output = torch.sum(output * torch.Tensor([1, 2, 3, 4, 5]), dim=1, keepdim=True)
                        index = torch.Tensor([True for i in range(data.shape[0])])
                        a_index = torch.from_numpy(
                            np.random.choice(
                                [i for i in range(data.shape[0])],
                                data.shape[0] // 2,
                                replace=False)
                        )
                    index[a_index] = False
                    output_a = output[index == True]
                    label_a = label[index == True]
                    output_b = output[index == False]
                    label_b = label[index == False]
                    loss_rank = self.criterion1(input1=output_a.squeeze(),
                                                input2=output_b.squeeze(),
                                                target=(label_a - label_b).sign())

                    loss_l1 = self.criterion2(input=output.squeeze(), target=label)
                    cnt += 1
                    total_l1 += loss_l1.item()
                    total_rank += loss_rank.item()
                    pbar.update()

            total_l1 /= cnt
            total_rank /= cnt
            self.writer.add_scalars(
                'validation/Loss',
                {"L1": total_l1,
                 "Rank": total_rank},
                epoch)
            self.logger.debug(
                "[Val %d] [L1: %f] [RankLoss: %f]"
                % (epoch, total_l1, total_rank)
            )

    def validation(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            total_gap = 0
            with tqdm(desc='validation %d' % epoch, total=len(self.val_loader)) as pbar:
                for i, data_batch in enumerate(self.val_loader):
                    data, label = data_batch
                    if self.cuda:
                        data = data.cuda()
                        label = label.cuda()

                    feature = self.model(data)
                    output = self.metric_fc(feature, label)
                    loss = self.criterion(output, label)

                    feature = torch.softmax(feature, dim=1).data.cpu().numpy()
                    score = np.sum(feature * np.array([1, 2, 3, 4, 5]), axis=-1)
                    label = label.data.cpu().numpy()
                    gap = np.mean((np.abs(score - label - 1)).astype(float))

                    cnt += 1
                    total_loss += loss.item()
                    total_gap += gap
                    pbar.update()

            total_loss /= cnt
            total_gap /= cnt

            self.loss_record.append(total_loss)
            self.writer.add_scalar('validation/loss', total_loss, epoch)
            self.writer.add_scalar('validation/gap', total_gap, epoch)
            self.logger.debug(
                "[Val %d] [loss: %f] [gap: %f]"
                % (epoch, total_loss, total_gap)
            )

    def train_fine_tune(self, epoch):
        with tqdm(desc='epoch %d' % epoch, total=len(self.train_loader)) as pbar:
            self.model.train()
            for i, data_batch in enumerate(self.train_loader):
                data, label = data_batch
                if data.shape[0] % 2:
                    need_index = torch.randperm(data.shape[0])[:data.shape[0] - 1]
                    data = data[need_index]
                    label = label[need_index]
                if self.cuda:
                    label = label.cuda(non_blocking=True)
                    data = data.cuda(non_blocking=True)

                feature = self.model(data)
                output = self.metric_fc(feature)

                if self.cuda:
                    output = torch.sum(output * torch.Tensor([1, 2, 3, 4, 5]).cuda(), dim=1, keepdim=True)
                    index = torch.Tensor([True for i in range(data.shape[0])]).cuda()
                    a_index = torch.from_numpy(
                        np.random.choice(
                            [i for i in range(data.shape[0])],
                            data.shape[0] // 2,
                            replace=False)
                    ).cuda()
                else:
                    output = torch.sum(output * torch.Tensor([1, 2, 3, 4, 5]), dim=1, keepdim=True)
                    index = torch.Tensor([True for i in range(data.shape[0])])
                    a_index = torch.from_numpy(
                        np.random.choice(
                            [i for i in range(data.shape[0])],
                            data.shape[0] // 2,
                            replace=False)
                    )
                index[a_index] = False
                output_a = output[index == True]
                label_a = label[index == True]
                output_b = output[index == False]
                label_b = label[index == False]
                loss_Rank = self.criterion1(input1=output_a.squeeze(),
                                            input2=output_b.squeeze(),
                                            target=(label_a - label_b).sign())
                loss_L1 = self.criterion2(input=output.squeeze(), target=label)
                loss = loss_Rank + self.config.bias * loss_L1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % self.config.log_steps == 0:
                    self.writer.add_scalars('Loss/train_loss',
                                            {"loss": loss.item(),
                                             "loss_L1": loss_L1.item(),
                                             "loss_Rank": loss_Rank.item()},
                                            len(self.train_loader) * epoch + i)

                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar('LR',
                                               param_group['lr'],
                                               len(self.train_loader) * epoch + i)

                    self.logger.debug(
                        "Train Epoch {} [{}/{}]"
                        "Loss_L1: {} Loss_Rank: {} Loss: {}".format(
                            epoch, i, len(self.train_loader),
                            loss_L1.item(),
                            loss_Rank.item(),
                            loss.item())
                    )

                pbar.update()

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
                loss.backward()
                self.optimizer.step()

                if i % self.config.log_steps == 0:
                    feature = torch.softmax(feature, dim=1).data.cpu().numpy()
                    score = np.sum(feature * np.array([1, 2, 3, 4, 5]), axis=-1)
                    label = label.data.cpu().numpy()
                    gap = np.mean((np.abs(score - label - 1)).astype(float))

                    self.writer.add_scalar('Loss/train_loss',
                                           loss.item(),
                                           len(self.train_loader) * epoch + i)

                    self.writer.add_scalar('Accuracy/train_gap',
                                           gap,
                                           len(self.train_loader) * epoch + i)

                    for param_group in self.optimizer.param_groups:
                        self.writer.add_scalar('LR',
                                               param_group['lr'],
                                               len(self.train_loader) * epoch + i)

                    self.logger.debug(
                        "Train Epoch {} [{}/{}]"
                        "Loss: {}, Gap: {}' ".format(epoch, i, len(self.train_loader),
                                                     loss.item(), gap)
                    )

                pbar.update()

    def run(self):
        for epoch in range(self.config.epochs):
            if self.config.fine_tune:
                self.train_fine_tune(epoch)
                self.validation_fine_tune(epoch)
            else:
                self.train(epoch)
                self.validation(epoch)
            # self.scheduler.step()
            self.scheduler_warmup.step()
            # save
            if (epoch + 1) % self.config.save_epoch == 0:
                self.save_weights(epoch + 1)
