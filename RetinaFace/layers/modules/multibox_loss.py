from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.box_utils import match, match_gender, log_sum_exp
from data import cfg_mnetv2
GPU = cfg_mnetv2['gpu_train']


class MultiBoxLossWithGender(nn.Module, ABC):
    """
    SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLossWithGender, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.bce = nn.BCELoss(reduction="sum")
    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data, gender_data = predictions
        priors = priors
        num = loc_data.size(0)  #batch
        num_priors = (priors.size(0))   # 先验框个数
        # 获取匹配每个prior box的 ground truth
        # 创建 loc_t 和 conf_t 保存真实box的位置和类别
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        gender_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data       # ground truth box信息
            labels = targets[idx][:, 14].data       # ground truth conf信息
            landms = targets[idx][:, 4:14].data     # ground truth landmark信息
            genders = targets[idx][:, -1].data    # ground truth gender信息
            defaults = priors.data                  # priors的 box 信息

            # 匹配 ground truth
            match_gender(self.threshold, truths, defaults, self.variance, labels, landms, genders, loc_t, conf_t, landm_t, gender_t, idx)

        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            gender_t = gender_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros

        num_pos_landm = pos1.long().sum(1, keepdim=True)            # 匹配中所有的正样本
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # gender Loss (binary_cross_entropy)
        # Shape: [batch,num_priors,10]
        # predict: gender_data
        # target: gender_t
        # torch.nn.functional.one_hot(l, 2)

        # b_face = torch.tensor(0).cuda()
        b_face = torch.tensor(0.9).cuda()
        pos2 = conf_t > b_face
        num_pos_gender = pos2.long().sum(1, keepdim=True)  # 匹配中所有的正样本
        N2 = max(num_pos_gender.data.sum().float(), 1)
        pos_idx2 = pos2.unsqueeze(pos2.dim()).expand_as(gender_data)

        gender_p = gender_data[pos_idx2].view(-1, 2)
        gender_t = gender_t[pos2]
        gender_t = torch.nn.functional.one_hot(gender_t.long(), 2).float()
        loss_gender = self.bce(torch.softmax(gender_p, dim=1), gender_t)

        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)           # 预测的正样本box信息
        loc_t = loc_t[pos_idx].view(-1, 4)              # 真实的正样本box信息
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')        # Smooth L1 损失

        '''
        Target；
            下面进行hard negative mining
        过程:
            1、 针对所有batch的conf，按照置信度误差(预测背景的置信度越小，误差越大)进行降序排列;
            2、 负样本的label全是背景，那么利用log softmax 计算出logP,
               logP越大，则背景概率越低,误差越大;
            3、 选取误差较大的top_k作为负样本，保证正负样本比例接近1:3;
        '''

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 使用logsoftmax，计算置信度
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now 把正样本排除，剩下的就全是负样本，可以进行抽样
        loss_c = loss_c.view(num, -1)
        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 抽取负样本
        # 每个batch中正样本的数目
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)    # 抽取前top_k个负样本
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        loss_gender /= N2

        return loss_l, loss_c, loss_landm, loss_gender


class MultiBoxLoss(nn.Module, ABC):
    """
    SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)  #batch
        num_priors = (priors.size(0))   # 先验框个数

        # 获取匹配每个prior box的 ground truth
        # 创建 loc_t 和 conf_t 保存真实box的位置和类别
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data       # ground truth box信息
            labels = targets[idx][:, -1].data       # ground truth conf信息
            landms = targets[idx][:, 4:14].data     # ground truth landmark信息
            defaults = priors.data                  # priors的 box 信息
            # 匹配 ground truth
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)            # 匹配中所有的正样本
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        landm_p = landm_data[pos_idx1].view(-1, 10)
        landm_t = landm_t[pos_idx1].view(-1, 10)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')


        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)           # 预测的正样本box信息
        loc_t = loc_t[pos_idx].view(-1, 4)              # 真实的正样本box信息
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')        # Smooth L1 损失

        '''
        Target；
            下面进行hard negative mining
        过程:
            1、 针对所有batch的conf，按照置信度误差(预测背景的置信度越小，误差越大)进行降序排列;
            2、 负样本的label全是背景，那么利用log softmax 计算出logP,
               logP越大，则背景概率越低,误差越大;
            3、 选取误差较大的top_k作为负样本，保证正负样本比例接近1:3;
        '''

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 使用logsoftmax，计算置信度
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now 把正样本排除，剩下的就全是负样本，可以进行抽样
        loss_c = loss_c.view(num, -1)
        # 两次sort排序，能够得到每个元素在降序排列中的位置idx_rank
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        # 抽取负样本
        # 每个batch中正样本的数目
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)    # 抽取前top_k个负样本

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 提取出所有筛选好的正负样本(预测的和真实的)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
