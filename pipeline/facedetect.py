from __future__ import print_function
import torch
import numpy as np
import cv2 as cv

from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms
from RetinaFace.utils.box_utils import decode, decode_landm

from tool import timer
from facebase import FaceBase


class FaceDetect(FaceBase):
    def __init__(self, weight_path, cuda=False, *, cfg, confidence_threshold=.02, nms_threshold=.04, pad=True):
        super().__init__(weight_path, cuda)
        self._pad = pad
        self._cfg = cfg
        self._confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = RetinaFace(cfg=cfg)
        self.initialize()

    @timer
    @torch.no_grad()
    def inference(self, image, *, target_size=None, max_size=None, top=-1):
        faces = []
        landmark = []
        points = []
        original = image.copy()
        h, w, c = original.shape
        im = np.float32(original)
        resize = 1
        if c != 3:
            im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
        if target_size:
            im_size_min = np.min((h, w))
            im_size_max = np.max((h, w))
            resize = float(target_size) / float(im_size_min)
            if max_size and np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            im = cv.resize(im, None, None, fx=resize, fy=resize, interpolation=cv.INTER_LINEAR)

        im_height, im_width, _ = im.shape
        scale = torch.Tensor([im.shape[1], im.shape[0], im.shape[1], im.shape[0]])

        im -= (104, 117, 123)
        im /= (57, 57, 58)
        im = im.transpose(2, 0, 1)
        img = torch.from_numpy(im).unsqueeze(0)
        if self._cuda:
            img = img.cuda()
            scale = scale.cuda()

        if self._cfg['gender']:
            loc, conf, landms, gender = self.model(img)  # forward pass
        else:
            loc, conf, landms = self.model(img)  # forward pass

        priorbox = PriorBox(self._cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        if self._cuda:
            priors = priors.cuda()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self._cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        if self._cfg['gender']:
            genders = gender.squeeze(0).data.cpu().numpy()

        landms = decode_landm(landms.data.squeeze(0), prior_data, self._cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        if self._cuda:
            scale1 = scale1.cuda()
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self._confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        if self._cfg['gender']:
            genders = genders[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        if self._cfg['gender']:
            genders = genders[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        if self._cfg['gender']:
            genders = genders[keep]
        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        if self._cfg['gender']:
            dets = np.concatenate((dets, landms, genders * 10000), axis=1).astype(np.float32)
        else:
            dets = np.concatenate((dets, landms), axis=1).astype(np.float32)

        vis_thres = 0.5
        for i, b in enumerate(dets):
            b = b.tolist()
            if b[4] < vis_thres:
                continue
            if top == -1 or i < top:
                landmark.append(b[5:15])
                b[0:15] = list(map(int, b[0:15]))
                if self._pad:
                    bw = b[2] - b[0]
                    bh = b[3] - b[1]
                    if bw >= bh:
                        b[1] -= (bw - bh) // 2
                        b[3] += (bw - bh + 1) // 2
                    else:
                        b[0] -= (bh - bw) // 2
                        b[2] += (bh - bw + 1) // 2
                points.append(b[0:4])
                faces.append(original[b[1]:b[3], b[0]:b[2], :])
        return faces, landmark, points
