from __future__ import print_function
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnetv2, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from tqdm import tqdm

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


model = "mobileNet"
cpu = False

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    confidence_threshold = .02
    nms_threshold = 0.04

    origin_size = True
    if model == "resnet50":
        cfg = cfg_re50
        if cfg['gender']:
            weight_path = './weights/Resnet50_Gender_Final.pth'
        else:
            weight_path = './weights/Resnet50_Final.pth'
    else:
        cfg = cfg_mnetv2
        weight_path = './weights/MobileNet_v2_Final.pth'
    # net and model
    net = RetinaFace(cfg=cfg)
    net = load_model(net, weight_path, cpu)

    net.eval()
    print('Finished loading model!')

    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)
    ori_path = "/data/datasets/FDDB/originalPics"
    file_path = "/data/datasets/FDDB/FDDB-folds"
    for i in range(1, 11):
        with open(os.path.join(file_path, "FDDB-fold-{:02d}.txt".format(i)), "r") as f, \
                open("FDDB_result/FDDB-fold-{:02d}_{}.txt".format(i, model), "w") as fw:
            lines = f.readlines()
            for line in tqdm(lines):
                line = line.rstrip('\n')
                fw.write(line + "\n")
                image_path = os.path.join(ori_path, line) + ".jpg"
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = np.float32(img_raw)

                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img /= (57, 57, 58)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                _t['forward_pass'].tic()
                if cfg['gender']:
                    loc, conf, landms, gender = net(img)  # forward pass
                else:
                    loc, conf, landms = net(img)  # forward pass

                _t['forward_pass'].toc()
                _t['misc'].tic()
                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                if cfg['gender']:
                    genders = gender.squeeze(0).data.cpu().numpy()

                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]
                if cfg['gender']:
                    genders = genders[inds]
                # keep top-K before NMS
                order = scores.argsort()[::-1]
                # order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]
                if cfg['gender']:
                    genders = genders[order]
                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, nms_threshold)
                # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]
                if cfg['gender']:
                    genders = genders[keep]
                # keep top-K faster NMS
                # dets = dets[:args.keep_top_k, :]
                # landms = landms[:args.keep_top_k, :]
                if cfg['gender']:
                    dets = np.concatenate((dets, landms, genders * 10000), axis=1).astype(np.float32)
                else:
                    dets = np.concatenate((dets, landms), axis=1).astype(np.float32)

                _t['misc'].toc()

                vis_thres = 0.5
                fw.write(str(len(dets)) + "\n")
                for b in dets:
                    x = int(b[0])
                    y = int(b[1])
                    w = int(b[2]) - int(b[0])
                    h = int(b[3]) - int(b[1])
                    confidence = str(b[4])

                    fw.write("{} {} {} {} {}\n".format(x, y, w, h, confidence))
                    fw.flush()



