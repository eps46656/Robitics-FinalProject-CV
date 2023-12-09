from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import shutil
import warnings
import json
from tqdm import tqdm

import sys

import config

sys.path.append(os.path.join(config.LOGO_CAP_DIR))

import logocap
from logocap.config import cfg,update_config,check_config
# from logocap.config import
import logocap.dataset as D
from logocap.utils.utils import get_optimizer, create_logger, save_checkpoint, setup_logger
from logocap.trainer import do_train
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
torch.set_grad_enabled(False)
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
import math

from logocap.utils.transforms import resize_align_multi_scale, get_multi_scale_size, get_affine_transform

import torchvision
import cv2 as cv

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--ckpt',
                        default = None,
                        type=str,
                        help='checkpoint path for inference'
    )

    args = parser.parse_args([
        "--cfg",
        f"{config.LOGO_CAP_DIR}/experiments/logocap-hrnet-w32-coco.yaml",
        "--ckpt",
        f"{config.LOGO_CAP_DIR}/weights/logocap/logocap-hrnet-w32-coco.pth.tar"
    ])

    return args

def Inference(outputs):
    poses = outputs['poses']
    if poses is None:
        return None, None

    poses = poses.cpu().detach().numpy()
    scores = outputs['scores'].cpu().detach().numpy()

    final_poses = poses

    inds = logocap.LogoCapInference.nms_core(cfg, torch.from_numpy(final_poses[...,:-1]),torch.from_numpy(scores))
    inds = np.array(inds)

    final_poses = final_poses[inds]
    scores = scores[inds]


    if final_poses.shape[0] > cfg.DATASET.MAX_NUM_PEOPLE:
        _, ind = torch.tensor(scores).topk(k=cfg.DATASET.MAX_NUM_PEOPLE)
        ind = ind.numpy()
        final_poses = final_poses[ind]
        scores = scores[ind]

    return final_poses, scores

keypoint_idx_to_str = [
    "nose",
    "l-eye",
    "r-eye",
    "l-ear",
    "r-ear",
    "l-shoulder",
    "r-shoulder",
    "l-elbow",
    "r-elbow",
    "l-wrist",
    "r-wrist",
    "l-hip",
    "r-hip",
    "l-knee",
    "r-knee",
    "l-ankle",
    "r-ankle",
]

keypoint_str_to_idx = {s:i for i, s in enumerate(keypoint_idx_to_str)}

keypoint_links = [
    ["l-shoulder", "r-shoulder"],
    ["l-hip", "r-hip"],

    ["l-shoulder", "l-hip"],
    ["l-hip", "l-knee"],
    ["l-knee", "l-ankle"],

    ["l-shoulder", "l-elbow"],
    ["l-elbow", "l-wrist"],

    ["l-shoulder", "l-hip"],
    ["l-hip", "l-knee"],
    ["l-knee", "l-ankle"],

    ["r-shoulder", "r-hip"],
    ["r-hip", "r-knee"],
    ["r-knee", "r-ankle"],

    ["r-shoulder", "r-elbow"],
    ["r-elbow", "r-wrist"],

    ["r-shoulder", "r-hip"],
    ["r-hip", "r-knee"],
    ["r-knee", "r-ankle"],
]

def load_model(args):
    model, targets_encoder = logocap.models.build_model(cfg, is_train=False)

    if args.ckpt:
        model_state_file = args.ckpt
    elif cfg.TEST.MODEL_FILE != "":
        model_state_file = cfg.TEST.MODEL_FILE
    else:
        raise RuntimeError()

    model.load_state_dict(torch.load(model_state_file,map_location='cpu'))

    return model

class PoseEstimator:
    def __init__(self):
        self.model = load_model(args)
        self.model.eval()
        self.model = self.model.to('cuda')

        self.rs_img_h, self.rs_img_w = 512, 512

    def Estimate(self, img):
        # img[h, w, 3] 0 ~ 255 rgb

        assert len(img.shape) == 3
        assert img.shape[2] == 3

        img = torch.tensor(img / 255, dtype=torch.float32).permute(2, 0, 1)

        img = torchvision.transforms.functional.normalize(
            img,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225))

        img = torchvision.transforms.functional.resize(
            img, (self.rs_img_h, self.rs_img_w))

        outputs, timing_per_im = self.model(
            img.reshape(1, 3, self.rs_img_h, self.rs_img_w).cuda())

        poses, scores = Inference(outputs)

        poses[:, :, 0] /= self.rs_img_w
        poses[:, :, 1] /= self.rs_img_h

        return poses, scores

def AnnoPoses(img, poses, scores):
    img_h, img_w, _ = img.shape

    img = img.copy()

    for pose, score in zip(poses, scores):
        if score < 0.02:
            continue

        # pose[17, 3] ratio of image
        # pose[k, 0] * img_w for w idx of keypoint k
        # pose[k, 1] * img_h for h idx of keypoint k

        keypoint = np.empty((17, 2), dtype=np.int32)
        keypoint[:, 0] = np.round(pose[:, 0] * img_w).astype(np.int32)
        keypoint[:, 1] = np.round(pose[:, 1] * img_h).astype(np.int32)

        xmin = keypoint[:, 0].min().item()
        ymin = keypoint[:, 1].min().item()
        xmax = keypoint[:, 0].max().item()
        ymax = keypoint[:, 1].max().item()

        # cv.rectangle(out_img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

        for keypoint_link in keypoint_links:
            cv.line(img,
                    keypoint[keypoint_str_to_idx[keypoint_link[0]], :],
                    keypoint[keypoint_str_to_idx[keypoint_link[1]], :],
                    (0, 255, 0), 2)

        for keypoint in pose:
            x = int(round(keypoint[0] * img_w))
            y = int(round(keypoint[1] * img_h))
            cv.circle(img, (x, y), 0, (255, 0, 0), 5)

    return img

args = parse_args()
update_config(cfg, args)
check_config(cfg)
