import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import open3d as o3d
import torch
import argparse
import os
import matplotlib.pyplot as plt
# from transformers import AutoImageProcessor

import sys

sys.path.append('/media/honamic1/New Volume/DepthAnything V2/inetrface depthV2Video/Depth_Anything_V2/metric_depth')

torch.cuda.empty_cache()

from depth_anything_v2.dpt import DepthAnythingV2

parser = argparse.ArgumentParser(description='Using Depth Anything V2 for video')

# parser.add_argument('--img-path', type=str , default='/home/honamic1/Downloads/5855006285739705328.jpg')

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vits'  # or 'vits', 'vitb'
# dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 80  # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load('/media/honamic1/New Volume/DepthAnything V2/depth_anything_v2_metric_vkitti_vits.pth',
                                 map_location='cpu'))
# model.load_state_dict(torch.load('/home/honamic1/inetrface depthV2Video/Depth_Anything_V2/metric_depth/depth_anything_v2_metric_hypersim_vits.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# 1 sal 5 50 ta 8 ip
# radio
# access FloatingPointError8700
# 1000
# 1650
# 11300

# service


# filename = "/home/honamic1/Downloads/pictues/907c74bd-90e9-49c1-bb2f-fb4b5366dbc5.jpeg" #args.img_path

filename = "/media/honamic1/New Volume/DepthAnything V2/pictues/imgTeh/image_teh/292836.jpg"  # args.img_path

color_image = Image.open(filename).convert('RGB')
width, height = color_image.size

# Read the image using OpenCV
image = cv2.imread(filename)

pred = model.infer_image(image, height)

depth_normalized = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)

# cv2.imshow('depth map',np.array(cv2.resize(pred, (320, 240)), dtype = np.uint8 ))
# cv2.imshow('depth map normalized',np.array(cv2.resize(depth_normalized, (320, 240)), dtype = np.uint8 ))
# cv2.imshow('input image ',np.array(cv2.resize(image, (320, 240)), dtype = np.uint8 ))


cv2.imshow('depth map', np.array(pred, dtype=np.uint8))
cv2.imshow('depth map normalized', np.array(depth_normalized, dtype=np.uint8))
cv2.imshow('input image ', np.array(image, dtype=np.uint8))

# cv2.imshow('depth map',pred)


k = cv2.waitKey(0) & 0xff
if k == 27:
    exit()

cv2.destroyAllWindows()
exit()