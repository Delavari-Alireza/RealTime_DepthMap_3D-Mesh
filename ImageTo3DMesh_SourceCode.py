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

sys.path.append('./Depth_Anything_V2/metric_depth')

torch.cuda.empty_cache()

from depth_anything_v2.dpt import DepthAnythingV2

parser = argparse.ArgumentParser(description='Using Depth Anything V2 for video')

parser.add_argument('--img-path', type=str, default='./pictures/kitchen.jpg')

parser.add_argument('--focal-length-x', type=float, default=470.5,
                    help='Focal length of the camera in pixels (x-direction)')
parser.add_argument('--focal-length-y', type=float, default=470.5,
                    help='Focal length of the camera in pixels (y-direction)')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

# encoder = 'vitl' # or 'vits', 'vitb'
# dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
# max_depth = 10 # 20 for indoor model, 80 for outdoor model

# model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
# model.load_state_dict(torch.load('/home/honamic1/inetrface depthV2Video/Depth_Anything_V2/metric_depth/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
# model = model.to(DEVICE).eval()


encoder = 'vits'  # or 'vits', 'vitb'
dataset = 'hypersim'  # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20  # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(
    'PATH_to/depth_anything_v2_metric_hypersim_vits.pth',
    map_location='cpu'))
model = model.to(DEVICE).eval()

filename = args.img_path
focal_length_x = args.focal_length_x
focal_length_y = args.focal_length_y

# Load the image


# image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")


color_image = Image.open(filename).convert('RGB')
width, height = color_image.size

# Read the image using OpenCV
image = cv2.imread(filename)

# inputs = image_processor(images=color_image, return_tensors="pt")
# image = np.array(inputs['pixel_values'])[0].transpose((1,2,0))
# try:
pred = model.infer_image(image, height)

depth_normalized = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('depth map', np.array(depth_normalized, dtype=np.uint8))

# except:


# image = cv2.resize(image, (int(width/2), int(height/2)))
# # color_image = color_image.resize(int(width/2), int(height/2))
# color_image = color_image.resize((int(width/2), int(height/2)), Image.NEAREST)
# width  = int(width/2)
# height = int(height/2)
# pred = model.infer_image(image, height)
# Resize depth prediction to match the original image size
resized_pred = Image.fromarray(pred).resize((width, height), Image.NEAREST)

# cv2.imshow('depth map',pred)

# Generate mesh grid and calculate point cloud coordinates
x, y = np.meshgrid(np.arange(width), np.arange(height))
x = (x - width / 2) / focal_length_x
y = (y - height / 2) / 470.4
z = np.array(resized_pred)
points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
colors = np.array(color_image).reshape(-1, 3) / 255.0

# Create the point cloud and save it to the output directory
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
# o3d.io.write_point_cloud(("pcd.ply"), pcd)


# pcd = o3d.io.read_point_cloud("/home/honamic1/Downloads/tmp/pcd.ply")


o3d.visualization.draw_geometries([pcd])

k = cv2.waitKey(0) & 0xff
if k == 27:
    exit()
cv2.destroyAllWindows()
