import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import argparse
import open3d as o3d

parser = argparse.ArgumentParser(description='Image to 3D')
parser.add_argument('--img-path', type=str, default='/pictues/kitchen.png')

args = parser.parse_args()

filename = args.img_path

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device_map='cuda')

image = Image.open(filename)

width, height = image.size
depth = pipe(image)["depth"]

depth = np.array(depth)

cv2.imshow('depth', depth)

# cv2.imwrite("cat_depth.png" , depth)


depth = -depth

cv2.imshow('envert depth', depth)
# depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
# depth_normalized = depth


resized_pred = Image.fromarray(depth_normalized).resize((width, height), Image.NEAREST)

x, y = np.meshgrid(np.arange(width), np.arange(height))
x = (x - width / 2) / 400  # 470.5
y = (y - height / 2) / 400  # 470.5
z = np.array(resized_pred)
points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
colors = np.array(image).reshape(-1, 3) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# o3d.io.write_point_cloud(("pcd.ply"), pcd)


# pcd = o3d.io.read_point_cloud("Downloads/tmp/pcd.ply")

o3d.visualization.draw_geometries([pcd])

k = cv2.waitKey(0) & 0xff
if k == 27:
    exit()
cv2.destroyAllWindows()
