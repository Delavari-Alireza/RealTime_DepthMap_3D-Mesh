import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
import argparse



parser = argparse.ArgumentParser(description='Image to Depth')


parser.add_argument('--img-path', type=str , default='/home/alireza/AlirezaFiles/DepthAnything V2/report/report/kitchen.png')

args = parser.parse_args()


filename = args.img_path


pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf" , device_map = 'cuda')

image = Image.open(filename)

width, height = image.size
depth = pipe(image)["depth"]


depth = np.array(depth)


cv2.imshow('depth' , depth)

cv2.imshow('raw image' , cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB))

k = cv2.waitKey(0) & 0xff
if k ==27:
    exit()
cv2.destroyAllWindows()
