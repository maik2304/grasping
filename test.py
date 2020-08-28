import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
import matplotlib.pyplot as plt

from utils.data.cornell_data import CornellDataset
from utils.dataset_processing import grasp, image

path = 'Dataset'

dataset = CornellDataset(path)

dataset.get_rgd(0)

g = dataset.get_gtbb(0)
g = g[0]
sample = dataset.__getitem__(0)
img = sample['img']
img = img.transpose((1,2,0))
#plt.imshow(img)

bb = sample['bb']
bb = grasp.Grasp((bb[0],bb[1]),np.arctan2(bb[2],bb[3])/2,bb[4],bb[5])
gr = bb.as_gr

print(gr.center,g.center)
img = image.Image(img)
figure, ax = plt.subplots(nrows=1, ncols=1)
img.show(ax)
gr.plot(ax)
iou = gr.iou(g)

print(iou)
'''
img = image.Image(dataset.get_rgb(0,rot=np.pi/4,normalise=False))
print(img.img[:,:,2].shape)

sample = dataset.__getitem__(0)
bbs = dataset.get_gtbb(0,rot=np.pi/4)
bb = bbs[0]
print(bb.angle*180/np.pi)
depth_img = dataset.get_depth(0,rot=np.pi/4)
print(depth_img.shape)
plt.imshow(depth_img,cmap='gray')

figure, ax = plt.subplots(nrows=1, ncols=1)
img.show(ax)
bb.plot(ax)
'''