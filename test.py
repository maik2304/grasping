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

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.visualisation.gridshow import gridshow
from utils.dataset_processing import grasp, image

path = 'Dataset'
dataset = 'cornell'

Dataset = get_dataset(dataset)
dataset = Dataset(path,
                  ds_rotate=False,
                  random_rotate=True,
                  random_zoom=True,
                  include_depth=1,
                  include_rgb=1)


depth = dataset.get_depth(0)


plt.imshow(depth,cmap='gray')

'''
img = dataset.get_rgb(882,normalise=False)
depth = dataset.get_depth(882)

plt.imshow(depth)
print(depth.shape)
'''