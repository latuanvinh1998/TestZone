from evaluate import *
import torch
from torchvision import datasets, transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

imgs = model_evaluate("pairs.txt", "../lfw")
img_batch = torch.utils.data.DataLoader(imgs, batch_size=32)
for batch in iter(img_batch):
	print(batch.shape)