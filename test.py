import torch
from torch import optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import pandas as pd
from torch.nn import CrossEntropyLoss
import os
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm
import cv2
import numpy as np

import copy
 
from datetime import datetime
from evaluate import *

# transform = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(112), 
# 	transforms.RandomHorizontalFlip(),
# 	transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# dataset = datasets.ImageFolder('../imgs/', transform=transform)
# class_num = dataset[-1][1] + 1
# print(class_num)

model = MobileFaceNet(512).to(torch.device("cuda:0"))
arc = Arcface(embedding_size=512, classnum=85742).to(torch.device("cuda:0"))

img_lfw, y_true, nrof_images = load_lfw("pairs.txt", "../lfw")
model_evaluate(model, img_lfw, y_true, nrof_images)
model.train()
torch.save(model.state_dict(), 'model%2f.pth'.format(5))
#model.load_state_dict(torch.load('model.pth'))
