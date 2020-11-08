import torch
from torch import optim
from torchvision import datasets, transforms
import pandas as pd
from torch.nn import CrossEntropyLoss
import os
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm
import cv2
import numpy as np
from evaluate import *

###### PREPARE DATA ######
transform = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(112), 
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

dataset = datasets.ImageFolder('../data/', transform=transform)
class_num = dataset[-1][1] + 1
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

###### LOAD LFW IMG ######
img_lfw, y_true, nrof_images = load_lfw("pairs.txt", "../lfw")


model = MobileFaceNet(512).to(torch.device("cuda:0"))
arc = Arcface(embedding_size=512, classnum=class_num).to(torch.device("cuda:0"))

paras_only_bn, paras_wo_bn = separate_bn_paras(model)
optimizer = optim.SGD([
	            {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
	            {'params': [paras_wo_bn[-1]] + [arc.kernel], 'weight_decay': 4e-4},
	            {'params': paras_only_bn}
	        ], lr = 1e-3, momentum = 0.9)

model_evaluate(model, img_lfw, y_true, nrof_images)

model.train()

epoch_step = 0
for img, label in iter(dataloader):
	img = img.to(torch.device("cuda:0"))
	label = label.to(torch.device("cuda:0"))
	optimizer.zero_grad()
	embedding = model(img)
	theta = arc(embedding, label)
	loss = CrossEntropyLoss()(theta, label)
	if epoch_step%50 == 0:
		print("Loss: %.3f" % (loss))
	loss.backward()
	optimizer.step()

model_evaluate(model, img_lfw, y_true, nrof_images)
