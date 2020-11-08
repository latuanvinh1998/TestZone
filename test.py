import torch
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

# transform = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(112), transforms.ToTensor()])

# dataset = datasets.ImageFolder('../data/', transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# images, labels = next(iter(dataloader))
# #data = iter(dataloader)
# # for image,label in tqdm(iter(dataloader), desc="BKB For Weak"):
# # 	#print(image.shape)
# # 	pass
# # plt.imshow(images[0].permute(1, 2, 0))
# # plt.show()
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# img = cv2.imread("test.jpg")
# img=cv2.resize(img,(112,112))
# img=transform(img)
# img = img.type(torch.FloatTensor)
# imgs = []
# for i in range(100):
# 	imgs.append(img)
# img_batch = torch.utils.data.DataLoader(imgs, batch_size=32)

# batch = next(iter(img_batch))
# print(batch.shape)
# print(len(imgs))

# model = MobileFaceNet(512).to(torch.device("cuda:0"))
# arc = Arcface(embedding_size=512, classnum=15).to(torch.device("cuda:0"))

# imgs = model_evaluate(model, "pairs.txt", "../lfw")
# model.eval()

# with torch.no_grad():
# 	batch = batch.to(torch.device("cuda:0"))
# 	#labels = labels.to(torch.device("cuda:0"))
# 	emb = model.forward(batch)
# print(emb.shape)
# prob = arc(emb, labels)
# loss = CrossEntropyLoss()(prob, labels)
# print(loss)
