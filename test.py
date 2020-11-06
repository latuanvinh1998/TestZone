import torch
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
import os
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm

transform = transforms.Compose([transforms.Resize(112), transforms.CenterCrop(112), transforms.ToTensor()])

dataset = datasets.ImageFolder('../data/', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

images, labels = next(iter(dataloader))
#data = iter(dataloader)
# for image,label in tqdm(iter(dataloader), desc="BKB For Weak"):
# 	#print(image.shape)
# 	pass
# plt.imshow(images[0].permute(1, 2, 0))
# plt.show()
model = MobileFaceNet(512).to(torch.device("cuda:0"))
arc = Arcface(embedding_size=512, classnum=15).to(torch.device("cuda:0"))
model.eval()
with torch.no_grad():
	images = images.to(torch.device("cuda:0"))
	labels = labels.to(torch.device("cuda:0"))
	emb = model.forward(images)
prob = arc(emb, labels)
loss = CrossEntropyLoss()(prob, labels)
print(loss)