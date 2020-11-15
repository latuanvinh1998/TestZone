import torch
from model import *
import numpy as np
from datetime import datetime
from evalsource import *



model = MobileFaceNet(512).to(torch.device("cuda:0"))
model.load_state_dict(torch.load('model_accuracy:0.6235.pth'))

lfw, lfw_issame = get_val_pair('data', 'lfw')
acc, std, best = evaluate(model=model, carray=lfw, issame=lfw_issame)
acc = np.round(acc*100, 3)
std = np.round(std*100, 3)

print("accuracy: {}+-{}".format(acc, std))

