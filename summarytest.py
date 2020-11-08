import torch
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter('scalar/tutorial')

for n_iter in range(100):
    writer.add_scalar('Test', np.random.random(), n_iter)
    # writer.add_scalar('Summary/Loss/test', np.random.random(), n_iter)
    # writer.add_scalar('Summary/Accuracy/train', np.random.random(), n_iter)
    # writer.add_scalar('Summary/Accuracy/test', np.random.random(), n_iter)
writer.close()
