import os
import numpy as np
# for i in range(10):


f = open('step.txt', 'r')
step = f.readline()
epoch = f.readline()
print(step)
print(epoch)
f = open('step.txt', 'w')
f.write('0' + '\n' + '5')
