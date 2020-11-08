import os
import numpy as np
# for i in range(10):
for i in range(10):
	f = open('test.txt', 'r')
	step = f.readline()
	f = open('test.txt', 'w')
	f.write(str(int(step)+10))
