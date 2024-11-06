import numpy as np
import os
import re
import sys 
path = './checkpoints/prostateroi/testlog3.txt'
lines = open(path,'r').readlines()[:-2]
values = []
for line in lines:
    dice = line.split(':')[-1].split(',')[0]
    dice = float(dice)
    dist = re.search(r"Center distance\s([0-9.]+)", line)
    dist = float(dist.group(1)) 
    values.append([dice, dist])

print(np.mean(np.stack(values, axis=0), axis=0))
print(np.std(np.stack(values, axis=0), axis=0))
