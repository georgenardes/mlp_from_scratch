import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data1 = np.load('layer0_grad_hist.npy')
x1 = data1[:, 0]
y1 = data1[:, 1]
z1 = data1[:, 2]

data2 = np.load('layer2_grad_hist.npy')
x2 = data2[:, 0]
y2 = data2[:, 1]
z2 = data2[:, 2]

data3 = np.load('layer4_grad_hist.npy')
x3 = data3[:, 0]
y3 = data3[:, 1]
z3 = data3[:, 2]

data4 = np.load('layer25_grad_hist.npy')
x4 = data4[:, 0]
y4 = data4[:, 1]
z4 = data4[:, 2]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})

axes[0].scatter(x1, y1, z1)
axes[1].scatter(x2, y2, z2)
axes[2].scatter(x3, y3, z3)
axes[3].scatter(x4, y4, z4)

plt.show()