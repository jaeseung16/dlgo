import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#filename='data/KGS-2015-19-8133-train_features_21.npy'
#X = np.load(filename)
#y = np.load(filename.replace('features', 'labels'))

X = np.load('data/train_features.npy')
y = np.load('data/train_labels.npy')

print("X.shape={}".format(X.shape))
print("y.shape={}".format(y.shape))

index = 17

xToPlot = X[index, ...]
print("xToPlot.shape={}".format(xToPlot.shape))
#yToPlot = to_categorical(y[index].astype(int), 19 * 19)
yToPlot = y[index]
print("y[index]={}, yToPlot.shape={}".format(y[index], yToPlot.shape))

print("xToPlot: feature8={}, feature9={}".format(np.count_nonzero(xToPlot[..., 8]), np.count_nonzero(xToPlot[..., 9])))

plt.subplot(3, 4, 1)
#plt.plot(range(361), xToPlot[...,0].reshape(361,1))
plt.imshow(xToPlot[..., 0], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 2)
#plt.plot(range(361), xToPlot[...,1].reshape(361,1))
plt.imshow(xToPlot[..., 1], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 3)
#plt.plot(range(361), xToPlot[:,:,2].reshape(361,1))
plt.imshow(xToPlot[..., 2], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 4)
#plt.plot(range(361), xToPlot[...,3].reshape(361,1))
plt.imshow(xToPlot[..., 3], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 5)
#plt.plot(range(361), xToPlot[...,4].reshape(361,1))
plt.imshow(xToPlot[..., 4], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 6)
#plt.plot(range(361), xToPlot[...,5].reshape(361,1))
plt.imshow(xToPlot[..., 5], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 7)
#plt.plot(range(361), xToPlot[...,6].reshape(361,1))
plt.imshow(xToPlot[..., 6], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 8)
#plt.plot(range(361), xToPlot[...,7].reshape(361,1))
plt.imshow(xToPlot[..., 7], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 9)
#plt.plot(range(361), xToPlot[...,8].reshape(361,1))
plt.imshow(xToPlot[..., 8], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 10)
#plt.plot(range(361), xToPlot[...,9].reshape(361,1))
plt.imshow(xToPlot[..., 9], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 11)
#plt.plot(range(361), xToPlot[...,10].reshape(361,1))
plt.imshow(xToPlot[..., 10], cmap='viridis', vmin=0, vmax=1)

plt.subplot(3, 4, 12)
plt.bar(range(361), yToPlot)

plt.show()

