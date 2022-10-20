import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
def compute_min_distance(x, centroids):
    K = centroids.shape[0]
    distances = np.zeros(K)
    for k in range(K):
        distances[k] = np.square(np.linalg.norm(x - centroids[k]))
    return np.argmin(distances)

#%%
data = pd.read_csv('wine.data', header=None)
data = data.sample(frac=1).reset_index(drop=True)
y = data[0]
x1 = data.loc[:, 1].to_numpy().reshape(178, 1)
x2 = data.loc[:, 3].to_numpy().reshape(178, 1)
x = np.concatenate([x1, x2], axis=1)
m = x.shape[0]
#%% initialize centroids
# data set has been shuffled
K = 3
centroids = np.zeros(shape=(K, x.shape[1]))
for k in range(K):
    centroids[k] = x[k]
#%%
c = np.zeros(x.shape[0])
for repeat in range(100):
    for i in range(m):
        c[i] = compute_min_distance(x[i], centroids)
    for k in range(K):
        centroids[k] = np.mean(x[c == k, :], axis=0)
#%% compute J
distance = np.zeros(x.shape[0])
for k in range(K):
    distance[k] = np.square(np.linalg.norm(x[c == k] - centroids[k]))
J = np.sum(distance)
#%%
plt.figure()
f1 = 0
f2 = 1
plt.plot(x[:, 0], x[:, 1], 'o', color='lightblue')
for k in range(K):
    plt.plot(centroids[k][f1], centroids[k][f2], 'o')
# col = ['red', 'blue', 'green', 'black', 'yellow', 'purple', 'orange']
# co = 0
# for k in range(K):
#     for i in range(178):
#         if c[i] == k:
#             plt.plot(x[i, 0], x[i, 1], 'o', color=col[co])
#     co += 1

plt.show()
#%%
c = c+1
accuracy = np.sum(c == y) / len(c)
print(f'accuracy= {accuracy*100} %')
print('error= ', J)
