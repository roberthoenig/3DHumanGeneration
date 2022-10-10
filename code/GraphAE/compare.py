import numpy as np
import os
import open3d as o3d
import sklearn as skl
import tqdm


train_data_path = "../../data/DFAUST/train.npy"
test_data_path = "../../train/0422_graphAE_dfaust/test_30/epoch198/ply"

train_data = np.load(train_data_path)
test_data = np.zeros((999, 6890, 3))
print(train_data.shape)

for i, filename in enumerate(os.listdir(test_data_path)):
   with open(os.path.join(test_data_path, filename), 'r') as f:
       test_o3d = o3d.io.read_point_cloud(f.name)
       test_np = np.asarray(test_o3d.points)
       test_data[i, :, :] = test_np

distance_matrix = np.zeros((1, 32933))
pbar = tqdm.tqdm(total=1)
for i in range(1):
    pbar2 = tqdm.tqdm(total=32933)
    for j in range(train_data.shape[0]):
        distance_matrix[i, j] = skl.metrics.pairwise.euclidean_distances(train_data[j, :, :], train_data[i, :, :]).mean()
        pbar2.update()
    pbar.update()
    pbar2.close()
pbar.close()

np.save("distance_train.npy", distance_matrix)

