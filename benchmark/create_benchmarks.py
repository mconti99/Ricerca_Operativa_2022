import numpy as np
from sklearn.datasets import make_blobs

n_points = [500,1000,2000,5000,1400,10000,4000,7000,300,800]
n_clusters = [7,4,3,2,3,2,4,5,4,6,4]
dim_points = [32,3,2,2,3,2,3,2,64,32]

for i in range(10):
    points, _ = make_blobs(n_samples=n_points[i], centers=n_clusters[i], n_features=dim_points[i], random_state=0)
    s = f'./benchmark{i+1}.txt'
    with open(s, 'w') as file:
        for p in points:
            for coord in p:
                file.write(str(coord) + ' ')
            file.write('\n')