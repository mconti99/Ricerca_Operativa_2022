import numpy as np
from sklearn.datasets import make_blobs

n_points = [500,1000,1000,1000,1000,1000,1500,1500,2000,3000,5000,10000]
n_clusters = [5,2,4,5,6,7,5,10,5,5,5,10]
dim_points = [32,32,32,32,20,20,18,18,18,16,16,16]

for i in range(12):
    points, _ = make_blobs(n_samples=n_points[i], centers=n_clusters[i], n_features=dim_points[i], random_state=0)
    s = f'benchmark{i+1}.txt'
    with open(s, 'w') as file:
        for p in points:
            for coord in p:
                file.write(str(coord) + ' ')
            file.write('\n')