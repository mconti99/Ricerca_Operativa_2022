import numpy as np
from sklearn.datasets import make_blobs

n_points = [500,100,200,500,1400,1000,400,700,300,800,5000,10000]
n_clusters = [7,4,3,2,3,2,4,5,4,6,4,5,10]
dim_points = [32,16,24,10,20,16,12,20,64,32,3,3]

for i in range(12):
    points, _ = make_blobs(n_samples=n_points[i], centers=n_clusters[i], n_features=dim_points[i], random_state=0)
    s = f'benchmark{i+1}.txt'
    with open(s, 'w') as file:
        for p in points:
            for coord in p:
                file.write(str(coord) + ' ')
            file.write('\n')