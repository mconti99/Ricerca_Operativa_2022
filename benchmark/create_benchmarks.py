import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

n_points = [100,200,500,500,1000,1000,1500,2000,3000,5000,10000,20000,30000,40000,50000,100,200,500,1000,1500,10000,2000,3000,5000]
n_clusters = [5,10,5,10,5,10,5,5,10,20,4,5,10,10,20,20,50, 5,5,5,10,5,5,20,30,50]
dim_points = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 16,24,10,32,16,3, 2,2,2]

N = len(n_points)

for i in range(N):
    points, _ = make_blobs(n_samples=n_points[i], centers=n_clusters[i], n_features=dim_points[i], random_state=i, cluster_std = 0.8)
    #plt.scatter(points[:,0], points[:,1])
    #plt.show()
    s = f'benchmark{i+1}.txt'
    with open(s, 'w') as file:
        for p in points:
            for coord in p:
                file.write(str(coord) + ' ')
            file.write('\n')