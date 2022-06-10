import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint

from numba import jit, njit

def load_points(file):
    f = open(file, 'r')
    points = []

    for i,line in enumerate(f.readlines()):
        nums = line.split(" ")
        coords = []
        for num in nums:
            if num != '' and num != '\n':
                coords.append(float(num))
        if len(coords)!=0:
            coords = np.array(coords)
            points.append(coords)

    points = np.array(points)
    f.close()
    return points

@jit(nopython=True)
def calc_centroids(sol, points, K):
    dim = np.shape(points)[1]
    centroids = np.zeros((K,dim))
    num_elems = np.zeros(K)
    for elem,cluster in enumerate(sol):
        cluster = int(cluster)
        elem = int(elem)
        centroids[cluster] = centroids[cluster] + points[elem]
        num_elems[cluster] = num_elems[cluster] + 1

    for i in range(K):
        centroids[i] = centroids[i] / num_elems[i]
    
    return centroids

@jit(nopython=True)
def squared_inner_distance(sol, points, K):
    dim = np.shape(points)[1]
    centroids = np.zeros((K,dim))
    num_elems = np.zeros(K)
    for elem,cluster in enumerate(sol):
        cluster = int(cluster)
        elem = int(elem)
        centroids[cluster] = centroids[cluster] + points[elem]
        num_elems[cluster] = num_elems[cluster] + 1

    for i in range(K):
        centroids[i] = centroids[i] / num_elems[i]
    
    tot_distance = 0

    for elem,cluster in enumerate(sol):
        cluster = int(cluster)
        elem = int(elem)
        tot_distance = np.sum((centroids[cluster]-points[elem])**2) + tot_distance

    return tot_distance

def create_initial_sol(points, K):
    N = points.shape[0]
    dimension = points.shape[1]
    centroids = np.zeros((K,dimension))
    clusters = np.zeros(N)
    choices = np.arange(N)

    for i in range(K):
        choice = np.random.randint(len(choices))
        centroids[i] = points[choice].copy()#centroids_list[i]
        choices = np.delete(choices, choice)
            
   
        for i in range(N):
            dist = -1
            centroid = -1
            for c in range(K):
                dist_c = np.linalg.norm(centroids[c]-points[i])
                if(dist_c < dist or dist == -1):
                    centroid = c
                    dist = dist_c
            clusters[i] = int(centroid)

    return clusters


def printR2sol(points, sol, K):
    N = len(sol)
    colors = []
    for i in range(K):
        colors.append('#%06X' % randint(0, 0xFFFFFF))


    clusters = []

    for i in range(K):
        cluster = []
        for j in range(N):
            if(sol[j]==i):
                cluster.append(points[j].copy())
        cluster = np.array(cluster)
        clusters.append(cluster)


    for i in range(K):
        to_draw = clusters[i]
        plt.scatter(to_draw[:,0], to_draw[:,1], color = colors[i])
    plt.show()
