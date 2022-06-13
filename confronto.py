import numpy as np
from random import choices, randrange
import random
import matplotlib.pyplot as plt
from numba import njit
from sklearn.datasets import make_blobs
from utility import *
from tqdm import tqdm

@njit
def change_sol(sol, K):
    new_sol = sol.copy()
    n = randrange(len(new_sol))
    old_value = new_sol[n]

    while old_value == new_sol[n]:
        new_sol[n] = randrange(K)

    return new_sol

@njit
def simulated_annealing(base_sol, points, K, iters, alpha, Ti = 10, Tf = 10**-8, verbose = True):
    best_sol = base_sol
    base_val = squared_inner_distance(best_sol, points, K)
    best_value = base_val
    T = Ti
    curr_sol = base_sol
    curr_val = best_value
    
    iter = 1
    old_sol = base_sol
    finito = False
    no_update = 0

    while(finito == False):
        if(verbose and iter%10 == 1):
            print("Iteration number:", iter, "Best value percentuale: ", curr_val/base_val*100, "% T:", T, no_update)
        
        iter = 1 + iter
        old_sol = curr_sol
        
        for i in range(iters):
            candidate = change_sol(curr_sol, K)
            val_candidate = squared_inner_distance(candidate, points, K)

            if(val_candidate < best_value):
                best_value = val_candidate
                best_sol = candidate
            
            if(val_candidate < curr_val):
                curr_val = val_candidate
                curr_sol = candidate
                no_update = 0
            else:
                r = random.uniform(0, 1)
                delta = abs(curr_val - val_candidate)
                tresh = np.exp(-delta/T)
                if(r < tresh):
                    curr_val = val_candidate
                    curr_sol = candidate
                    no_update = 0
                else:
                    no_update += 1

        if(no_update >= 10000 or T < Tf):
            finito = True
            break
        
        T = alpha*T
    return best_sol

@njit
def neighbors_clustering(sol, points, K, q):
    neighbors = []

    q = int(len(points)/50)

    choices = np.arange(len(sol))
    orig_val = squared_inner_distance(sol, points, K)
    centroids = calc_centroids(sol, points, K)

    for i in range(q):
        choice = np.random.randint(len(choices))
        choices = np.delete(choices, choice)
        for k in range(K):
            new_sol = sol.copy()
            if(k != new_sol[choice]):
                new_sol[choice] = k
                new_val = orig_val - np.linalg.norm(points[choice]-centroids[sol[choice]]) + np.linalg.norm(points[choice]-centroids[k])
                neighbors.append((new_sol,new_val))


    choices1 = np.arange(len(sol))
    choices2 = np.arange(len(sol))
    for i in range(q):
        choice1 = np.random.randint(len(choices1))
        choices1 = np.delete(choices1, choice1)
        choice2 = np.random.randint(len(choices2))
        choices2 = np.delete(choices2, choice2)

        new_sol = sol.copy()
        new_sol[choice1], new_sol[choice2] = new_sol[choice2], new_sol[choice1]

        new_val = orig_val - np.linalg.norm(points[choice1]-centroids[sol[choice1]]) - np.linalg.norm(points[choice2]-centroids[sol[choice2]]) + np.linalg.norm(points[choice1]-centroids[sol[choice2]]) + np.linalg.norm(points[choice2]-centroids[sol[choice1]])
        neighbors.append((new_sol,new_val))
        
    return neighbors

@njit
def local_search(base_sol, points, K, q = 100, verbose = True):
    old_sol = base_sol
    base_val = squared_inner_distance(old_sol, points, K)
    iter = 1
    same_sol = 0

    while True:
        neighbourhood = neighbors_clustering(old_sol, points, K, q)
        best_val = squared_inner_distance(old_sol, points , K)
        best_sol = old_sol

        if verbose:
            print("Iteration number:", iter, "Valore percentuale:", best_val/base_val*100, "%")

        for sol,val in neighbourhood:
            if(val < best_val):
                best_val = val
                best_sol = sol
        
        if((best_sol == old_sol).all()):
            same_sol = same_sol + 1
            if(same_sol == 500):
                break
        else:
            same_sol = 0
            old_sol = best_sol

        iter = iter+1
    return old_sol

def k_means(points, K, iters = 10):
    N = points.shape[0]
    dimension = points.shape[1]
    #inizializzazione dei centroidi in corrispondenza di K punti 
    centroids = np.zeros((K,dimension))
    clusters = np.zeros(N)
    best_val = -1
    
    for iter in range(iters):
        choices = np.arange(N)
        for i in range(K):
            choice = np.random.randint(len(choices))
            centroids[i] = points[choice].copy()#centroids_list[i]
            choices = np.delete(choices, choice)
        finito = False
        while(finito == False):
            for i in range(N):
                dist = -1
                centroid = -1
                for c in range(K):
                    dist_c = np.linalg.norm(centroids[c]-points[i])
                    if(dist_c < dist or dist == -1):
                        centroid = c
                        dist = dist_c
                clusters[i] = centroid        
            old_centroids = centroids.copy()

            for i in range(K): #calcolo nuovi centroidi
                centroids[i] = np.zeros(dimension)
                n_elem = 0
                for j in range(N):
                    if clusters[j] == i:
                        n_elem = n_elem+1
                        centroids[i] = centroids[i] + points[j]
                if(n_elem == 0):
                    centroids[i] = points[np.random.randint(len(choices))].copy()
                else:
                    centroids[i] = centroids[i] / n_elem

            if((old_centroids == centroids).all()):#criterio di arresto
                finito = True    
                val = squared_inner_distance(clusters, points, K)
                if(best_val == -1 or best_val > val):
                    best_val = val
                    best_sol = clusters

    return best_sol


from confronto import *
import xlwt
from xlwt import Workbook



wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')

sheet1.write(0, 1, f"N Punti")
sheet1.write(0, 2, "stdev")
sheet1.write(0, 3, "ncluster")
sheet1.write(0, 4, "s.a.")
sheet1.write(0, 5, "k-means")
sheet1.write(0, 6, "local search")



n_points = [100,200,500,500,1000,1000,1500,2000,3000,5000,10000,20000,30000,40000,50000,100,200,500,1000,1500,10000,2000,3000,5000,1000,1000,1000,1000,1000,2000,2000,2000,2000,2000,3000,3000,3000,3000,3000,4000,4000,4000,4000,4000,5000,5000,5000,5000,5000]
n_clusters = [5,10,5,10,5,10,5,5,5,5,4,5,10,5,5,5,5,5,10,5,5,20,30,50,10,10,10,10,10,15,15,15,15,15,20,20,20,20,20,5,5,5,5,5,10,10,10,10,10]

for i in tqdm(range(1,50)):
    sheet1.write(i, 0, f"Test {i}")
    N = n_points[i-1]
    K = n_clusters[i-1]
    sheet1.write(i, 1, N)
    sheet1.write(i, 3, K)
    points = load_points(f'benchmark{i}.txt')
    for j in range(3):
        if(j == 0):
            sol = create_initial_sol(points,K)
            sol = simulated_annealing(sol, points, K, 100, 0.99, 100, 0.001, verbose = False)
        if(j == 1):
            sol = k_means(points,K)
        if(j == 2):
            sol = np.random.randint(K, size = N)
            sol = local_search(sol, points, K, q = 300,verbose = False)
        val = squared_inner_distance(sol, points, K)
        sheet1.write(i, j+4, val)
    wb.save('res_auto.xls')