from computeCentroids import computeCentroids
from findClosestCentroids import findClosestCentroids
import matplotlib.pyplot as plt
import numpy as np

def runkMeans(X, initial_centroids, max_iters, plot_progress = False):
    """ 
        Функция выполняет алгоритм K-средних для матрицы объекты-признаки X. 
        Формальный параметр initial_centroids определяет начальное 
        расположение средних, max_iters определяет число итераций алгоритма, 
        а plot_progress отвечает за визуализацию процесса сходимости в ходе 
        обучения
    """
    
    K = initial_centroids.shape[0]
    centroids = initial_centroids 
    previous_centroids = centroids

    if plot_progress:
            plt.plot(X[:, 0], X[:, 1], 'bo')

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids) # первый шаг алгоритма K-средних
        
        if plot_progress:

            for j in range(K):
                plt.plot([centroids[j, 0], previous_centroids[j, 0]], [centroids[j, 1], previous_centroids[j, 1]], 'r-x')

        previous_centroids = centroids # второй шаг алгоритма K-средних
        centroids = computeCentroids(X, idx, K) # третий шаг алгоритма K-средних

    if plot_progress:
        plt.xlabel('Первый признак')
        plt.ylabel('Второй признак')
        plt.grid()
        plt.show()

    return (centroids, idx)