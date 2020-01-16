import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    """
        Функция позволяет выполнить визуализацию данных c маркером
        + для положительных примеров и маркером o для отрицательных 
        примеров. X - матрица объекты-признаки размера mx2, 
        а y - вектор меток размера mx1, где m - размер базы данных
    """
    
    # ====================== Ваш код здесь ======================
    # Инструкция: визуализируйте положительные и отрицатедьные 
    # примеры на двумерной плоскости, используя маркер + для 
    # обозначения положительных примеров и маркер o для обозначения 
    # отрицательных примеров
    
    plt.figure()
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.plot(X[pos, 0], X[pos, 1], '+', markersize = 6, markeredgecolor = 'black', markeredgewidth = 2)
    plt.plot(X[neg, 0], X[neg, 1], 'o', markersize = 6, markeredgecolor = 'black', markerfacecolor = 'yellow')
    plt.grid()
    
    # ============================================================