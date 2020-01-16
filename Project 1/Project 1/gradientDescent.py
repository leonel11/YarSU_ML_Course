import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """
    
    J_history = []
    m = y.shape[0]

    for i in range(num_iters):

        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить градиентный спуск для num_iters итераций 
        # с целью вычисления вектора параметров theta, минимизирующего 
        # стоимостную функцию

        s = np.zeros([theta.shape[0], 1])
        for idx in range(0, theta.shape[0]):
            for j in range(0, m):
                s[idx] = s[idx] + (np.dot(np.transpose(theta), X[j]) - y[j]) * X[j,idx]
        theta = theta - alpha/m*s

        # ============================================================
        
        J_history.append(computeCost(X, y, theta)) # сохранение значений стоимостной функции
                                                   # на каждой итерации
    
    return theta, J_history