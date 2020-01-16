import numpy as np
from computeCost import computeCost
from sigmoid import sigmoid

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
        
        theta = theta - (alpha / m) * X.transpose().dot((sigmoid(np.dot(X, theta)) - y))
        
        J_history.append(computeCost(X, y, theta)) # сохранение значений стоимостной функции
                                                   # на каждой итерации
    
    return theta, J_history