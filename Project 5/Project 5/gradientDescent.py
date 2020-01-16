import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters, lam):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели theta, используя матрицу объекты-признаки X, 
        вектор меток y, параметр сходимости alpha и число итераций 
        алгоритма num_iters
    """
    
    J_history = []
    m = y.shape[0]

    J_history.append(computeCost(X, y, theta, lam))

    for i in range(num_iters):
        
        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить градиентный спуск для num_iters итераций 
        # с целью вычисления вектора параметров theta, минимизирующего 
        # стоимостную функцию
        
        # формулы из лекции № 4 (слайд №16), см. практика №1 с добавлением дополнительного слагаемого

        sums = X.transpose().dot((np.dot(X, theta) - y)) # скалярное произведение одномерных массивов,
                                                         # а значит суммы уже подсчитаны
        th = theta[0] - alpha / m * sums[0]
        for idx in range(theta.shape[0]):
            theta[idx] = theta[idx] * (1.0 - alpha * lam / m) - alpha / m * sums[idx]
        theta[0] = th

        # ============================================================
        
        J_history.append(computeCost(X, y, theta, lam)) # сохранение значений стоимостной функции
                                                        # на каждой итерации
    
    return theta, J_history