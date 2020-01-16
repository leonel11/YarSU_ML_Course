import numpy as np
from sigmoid import sigmoid

def computeCost(X, y, theta):
    """
        Функция позволяет вычислить значение стоимостной функции, 
        используя theta в качестве параметров для логистической 
        регрессии, матрицу объекты-признаки X и вектор меток y
    """
    
    m = y.shape[0]
    J = 0

    # ====================== Ваш код здесь ======================
    # Инструкция: вычислить значение стоимостной функции для заданных 
    # theta, X и y. Присвоить полученный результат в J для стоимостной 
    # функции

    hqxs = []
    count = 0
    for i in range(0, m):
        hqxs.append(sigmoid(np.dot(np.transpose(theta), X[i])))
    for i in range(0, m):
        count = count + (-1.0/m) * (y[i]*np.log(hqxs[i]) + (1.0 - y[i])*np.log(1.0 - hqxs[i]))
    J = count[0]

    # ============================================================
    
    return J