import numpy as np
from sigmoid import sigmoid

def computeCost(X, y, theta):
    """
        Функция позволяет вычислить значение стоимостной функции, 
        используя theta в качестве параметров для логистической 
        регрессии, матрицу объекты-признаки X и вектор меток y
    """
    
    m = y.shape[0]
    
    J = (- np.sum(y * np.log(sigmoid(np.dot(X, theta)))) - np.sum((1 - y) * np.log(1 - sigmoid(np.dot(X, theta))))) / m
    
    return J