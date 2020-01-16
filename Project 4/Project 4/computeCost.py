import numpy as np
from sigmoid import sigmoid

def computeCost(X, y, num_labels, Theta1, Theta2, lam):
    """
        Функция позволяет вычислить значение стоимостной функции, 
        используя матрицы Theta1 и Theta2 в качестве параметров 
        для нейронной сети, параметр регуляризации lam, матрицу 
        объекты-признаки X и вектор меток y
    """
    
    m = y.shape[0]
    J = 0

    # ====================== Ваш код здесь ======================
    # Инструкция: вычислить значение стоимостной функции для заданных 
    # Theta1, Theta2, lam, X и y. Присвоить полученный результат в J 
    # для стоимостной функции

    a1 = X
    a2 = sigmoid(np.dot(a1, Theta1.transpose()))
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1)
    a3 = sigmoid(np.dot(a2, Theta2.transpose()))
    Hqx = a3

    Y = np.zeros((m, num_labels))
    for c in range(num_labels):
        Y[np.where(y == c)[0], c] = 1

    for i in range(m):
        for k in range(num_labels):
            J = J - 1.0/m * (Y[i, k]*np.log(Hqx[i, k]) + (1.0 - Y[i, k])*np.log(1.0 - Hqx[i, k]))

    J = J + lam / (2.0 * m) * (np.sum(Theta1 ** 2) + np.sum(Theta2 ** 2) - np.sum(Theta1[:, 0] ** 2) - np.sum(Theta2[:, 0] ** 2))

    # ============================================================
    
    return J