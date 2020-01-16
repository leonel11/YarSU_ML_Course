import numpy as np
from computeCost import computeCost
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def gradientDescent(X, y, num_labels, Theta1, Theta2, alpha, num_iters, lam):
    """
        Функция позволяет выполнить градиентный спуск для поиска 
        параметров модели Theta1 и Theta2, используя матрицу 
        объекты-признаки X, вектор меток y, число классов num_labels, 
        параметр сходимости alpha, число итераций алгоритма num_iters 
        и параметр регуляризации lam
    """
    
    J_history = []
    m = y.shape[0]

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    Y = np.zeros((m, num_labels))
    for c in range(num_labels):
        Y[np.where(y == c)[0], c] = 1
    
    for i in range(num_iters):

        print('Эпоха обучения №', i + 1)
        
        # ====================== Ваш код здесь ======================
        # Инструкция: выполнить алгоритм обратного распространения ошибки 
        # с целью поиска частных производных от стоимостной функции по 
        # параметрам модели
        
        D1 = np.zeros(Theta1.shape)
        D2 = np.zeros(Theta2.shape)

        for i in range(m):
            a1 = X[i:i+1, :]
            a2 = sigmoid(np.dot(a1, Theta1.transpose()))
            a2 = np.concatenate((np.ones((1, 1)), a2), axis=1)
            a3 = sigmoid(np.dot(a2, Theta2.transpose()))
            h = a3
            delta3 = (h - Y[i:i+1]).transpose()
            delta2 = np.dot(Theta2.transpose(), delta3)
            delta2 = delta2[1:, :]*sigmoidGradient((np.dot(a1, Theta1.transpose())).transpose())
            D1 = D1 + np.dot(delta2, a1)
            D2 = D2 + np.dot(delta3, a2)

        Temp1 = np.copy(Theta1)
        Temp1[:, 0] = 0
        Temp2 = np.copy(Theta2)
        Temp2[:, 0] = 0
        Theta1_grad = D1 / m + Temp1 * lam / m
        Theta2_grad = D2 / m + Temp2 * lam / m
        
        # ============================================================
        
        Theta1 = Theta1 - alpha * Theta1_grad
        Theta2 = Theta2 - alpha * Theta2_grad
        
        J_history.append(computeCost(X, y, num_labels, Theta1, Theta2, lam)) # сохранение значений стоимостной функции
                                                                             # на каждой итерации
    
    return Theta1, Theta2, J_history