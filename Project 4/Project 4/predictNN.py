import numpy as np
from sigmoid import sigmoid

def predictNN(X, Theta1, Theta2):
    """
        Функция позволяет выполнить предсказание метки класса p 
        в диапазоне от 0 до K (число классов равно K + 1) для 
        множества объектов, описанных в матрице объекты-признаки X. 
        Предсказание метки выполняется с использованием матриц 
        обучененных параметров модели Theta1, Theta2 трехслойной 
        нейронной сети
    """
    
    m = X.shape[0]
    p = np.zeros([m, 1])
    
    a1 = X
    a2 = sigmoid(np.dot(a1, Theta1.transpose()))
    a2 = np.concatenate((np.ones((m, 1)), a2), axis = 1)
    
    a3 = sigmoid(np.dot(a2, Theta2.transpose()))
    h  = a3
    
    p = np.argmax(h, axis = 1)
    p = np.array([p]).transpose().astype('uint8')
    
    return p