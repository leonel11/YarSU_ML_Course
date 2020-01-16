import numpy as np

def sigmoid(z):
    """
        Функция вычисляет значение сигмоидной функции для z
    """
    
    g = 1 / (1 + np.exp(-z))
    
    return g