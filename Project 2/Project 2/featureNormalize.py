import numpy as np
from numpy.matlib import repmat

def featureNormalize(X):
    """
        Функция позволяет вычислить нормализованную версию матрицы 
        объекты-признаки X со средним значением для каждого признака 
        равным 0 и среднеквадратическим отклонением равным 1
    """
    
    X_norm = X
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0, ddof = 1)
    X_norm = np.divide(X - repmat(mu, X.shape[0], 1), repmat(sigma, X.shape[0], 1))
    
    return X_norm, mu, sigma