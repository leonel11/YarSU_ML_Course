import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from numpy.matlib import repmat

def plotDecisionBoundary(X, y, theta, mu, sigma):
    """
        Функция позволяет выполнить визуализацию данных c маркером
        + для положительных примеров и маркером o для отрицательных 
        примеров, с границей решения для заданного множества 
        параметров модели theta, матрицы объекты-признаки X и вектора 
        меток y
    """

    n = X.shape[1]
    X = repmat(sigma, X.shape[0], 1) * X[:, 1:n] + repmat(mu, X.shape[0], 1)
    plotData(X, y)

    Theta = np.zeros(theta.shape)

    Theta[1, 0] = theta[1, 0] / sigma[0]
    Theta[2, 0] = theta[2, 0] / sigma[1]
    Theta[0, 0] = theta[0, 0] - mu[0] * Theta[1, 0] - mu[1] * Theta[2, 0]

    plot_x = np.array([min(X[:, 1]), max(X[:, 1])])
    plot_y = (-1./Theta[2, 0]) * (Theta[1, 0] * plot_x + Theta[0, 0])
    plt.plot(plot_x, plot_y)