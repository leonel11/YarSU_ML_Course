## Практическое задание № 3. Многоклассовая классификация и нейронные сети ("один против всех" логистическая регрессия)

# Инициализация
import numpy as np
import scipy.io as spi
import matplotlib.pyplot as plt

from displayData import displayData
from oneVsAll import oneVsAll
from sigmoid import sigmoid
from predictOneVsAll import predictOneVsAll

# ================= Часть 1. Визуализация данных =================

print('Часть 1. Визуализация данных')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = spi.loadmat('data.mat')

X = data['X']
y = data['y']

m = X.shape[0]

# Визуализация данных
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel, 10)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# = Часть 2. Обучение "один против всех" логистической регрессии =

print('Часть 2. Обучение "один против всех" логистической регрессии')

# Нормализация свойств и добавление единичного признака
m, n = X.shape
X = np.concatenate((np.ones((m, 1)), X), axis = 1)

# Задание общего числа классов (меток)
num_labels = 10

# Задание начальных параметров модели
initial_theta = np.zeros([n + 1, num_labels])

# Задание параметров градиентного спуска
iterations = 1500
alpha = 1

# Визуализация процесса сходимости для i-го классифкатора
flag = False

# Выполнение процедуры обучения параметров модели
all_theta = oneVsAll(X, y, num_labels, initial_theta, alpha, iterations, flag)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# == Часть 3. Вычисление доли правильных ответов классификатора ==

print('Часть 3. Вычисление доли правильных ответов классификатора')

# Вычисление доли правильных ответов классификатора
p = predictOneVsAll(X, all_theta)
acc = np.sum((p == y).astype('float64')) / len(y) * 100
print('Доля правильных ответов обученного классификатора = {:.4f}'.format(acc))