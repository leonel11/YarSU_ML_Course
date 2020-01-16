## Практическое задание № 1. Регуляризованная линейная регрессия. Недообучение и переобучение

# Инициализация
import numpy as np
import scipy.io as spi
import matplotlib.pyplot as plt
from numpy.matlib import repmat

from computeCost import computeCost
from featureNormalize import featureNormalize
from gradientDescent import gradientDescent
from learningCurve import learningCurve
from polyFeatures import polyFeatures

# ================= Часть 1. Визуализация данных =================

print('Часть 1. Визуализация данных')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = spi.loadmat('data.mat')

# Формирование обучающих данных
X = data['X']
y = data['y']

m = X.shape[0]

# Визуализация данных
plt.figure()
plt.plot(X, y, 'rx', markersize = 5, label = 'Тренировочные данные')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Изменение уровня воды в реке')
plt.ylabel('Количество воды, сбрасываемое дамбой')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ======== Часть 2. Стоимостная функция с регуляризацией =========

print('Часть 2. Стоимостная функция с регуляризацией')

# Задание параметров модели
theta = np.ones([2, 1])

# Задание параметра регуляризации
lam = 1

# Нормализация и добавление единичного признака
X_norm, mu, sigma = featureNormalize(X)
X_norm = np.concatenate((np.ones((m, 1)), X_norm), axis = 1)

# Вычисление значения стоимостной функции для начального theta
J = computeCost(X_norm, y, theta, lam)
print('Значение стоимостной функции: {:.4f}'.format(J))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ==== Часть 3. Обучение регуляризованной линейной регрессии =====

print('Часть 3. Обучение регуляризованной линейной регрессии')

# Задание начальных значений параметров модели
theta = np.zeros([2, 1])

# Задание параметров градиентного спуска
iterations = 1500
alpha = 0.05

# Задание параметра регуляризации
lam = 0

# Выполнение градиентного спуска
theta, J_history = gradientDescent(X_norm, y, theta, alpha, iterations, lam)
print('Найденные параметры модели: {:.4f} {:.4f}'.format(theta[0, 0], theta[1, 0]))

# Визуализация аппроксимации для линейной регрессии
plt.figure()
plt.plot(X, y, 'rx', markersize = 5, label = 'Тренировочные данные')
plt.plot(X, np.dot(X_norm, theta), 'b-', label = 'Регуляризованная линейная регрессия')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Изменение уровня воды в реке')
plt.ylabel('Количество воды, сбрасываемое дамбой')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ===== Часть 4. Кривые обучения для рег. линейной регрессии =====

print('Часть 4. Кривые обучения для регуляризованной линейной регрессии')

# Формирование проверочных данных
X_val = data['Xval']
y_val = data['yval']

m_val = X_val.shape[0]

# Нормализация и добавление единичного признака
X_val_norm = np.divide(X_val - repmat(mu, X_val.shape[0], 1), repmat(sigma, X_val.shape[0], 1))
X_val_norm = np.concatenate((np.ones((m_val, 1)), X_val_norm), axis = 1)

# Вычисление ошибок на обучающем и проверочном множествах
error_train, error_val = learningCurve(X_norm, y, X_val_norm, y_val, alpha, iterations, lam)

# Визуализация кривых обучения
plt.figure()
plt.plot(range(1, m+1), error_train, 'r-', label = 'Ошибка обучения')
plt.plot(range(1, m+1), error_val, 'b-', label = 'Ошибка проверки')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Число тренировочных примеров')
plt.ylabel('Ошибка')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ==== Часть 5. Создание свойств для полиномиальной регрессии ====

print('Часть 5. Создание свойств для полиномиальной регрессии')

# Задание степени полинома
p = 8

X_poly = polyFeatures(X, p)
X_norm_poly, mu_poly, sigma_poly = featureNormalize(X_poly)
X_norm_poly = np.concatenate((np.ones((m, 1)), X_norm_poly), axis = 1)

X_val_poly = polyFeatures(X_val, p)
X_val_norm_poly = np.divide(X_val_poly - repmat(mu_poly, X_val_poly.shape[0], 1), repmat(sigma_poly, X_val_poly.shape[0], 1))
X_val_norm_poly = np.concatenate((np.ones((m_val, 1)), X_val_norm_poly), axis = 1)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ========= Часть 6. Обучение полиномиальной регрессии ===========

print('Часть 6. Обучение полиномиальной регрессии')

# Задание начальных значений параметров модели
theta = np.zeros([p + 1, 1])

# Задание параметров градиентного спуска
iterations = 1500
alpha = 0.05

# Задание параметра регуляризации
lam = 3

# Выполнение градиентного спуска
theta, J_history = gradientDescent(X_norm_poly, y, theta, alpha, iterations, lam)

# Визуализация аппроксимации для полиномиальной регрессии
plt.figure()
plt.plot(X, y, 'rx', markersize = 5, label = 'Тренировочные данные')

x = np.array([np.arange(np.min(X) - 15, np.max(X) + 25, 0.05)]).transpose()
x_poly = polyFeatures(x, p)
x_norm_poly = np.divide(x_poly - repmat(mu_poly, x_poly.shape[0], 1), repmat(sigma_poly, x_poly.shape[0], 1))
x_norm_poly = np.concatenate((np.ones((x.shape[0], 1)), x_norm_poly), axis = 1)

plt.plot(x, np.dot(x_norm_poly, theta), 'b-', label = 'Полиномиальная регрессия')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Изменение уровня воды в реке')
plt.ylabel('Количество воды, сбрасываемое дамбой')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ==== Часть 7. Кривые обучения для полиномиальной регрессии =====

print('Часть 7. Кривые обучения для полиномиальной регрессии')

# Вычисление ошибок на обучающем и проверочном множествах
error_train, error_val = learningCurve(X_norm_poly, y, X_val_norm_poly, y_val, alpha, iterations, lam)

# Визуализация кривых обучения
plt.figure()
plt.plot(range(1, m+1), error_train, 'r-', label = 'Ошибка обучения')
plt.plot(range(1, m+1), error_val, 'b-', label = 'Ошибка проверки')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Число тренировочных примеров')
plt.ylabel('Ошибка')
plt.grid()
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ============ Часть 8. Выбор параметра регуляризации ============

print('Часть 8. Выбор параметра регуляризации')

lam = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

error_train = np.zeros(lam.shape)
error_val = np.zeros(lam.shape)

# Задание параметров градиентного спуска
iterations = 1500
alpha = 0.05

j = 0
for i in lam:
    # Выполнение градиентного спуска
	theta, J_history = gradientDescent(X_norm_poly, y, theta, alpha, iterations, i)
	
    # Вычисление ошибок на обучающем и проверочном множествах
	error_train[j] = computeCost(X_norm_poly, y, theta, 0)
	error_val[j] = computeCost(X_val_norm_poly, y_val, theta, 0)
	
	j = j + 1

# Визуализация для выбора параметра регуляризации
plt.figure()
plt.plot(lam, error_train, 'r-', label = 'Ошибка обучения')
plt.plot(lam, error_val, 'b-', label = 'Ошибка проверки')
plt.legend(loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Параметр регуляризации')
plt.ylabel('Ошибка')
plt.grid()
plt.show()