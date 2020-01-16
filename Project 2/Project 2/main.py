## Практическое задание № 2. Логистическая регрессия

# Инициализация
import numpy as np
import matplotlib.pyplot as plt

from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from featureNormalize import featureNormalize
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict

# ================= Часть 1. Визуализация данных =================

print('Часть 1. Визуализация данных')

# Загрузка данных и формирование матрицы объекты-признаки X и вектора меток y
data = np.loadtxt('data.txt', delimiter = ',')
m = data.shape[0]

X = np.array(data[:, 0:2])
y = np.array(data[:, 2:3])

# Визуализация данных
plotData(X, y)
plt.legend(['Аттестован', 'Не аттестован'], loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Первый экзамен студента')
plt.ylabel('Второй экзамен студента')
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# =========== Часть 2. Вычисление стоимостной функции ============

print('Часть 2. Вычисление стоимостной функции')

# Нормализация свойств и добавление единичного признака
m, n = X.shape
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones((m, 1)), X), axis = 1)

# Задание начальных параметров модели
initial_theta = np.zeros([n + 1, 1])

# Вычисление значений стоимостной функции
J = computeCost(X, y, initial_theta)
print('Значение стоимостной функции для начальных параметров модели: {:.4f}'.format(J))
#print('Значение стоимостной функции для начальных параметров модели: '+ str(J))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ================== Часть 3. Градиентный спуск ==================

print('Часть 3. Градиентный спуск')

# Задание параметров градиентного спуска
iterations = 400
alpha = 1

# Выполнение градиентного спуска
theta, J_history = gradientDescent(X, y, initial_theta, alpha, iterations)
print('Найденные параметры модели: ')
print(theta)

# Визуализация процесса сходимости
plt.figure()
plt.plot(np.arange(len(J_history)) + 1, J_history, '-b', linewidth = 2)
plt.xlabel('Число итераций');
plt.ylabel('Значение стоимостной функции')
plt.grid()
plt.show()

# Визуализация границы решения
plotDecisionBoundary(X, y, theta, mu, sigma)
plt.legend(['Аттестован', 'Не аттестован'], loc = 'upper right', shadow = True, fontsize = 12, numpoints = 1)
plt.xlabel('Первый экзамен студента')
plt.ylabel('Второй экзамен студента')
plt.show()

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ======= Часть 4. Предсказание и доля правильных ответов ========


print('Часть 4. Предсказание и доля правильных ответов')

# Предсказание вероятности аттестации студента для оценки за первый 
# экзамен равной 45 и второй экзамен равной 85
prob = sigmoid(np.dot(np.array([[1, (45 - mu[0]) / sigma[0], (85 - mu[1]) / sigma[1]]]), theta))
print('Для оценки за первый экзамен равной 45 и второй экзамен равной 85 вероятность аттестации студента = {:.4f}'.format(prob[0, 0]))

# Вычисление доли правильных ответов обученной логистической регрессии
p = predict(X, theta)
acc = np.sum(1 - np.abs(p - y)) / len(y) * 100
print('Доля правильных ответов обученной логистической регрессии = {:.4f}'.format(acc))