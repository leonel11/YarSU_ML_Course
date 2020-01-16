## Практическое задание № 6. Кластеризация с использованием алгоритма K-средних

# Инициализация
import numpy as np
import scipy.io as spi
import scipy.misc as spm
import matplotlib.pyplot as plt

from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids

# =============== Часть 1. Поиск ближайших средних ===============

print('Часть 1. Поиск ближайших средних')

# Загрузка данных и формирование матрицы объекты-признаки X
data = spi.loadmat('data.mat')

X = data['X']

# Выбор начального множества средних
K = 3 # число средних
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Поиск ближайших средних с учетом их начальных значений
idx = findClosestCentroids(X, initial_centroids)

print('Номера ближайших средних для первых трех примеров: {:.0f} {:.0f} {:.0f}'.format(idx[0, 0], idx[1, 0], idx[2, 0]))

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ============= Часть 2. Вычисление значений средних =============

print('Часть 2. Вычисление значений средних')

# Вычисление значений средних
centroids = computeCentroids(X, idx, K)

print('Вычисленные значения средних:')
print(centroids)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ====== Часть 3. Кластеризация с использованием K-средних =======

print('Часть 3. Кластеризация с использованием K-средних')

# Задание числа средних и числа итераций алгоритма
K = 3
max_iters = 10

# Выбор начального множества средних
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Запуск алгоритма K-средних
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ==== Часть 4. Кластеризация на основе К-средних на пикселях ====

print('Часть 4. Кластеризация на основе К-средних на пикселях')

#  Загрузка изображения
A = spm.imread('bird_small.png').astype('float64')

# Нормировка изображения
A = A / 255

# Вычисление размера изображения и формирование матрицы-объекты признаки
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3)

# Задание числа средних и числа итераций алгоритма
K = 3
max_iters = 10

# Случайная инициализация средних
initial_centroids = kMeansInitCentroids(X, K)

# Запуск алгоритма K-средних
centroids, idx = runkMeans(X, initial_centroids, max_iters)

input('Программа остановлена. Нажмите Enter для продолжения ... \n')

# ================= Часть 5. Сжатие изображения ==================

print('Часть 5. Сжатие изображения')

# Поиск ближайших средних пикселей изображения
idx = findClosestCentroids(X, centroids).astype('uint8')

# Формирование сжатого изображения
X_recovered = np.zeros(X.shape)
for i in range(X.shape[0]):
	X_recovered[i, :] = centroids[idx[i, :], :]

X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Визулизация исходного и сжатого изображений
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title('Исходное изображение')

plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title('Сжатое изображение')

# СЛУЧАЙНАЯ ИНИЦИАЛИЗАЦИЯ КЛАСТЕРОВ

plt.show()