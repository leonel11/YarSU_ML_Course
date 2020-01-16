import numpy as np
import matplotlib.pyplot as plt

def displayData(X, num):
    """
        Функция позволяет выполнить визуализацию данных, сохраненных 
        в матрице объекты-признаки X. Данные визуализируется в виде 
        цифровых изображений, размещенных в позициях прямоугольной 
        сетки размеры num x num
    """
    
    plt.figure(figsize = (4, 4))
    
    img_num = 0
    for i in range(num):
        for j in range(num):
            img = X[img_num, :].reshape(20, 20).transpose()
            img = img + np.abs(np.min(X)); img = img / np.max(img)
            plt.subplot(num, num, img_num + 1)
            plt.imshow(img, cmap = 'gray')
            plt.axis('off')
            img_num = img_num + 1
    
    plt.show()