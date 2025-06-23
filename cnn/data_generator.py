import cv2
import numpy as np
from video_stream.virtual_camera import FrameGenerator

# Генерация кадров для CNN: 3 класса по папкам
# Параметры кадров:
camera = FrameGenerator(width=128, height=128)
camera.generate_cnn_data(number=6000, output_dir='cnn_data', noise=False, canny=True)

# Контроль данных
img_triangle = cv2.imread('cnn_data/triangle/5.png', cv2.IMREAD_GRAYSCALE)
# Парметры 6 кадра: shape = (128, 128), dtype = uint8, format = grayscale [0/255]
print(img_triangle.shape, img_triangle.dtype, np.unique(img_triangle))
cv2.imshow('Frame_circle_5', img_triangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(f'Размер обучающего датасета: {len(camera.frames)}')
