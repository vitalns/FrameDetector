import os
import math
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Отключение TensorFlow warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model

class ShapeClassifier:
    def __init__(self, method='geometry', cnn_model='model.h5'):
        """
        Классификация формы контура

        Параметры:
        cnn_model: имя модели классификации контуров
        method (str): метод классификации (geometry (по умолчанию), cnn, hybrid)
        """

        self.classes = ['circle', 'square', 'triangle']
        self.method = method

        if self.method != 'geometry':
            self.cnn_model_name = cnn_model
            self.cnn_model_path = os.path.join("cnn", self.cnn_model_name)
            if not os.path.exists(self.cnn_model_path):
                self.cnn_model_path = os.path.join("..", "cnn", self.cnn_model_name)
                print(f'Файл модели не найден: {self.cnn_model_path}')
                # raise FileNotFoundError(f'Файл модели не найден: {self.cnn_model_path}')
            self.cnn_model = load_model(self.cnn_model_path)  # Загрузка модели



    def classify_shape(self, frame=None, contour=None, contour_properties=None):
        """
        Классификация формы контура

        Параметры:
        frame (np.array): входной кадр в формате GrayScale [0/255]
        contour (np.array): Массив точек контура
        contour_properties (tuple): Характеристики контура (center_x, center_y, width, height, area)

        Возвращает:
        str: Идентификатор формы (circle, square, triangle, unknown)
        """
        geom_type = self._classify_with_geometry(contour)

        if self.method == 'cnn':
            return self._classify_with_cnn(frame, contour_properties)
        elif self.method == 'hybrid':
            if geom_type != 'unknown':
                return geom_type
            return self._classify_with_cnn(frame, contour_properties)
        else:
            return geom_type

    def _classify_with_geometry(self, contour):
        """
        Классификация формы контура с использованием геометрических характеристик

        Параметры:
        contour (np.array): Массив точек контура

        Возвращает:
        str: Идентификатор формы (circle, square, triangle, (rectangle), unknown)
        """

        # Вычисление периметра и площади контура
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        if perimeter == 0:
            return "unknown"

        # Коэффициент компактности (1.0 - идеальный круг)
        compactness = (4 * math.pi * area) / (perimeter ** 2)

        # Если компактность > 0.8 считаем кругом
        if compactness > 0.8:
            return "circle"

        # Аппроксимация контура полигонами (упрощение формы)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)

        # Определение формы по количеству вершин
        if vertices == 3:
            return "triangle"
        elif vertices == 4:
            # Проверка соотношения сторон для разделения квадрата и прямоугольника
            # x, y, w, h = cv2.boundingRect(contour)
            # aspect_ratio = w / float(h)
            return "square"  # if 0.95 <= aspect_ratio <= 1.05 else "rectangle"

        return "unknown"

    def _classify_with_cnn(self, frame, contour_properties):
        """
        Классификация контура с помощью CNN

        Параметры:
        frame (np.array): входной кадр в формате GrayScale [0/255]
        contour_properties (tuple): Характеристики контура (center_x, center_y, width, height, area)

        Возвращает:
        str: Идентификатор формы (circle, square, triangle)
        """

        processed_contour = self._preprocess_contour_for_cnn(frame, contour_properties)

        if processed_contour is None:
            return "unknown"

        pred = self.cnn_model.predict(processed_contour, verbose=0)  # Форма предсказания [[P0, P1, P2]]
        class_idx = np.argmax(pred)  # Находим индекс максимального значения в массиве
        return self.classes[class_idx]

    def _preprocess_contour_for_cnn(self, frame, contour_properties, img_size=(128, 128)):
        """
        Подготовка ROI для CNN

        Параметры:
        frame (np.array): входной кадр в формате GrayScale [0/255]
        contour_properties (tuple): Характеристики контура (center_x, center_y, width, height, area)
        img_size (tuple): размер области ROI для CNN (w, h)

        Возвращает:
        np.array: выделенная область кадра с контуром (shape = (1, 128, 128, 1), dtype = float32, format = [0/1])
        """

        Cx, Cy, w, h, _ = contour_properties

        margin = 10
        roi = frame[Cy - h // 2 - margin:Cy + h // 2 + margin, Cx - w // 2 - margin:Cx + w // 2 + margin]

        if roi.size == 0:
            return None

        # Подгоняем размер ROI под размер фото в обучающем датасете, для Canny используем интерполяцию INTER_NEAREST,
        # так как она сохраняет бинарность изображений [0/255]
        roi_cnn = cv2.resize(roi, img_size, interpolation=cv2.INTER_NEAREST)
        roi_cnn = roi_cnn.astype('float32') / 255.0  # Нормализация как при обучении модели
        roi_cnn = np.expand_dims(roi_cnn, axis=-1)  # Добавляем размерность канала
        roi_cnn = np.expand_dims(roi_cnn, axis=0)  # Добавляем размерность батча

        return roi_cnn  # Выделенная область кадра с контуром в форме (1, 128, 128, 1) float32 [0/1]