import os
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from video_analyzer.shape_classifier import ShapeClassifier


class FrameAnalyzer:
    def __init__(self, output_dir="frames_data", min_contour_area=350, noise=False, method='geometry'):
        """
        Обработка кадра и поиск контуров

        Параметры:
        output_dir: папка для хранения результатов анализа
        min_contour_area: минимальный размер искомого контура в пикселях (обычно 150 - 250)
        noise (bool): при шуме увиличивается размер окна свертки алгоритма Canny
        method (str): метод классификации (geometry (по умолчанию), cnn, hybrid)

        """
        # # Инициализация видеопотока, захват видеопотока происходит в ethernet_connection
        # (init_param video_source: источник видео (0 - веб-камера, путь к файлу или IP-камеры))
        # self.cap = cv2.VideoCapture(video_source)
        # if not self.cap.isOpened():
        #     raise ValueError("Could not open video source")

        self.min_contour_area = min_contour_area
        self.noise = noise
        self.shape_classifier = ShapeClassifier(method=method)

        # При вызове методов по отдельности и в произвольном порядке проверяется существование временной метки
        self.timestamp = 0
        
        # Создание директории для результатов
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.frame_attributes = os.path.join(self.output_dir, 'frames_attributes')
        os.makedirs(self.frame_attributes, exist_ok=True)
        self.detector_results = os.path.join(self.output_dir, 'detector_results')
        os.makedirs(self.detector_results, exist_ok=True)

        # Общая таблица с атрибутами для всех кадров
        self.frame_attribute_result = pd.DataFrame(columns=['timestamp',
                                                            'width',
                                                            'height',
                                                            'saturation',
                                                            'white_pixels',
                                                            'black_pixels'])
        # Общая таблица с результатами детекции фигур для всех кадров
        self.contour_detector_result = pd.DataFrame(columns=['timestamp', 'number', 'valid_number'])

    def get_frame_attributes(self, frame, success=True):
        """
        Обработка и анализ текущего кадра

        Параметры:
        frame (np.array): входной кадр в формате BGR
        success (bool): индикатор чтения кадра

        Возвращает:
        (pd.DataFrame, self): таблица с параметрами кадров
        """
        # Чтение кадра из видеопотока, чтение видеопотока происходит в ethernet_connection
        # ret, frame = self.cap.read()
        # if not ret:
        #     return None

        if not success:
            return None

        # Анализ параметров кадра ---------------------------------
        # Размер кадра
        height, width = frame.shape[:2]
        # Насыщенность
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Преобразование в HSV (hue, saturation, value)
        saturation = np.mean(hsv[:, :, 1])  # Среднее значение S-канала

        # Проверка на чисто белый и черный цвета
        white_mask = cv2.inRange(frame, (255, 255, 255), (255, 255, 255))  # Создание белой цветовой маски
        white_pixels = cv2.countNonZero(white_mask) # Подсчёт белых (ненулевых) пикселей
        black_mask = cv2.inRange(frame, (0, 0, 0), (0, 0, 0))  # Создание чёрной цветовой маски
        black_pixels = cv2.countNonZero(black_mask) # Подсчёт чёрных (ненулевых) пикселей

        # Проверка на чисто белый цвет на выходе True/False
        # white_mask = cv2.inRange(frame, (255, 255, 255), (255, 255, 255))  # Создание белой цветовой маски
        # white_pixels = cv2.countNonZero(white_mask) > 0  # Подсчёт белых (ненулевых) пикселей
        # или через numpy
        # white_mask = np.all(frame == 255, axis=2)  # Если все 3 канала 255, то пиксель True
        # has_white = np.any(white_mask)  # Если есть хотя бы один True, то True

        # Текущая дата и время
        self.timestamp = datetime.now()

        # Сохранение результатов ---------------------------------
        # Генерация уникального имени файла с датой и временем с точностью до мс
        filename = self.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filepath = os.path.join(self.frame_attributes, f"frame_{filename}.csv")
        # Создание DataFrame с результатами
        frame_data = pd.DataFrame([{
            'timestamp': self.timestamp,
            'width': width,
            'height': height,
            'saturation': round(saturation, 2),  # Округление до 2 знаков
            'white_pixels': white_pixels,
            'black_pixels': black_pixels
        }])
        frame_data.to_csv(filepath, index=False)  # Сохранение без индексов
        self.frame_attribute_result = pd.concat([self.frame_attribute_result, frame_data], ignore_index=True)
        return frame_data, self

    def edge_detection(self, frame, success=True):
        """
        Обнаружение краев на кадре с помощью алгоритма Canny

        Параметры:
        frame (np.array): входной кадр в формате BGR
        success (bool): индикатор чтения кадра

        Возвращает:
        self: кадр с выделенными краями в формате BGR и Grayscale (np.array), список найденых контуров (List[np.array])
        """
        if not success:
            return None

        # Определение контуров на изображении:
        # Конвертация в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Гауссово размытие для шумоподавления
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.noise:  # При шуме результаты детекции с Canny лучше с apertureSize=7(5).
            apertureSize = 7
        else:
            apertureSize = 3

        # Детекция краев алгоритмом Canny
        self.edges = cv2.Canny(blurred, 100, 200, apertureSize=apertureSize)

        # Конвертация обратно в BGR для возможности отображения
        self.edge_frame = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)

        # Поиск контуров
        contours, _ = cv2.findContours(self.edges,
                                       cv2.RETR_EXTERNAL,  # Режим поиска
                                       cv2.CHAIN_APPROX_SIMPLE)  # Метод аппроксимации

        # Фильтрация контуров по минимальной площади
        self.valid_contours = [c for c in contours
                                    if cv2.contourArea(c) >= self.min_contour_area]

        # Сохранение результатов поиска контуров в общую таблицу с временной меткой
        if self.timestamp:
            timestamp = self.timestamp
        else:
            timestamp = datetime.now()

        edge_data = pd.DataFrame([{'timestamp': timestamp,
                                    'number': len(contours),
                                    'valid_number': len(self.valid_contours)}])

        self.contour_detector_result = pd.concat([self.contour_detector_result, edge_data], ignore_index=True)

        return self


    def _get_contour_properties(self, contour):
        """
        Вычисление геометрических характеристик контура

        Параметры:
        contour (np.array): Массив точек контура

        Возвращает:
        tuple: (center_x, center_y, width, height, area)
        """

        # # Вычисление моментов для определения центра масс
        # M = cv2.moments(contour)
        # if M["m00"] == 0:
        #     return None
        #
        # # Центр контура
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])

        # Bounding box и площадь
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Центр контура
        cX = x + w // 2
        cY = y + h // 2


        return cX, cY, w, h, area

    def process_contours(self, gray_frame=None, contours=None):
        """
        Обработка всех контуров в кадре

        Параметры:
        frame (np.array): кадр в формате GrayScale [0/255]
        contours (list of np.array): Список массивов точек контуров

        Возвращает:
        DataFrame: Таблица с данными контуров
        """
        if contours is None:
            contours = self.valid_contours

        if gray_frame is None:
            gray_frame = self.edges

        # Список для результатов обработки всех контуров на кадре
        contour_data = []

        # Цикл обработки всех обнаруженных контуров
        for i, contour in enumerate(contours):
            # Фильтрация по размеру контура (пропуск маленьких контуров) сделана в edge_detector
            # if cv2.contourArea(contour) < min_contour_area:
            #     continue

            # Получение характеристик контура
            properties = self._get_contour_properties(contour)
            if not properties:
                continue

            cX, cY, w, h, area = properties

            # Классификация формы
            shape_type = self.shape_classifier.classify_shape(gray_frame, contour, properties)

            # Сохранение результатов в список
            contour_data.append([i, self.timestamp, shape_type, cX, cY, w, h, area])

        # Сохранение результатов ---------------------------------
        # Создание таблицы с результатами
        columns = ['contour_id', 'timestamp', 'shape_type', 'center_x', 'center_y', 'width', 'height', 'area']
        df = pd.DataFrame(contour_data, columns=columns)
        # Генерация уникального имени файла с датой и временем с точностью до мс
        filename = self.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filepath = os.path.join(self.detector_results, f"contours_{filename}.csv")

        # Сохранение результатов в CSV-файл
        df.to_csv(filepath, index=False)  # Сохранение без индексов

        return df

    def draw_contour_info(self, frame, contour_data,  contours=None, contour_color=(0, 0, 255), thickness=2):
        """
        Отрисовка контура и текстовой информации на кадре

        Параметры:
        frame (np.array): Исходный кадр
        contours (list of np.array): Список контуров для отрисовки
        contour_data (dict): Данные контура
        """
        if contours is None:
            contours = self.valid_contours

        # Отрисовка контуров
        for i, (contour, (index, data)) in enumerate(zip(contours, contour_data.iterrows())):
            cv2.drawContours(frame, [contour], -1, contour_color, thickness)

            # Формирование текстовой метки
            label = f"{data['shape_type']} ({data['contour_id']})"

            # Отрисовка текста
            cv2.putText(frame, label,
                        (data['center_x'], data['center_y']),  # Позиция
                        cv2.FONT_HERSHEY_SIMPLEX,  # Шрифт
                        0.5,  # Масштаб
                        (255, 0, 0),  # Цвет (BGR)
                        2)  # Толщина
        return frame

    def clear_directory(self):
        """Очистка директории сохранения результатов анализа"""
        if not os.path.exists(self.output_dir):
            print(f"Директория {self.output_dir} не существует!")
        for filename in os.listdir(self.frame_attributes):
            file_path = os.path.join(self.frame_attributes, filename)
            os.remove(file_path)
        for filename in os.listdir(self.detector_results):
            file_path = os.path.join(self.detector_results, filename)
            os.remove(file_path)
