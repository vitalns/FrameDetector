import os
import cv2
import numpy as np
import random
import math
import time


class FrameGenerator:
    """
    Класс для создания синтетических данных с разноцветными геометрическими фигурами (круг, квадрат, треугольник).
    В классе 4 основных метода: генерация потока кадров (generate_frame_stream),
    генерация видеофайла 'avi' (generate_video), генерация случайных кадров (generate_random_frames) и
    генерация случайных кадров с предобработкой для обучений CNN (generate_cnn_data).
    Параметры фигур (размер, цвет, поворот, скорость движения, начальная позиция на кадре) задаются случайным образом.
    Фигуры движутся прямолинейно. При столкновении с границей кадра происходит отскок с инверсией скорости.
    На кадр можно добавить гаусовый шум.
    Минимальный размер кадра = 3 * максимальный размер фигуры (по умолчанию 50)
    """
    def __init__(self, width=800, height=600, fps=30):
        """
        Инициализация генератора фигур
        :param width: ширина кадра
        :param height: высота кадра
        :param fps: частота кадров (frames per second)
        """
        self.width = width
        self.height = height
        self.fps = fps

        self.objects = []  # Список для хранения объектов (фигур)
        self.frames = []  # Список для хранения кадров
        self.labels = []  # Список для хранения маркеров кадра

        self.classes = {  # Словарь классов фигур
            0: {'name': 'circle'},  # Круг
            1: {'name': 'square'},  # Квадратq
            2: {'name': 'triangle'}  # Треугольник
        }

    def add_object(self, class_id):
        """
        Добавление нового объекта в список
        :param class_id: идентификатор класса фигуры (0-2)
        """
        obj = {
            'class_id': class_id,  # Тип фигуры
            'position': [  # Начальная позиция (X, Y)
                random.randint(20, self.width - 20),  # X: отступ 20px от краев
                random.randint(20, self.height - 20)  # Y: отступ 20px от краев
            ],
            'speed': [  # Скорость перемещения (по X и Y)
                random.choice([-3, -2, -1, 1, 2, 3]),  # Случайная скорость X
                random.choice([-3, -2, -1, 1, 2, 3])  # Случайная скорость Y
            ],
            'size': random.randint(20, 40),  # Размер фигуры
            'angle': random.randint(0, 360),  # Угол поворота
            'color': (  # Случайный цвет в формате BGR
                random.randint(0, 255),  # Синий канал
                random.randint(0, 255),  # Зеленый канал
                random.randint(0, 255)  # Красный канал
            )
        }
        self.objects.append(obj)

    def draw_objects(self, frame):
        """
        Отрисовка всех объектов на кадре
        :param frame: текущий кадр видео
        :return: кадр с нарисованными фигурами
        """
        for obj in self.objects:
            # Извлекаем параметры объекта
            color = obj['color']  # Цвет фигуры
            class_name = self.classes[obj['class_id']]['name']  # Название класса
            center = tuple(obj['position'])  # Центр фигуры (X, Y)
            size = obj['size']  # Размер фигуры
            angle = obj['angle']  # Угол поворота
            # Отрисовка круга
            if class_name == 'circle':
                cv2.circle(
                    img=frame,
                    center=center,
                    radius=size,
                    color=color,
                    thickness=-1  # -1 означает заливку фигуры
                )
            # Отрисовка квадрата или треугольника
            else:
                # Генерация вершин для фигуры
                if class_name == 'square':
                    vertices = self.draw_rotated_shape(center, size, angle, 4)
                elif class_name == 'triangle':
                    vertices = self.draw_rotated_shape(center, size, angle, 3)

                # Рисуем многоугольник
                cv2.fillPoly(img=frame, pts=[vertices], color=color)

        return frame

    def update_positions(self):
        """Обновление позиций объектов с проверкой границ кадра"""
        for obj in self.objects:
            for i in range(2):  # Обрабатываем X (i=0) и Y (i=1)
                obj['position'][i] += obj['speed'][i]  # Обновляем позицию

                # Проверка выхода за границы кадра
                max_dim = self.width if i == 0 else self.height
                if obj['position'][i] < 0 or obj['position'][i] > max_dim:
                    obj['speed'][i] *= -1  # Инверсия скорости при столкновении с границей

    def draw_rotated_shape(self, center, size, angle, num_sides):
        """
        Генерация вершин для повернутой фигуры
        :param center: центр фигуры (X, Y)
        :param size: размер фигуры (радиус для круга)
        :param angle: угол поворота в градусах
        :param num_sides: количество сторон (3 для треугольника, 4 для квадрата)
        :return: массив координат вершин
        """
        points = []
        for i in range(num_sides):
            # Рассчитываем угол для каждой вершины
            theta = math.radians(angle + (360 / num_sides) * i)

            # Вычисляем координаты вершины
            x = center[0] + size * math.cos(theta)
            y = center[1] + size * math.sin(theta)

            points.append((int(x), int(y)))

        return np.array(points, dtype=np.int32)

    def generate_cnn_data(self, number=30, output_dir='cnn_data', noise=False, canny=True):
        """
        Генерация кадров с разноцветными фигурами и предобработкой для CNN
        :param number: число кадров
        :param output_dir: директория для сохранения
        :param noise: можно добавить на кадр гауссов шум
        :param canny: можно обработать фигуру на кадре с помощью алгоритма Canny
        :return: кортеж из списка кадров в виде numpy массивов и списка меток для фигур на кадрах,
            а также кадры сохраняются в формате PNG в директории output_dir (для каждой фигуры своя папка).
        """

        # Создание директории для результатов
        os.makedirs(output_dir, exist_ok=True)

        for label in list(self.classes.keys()):
            # Создание папки для результатов
            data_path = os.path.join(output_dir, self.classes[label]['name'])
            os.makedirs(data_path, exist_ok=True)

            # Очистка папки с результатами, если уже сущестыует
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                os.remove(file_path)

            # Генерация кадров
            n = 0
            while n < number//3:
                data_path = os.path.join(output_dir, self.classes[label]['name'])
                os.makedirs(data_path, exist_ok=True)
                frame_path = os.path.join(data_path, f"{n}.png")  # Сохраняем в png c 1 цветовым каналом
                # (В JPEG всегда 3 канала даже если исходник серый)
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Создаем пустой кадр (черный фон)
                self.objects = []  # Обнуляем список объектов
                self.add_object(label)  # Создаем объект
                frame = self.draw_objects(frame)  # Рисуем объекты на кадре

                if noise:
                    gauss_noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
                    frame = cv2.add(frame, gauss_noise)  # Добавляем гауссов шум

                # Для распознования формы фигур предпочтительнее использовать градации серого
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Конвертация в градации серого

                # Алгоритм Canny плохо распонаёт контуры фигур цвет которых близок к цвету фона:
                # примерно 3% черных кадров, если есть предобработка гаусовым фильтром - 8%.
                if canny:  # На выходе серое изображение в формате [0/255] тип uint8
                    blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # Гауссово размытие для шумоподавления
                    frame = cv2.Canny(blurred, 100, 200, apertureSize=3)  # Детекция краев алгоритмом Canny
                    if  np.all(frame == 0):  # Убираем чисто черные кадры
                        continue
                n += 1

                cv2.imwrite(frame_path, frame)  # Сохраняется изображение в grayscale [0/255] uint8
                self.labels.append(label)
                self.frames.append(frame.copy())  # Сохраняем кадр в список

        return self.frames, self.labels

    def generate_frame_stream(self, noise=False):
        """
        Генерация потока кадров с движущимися фигурами
        :param noise: можно добавить на кадр гауссов шум
        :return: массив точек кадра
        """

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Создаем пустой кадр (черный фон)

        self.update_positions()  # Обновляем позиции объектов
        frame = self.draw_objects(frame)  # Рисуем объекты на кадре

        if noise:
            gauss_noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)  # Создаём шум
            frame = cv2.add(frame, gauss_noise)  # Добавляем гауссов шум

        time.sleep(1 / self.fps)  # Добавляем задержку кадра

        self.frames.append(frame.copy())  # Сохраняем кадр в список

        return frame

    def generate_video(self, filename, duration=10, object_count=1, noise=True, preview=True):
        """
        Генерация видео с движущимися разноцветными фигурами
        :param filename: имя для сохранения видео
        :param duration: продолжительность в секундах
        :param object_count: число фигур на кадре
        :param noise: можно добавить на кадр гауссов шум
        :param preview: просмотр сгенерированного видео
        :return: сгенерированное видео сохраняется в рабочей дирректории
        """

        # Создаем указанное количество объектов
        for _ in range(object_count):
            self.add_object(random.choice(list(self.classes.keys())))

        # Настройка VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))

        # Цикл генерации кадров
        for _ in range(int(duration * self.fps)):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Создаем пустой кадр (черный фон)

            self.update_positions()  # Обновляем позиции объектов
            frame = self.draw_objects(frame)  # Рисуем объекты на кадре

            if noise:
                gauss_noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, gauss_noise)  # Добавляем гауссов шум

            self.frames.append(frame.copy())  # Сохраняем кадр в список
            out.write(frame)  # Записываем кадр в видео

            # Отображаем кадры в реальном времени
            if preview:
                cv2.imshow('Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Выход по клавише Q
                    break

        # Завершение работы
        out.release()  # Закрываем видеофайл
        cv2.destroyAllWindows()  # Закрываем все окна OpenCV

    def generate_random_frames(self, number=20, noise=True):
        """
        Генерация кадров со случайными разноцветными фигурами
        :param number: количество генерируемых кадров
        :param noise: можно добавить на кадр гауссов шум
        :return: список кадров в виде массива np.array и список меток для фигур на кадрах
        """
        for i in range(number):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # Создаем пустой кадр (черный фон)
            label = random.choice(list(self.classes.keys()))
            self.objects = []  # Обнуляем список объектов
            self.add_object(label)  # Создаем объект
            frame = self.draw_objects(frame)  # Рисуем объекты на кадре
            if noise:
                gauss_noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, gauss_noise)  # Добавляем гауссов шум
            self.labels.append(label)
            self.frames.append(frame.copy())  # Сохраняем кадр в список

        return self.frames, self.labels

    def get_frames(self):
        """Возвращает список сохраннёных кадров"""
        return self.frames

    def clear_frames(self):
        """Очищает список кадров"""
        self.frames.clear()
