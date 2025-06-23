import cv2
import random
from video_stream.virtual_camera import FrameGenerator

# Инициализация виртуальной камеры ---------------------------------
camera = FrameGenerator(width=320, height=240)
# Создаем необходимое количество объектов
for _ in range(3):
    camera.add_object(random.choice(list(camera.classes.keys())))  # Добавляем случайный объект

while True:
    frame = camera.generate_frame_stream(noise=False)  # Генерация кадра

    cv2.imshow('Frame Stream', frame)  # Отображение результатов

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Прерывание цикла по нажатию 'Q'
        break

# Закрытие всех окон OpenCV
cv2.destroyAllWindows()


# Запуск генерации видео ---------------------------------
# generator = FrameGenerator(width=320, height=240)
# generator.generate_video('test_video.avi', duration=5, object_count=5, preview=True)


# Генерация кадров для CNN: 3 класса по папкам ---------------------------------
# camera = FrameGenerator(width=320, height=240)
# camera.generate_cnn_data(number=30, output_dir='cnn_data', noise=True)
# cv2.imshow('Frame_circle_5', cv2.imread('cnn_data/circle/5.jpg'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(camera.labels)
# print(len(camera.frames))


# Инициализация случайных кадров ---------------------------------
# camera = FrameGenerator(width=320, height=240)
# camera.clear_frames()  # Очищаем список кадров
# # Генерация кадров
# frames, labels = camera.generate_random_frames(number=50)  # Можно не распаковывать, а обращаться через атрибуты
#
# # Отображение результатов как видео
# for frame in frames:
#     cv2.imshow('Random Frames', frame)
#     # Прерывание цикла по нажатию 'Q'
#     if cv2.waitKey(int(1000 / camera.fps)) & 0xFF == ord('q'):  # Задержка между кадрами 1000 / camera.fps
#         break
#
# cv2.destroyAllWindows()   # Закрытие всех окон OpenCV
#
# cv2.imshow('Frame_5', frames[5])
# cv2.waitKey(0)
#
# cv2.destroyAllWindows() # Закрытие всех окон OpenCV
#
# print(camera.frames[0].shape)  # Можно не распаковывать, а обращаться через атрибуты
# print(labels)  # Список маркеров
# print(len(frames))  # Длина списка кадров
