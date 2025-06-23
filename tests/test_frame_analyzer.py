import cv2
import random
from video_stream.virtual_camera import FrameGenerator
from video_analyzer.frame_analyzer import FrameAnalyzer

# Запуск через генерацию видео
# generator = FrameGenerator(width=640, height=480)
# generator.generate_video('test_video.avi', duration=5, object_count=7, noise=False, preview=True)
# cap = cv2.VideoCapture('test_video.avi')
#
# analyzer = FrameAnalyzer(method='cnn')
# analyzer.clear_directory()
#
# while True:
#     success, frame = cap.read()
#     if not success:
#         print("Stop reading")
#         break
#
#     # Получение информации о кадре
#     attributes, _ = analyzer.get_frame_attributes(frame, success)
#
#     # Выделение контуров на кадре
#     processed_frame = analyzer.edge_detection(frame, success)
#     cv2.imshow('Contour detection', processed_frame.edge_frame)  # Отображение видео после детекции контуров
#
#     # Обработка и распознание контуров
#     contour_data = processed_frame.process_contours()
#     # contour_data = analyzer.edge_detection(frame, ret).process_contours()  # Методы можно использовать последов.
#     # print(contour_data)
#
#     # Отображение видео после распознования
#     contour_frame = analyzer.draw_contour_info(frame, contour_data)
#     cv2.imshow('Contour classification', contour_frame)
#
#     # Вывод характеристик кадров
#     print(f"Time: {str(attributes.at[0, 'timestamp'])[:-3]}")
#     # print(f"Frame size: {int(attributes['width'])}x{int(attributes['height'])}")
#     # print(f"Saturation: {int(attributes['saturation'])}")
#     # print(f"Contains white: {int(attributes['white_pixels'])}")
#     # print(f"Contains black: {int(attributes['black_pixels'])}")
#     print("-" * 50)
#
#     # Для выхода из цикла нажмите 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # # Закрытие всех окон OpenCV
# # cv2.destroyAllWindows()
#
# # attribute_results = analyzer.frame_attribute_result
# # print(attribute_results[attribute_results.white_pixels < 50])  # Фильтрация по колличеству белых пикселей
# edge_results = analyzer.contour_detector_result
# print(f"Mean_valid_number of detected contour {edge_results.valid_number.mean().round(1)}")



# Запуск через виртуальную камеру ---------------------------------
camera = FrameGenerator(width=640, height=480)
for _ in range(5):
    camera.add_object(random.choice(list(camera.classes.keys())))  # Добавляем случайный объект

analyzer = FrameAnalyzer(method='hybrid')
analyzer.clear_directory()

while True:
    # Генерация кадра
    frame = camera.generate_frame_stream(noise=False)  # Генерация кадра
    cv2.imshow('Frame Stream', frame)  # Отображение кадра

    # Получение информации о кадре
    attributes, _ = analyzer.get_frame_attributes(frame)

    # Выделение контуров на кадре
    processed_frame = analyzer.edge_detection(frame)
    cv2.imshow('Contour detection', processed_frame.edge_frame)  # Отображение видео после детекции контуров

    # Обработка и распознание контуров
    contour_data = processed_frame.process_contours()

    # Отображение видео после распознования
    contour_frame = analyzer.draw_contour_info(frame, contour_data)
    cv2.imshow('Contour classification', contour_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Прерывание цикла по нажатию 'Q'
        break



# Закрытие всех окон OpenCV
cv2.destroyAllWindows()

# Вывод результатов анализа кадров
analys_results = analyzer.frame_attribute_result
print(analys_results[analys_results.white_pixels > 350])
edge_results = analyzer.contour_detector_result
print(f"Mean_valid_number of detected contour {edge_results.valid_number.mean().round(1)}")