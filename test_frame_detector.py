import cv2
from video_stream.ethernet_connection import EthernetVideoStream
from video_stream.parse_rtsp import get_RTSP
from video_stream.virtual_camera import FrameGenerator
from video_analyzer.frame_analyzer import FrameAnalyzer

import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

if __name__ == "__main__":
    """Генерация RTSP URL"""
    # rtsp_url = "rtsp://192.168.1.100:554"  # Для запуска в IDE
    rtsp_url = get_RTSP.parse_command_line().generate_rtsp_url()  # Для запуска через консоль
    print(f"Connecting to: {rtsp_url}")

    with EthernetVideoStream(rtsp_url) as stream:

        if stream.is_connected:
            cap = stream.cap
        else:
            stream.cap.release()
            print("Start dummy video")
            video_gen = FrameGenerator(width=640, height=480)  # Инициализация виртуальной камеры
            video_gen.generate_video('test_video.avi', duration=20, object_count=7, noise=False, preview=True)
            cap = cv2.VideoCapture('test_video.avi')

        # Создание объекта класса FrameAnalyzer для обработки кадров
        analyzer = FrameAnalyzer(method='cnn')
        analyzer.clear_directory()

        while True:
            success, frame = cap.read()
            if not success:
                print("Unable to read")
                print("Stop reading")
                break

            # Получение информации о кадре и запись в файл frame_data\detector_results\'frame_time_stamp'.csv
            attributes, _ = analyzer.get_frame_attributes(frame, success)

            # Выделение контуров на кадре
            processed_frame = analyzer.edge_detection(frame, success)
            cv2.imshow('Contour detection', processed_frame.edge_frame)  # Отображение видео после детекции контуров

            # Обработка и распознание контуров
            # (результаты записываются в файл frame_data\frame_attributes\contours_time_stamp'.csv)
            contour_data = processed_frame.process_contours()
            # contour_data = analyzer.edge_detection(frame, success).process_contours()  # Методы можно использовать последов.

            # Отображение видео после распознования
            contour_frame = analyzer.draw_contour_info(frame, contour_data)
            cv2.imshow('Contour classification', contour_frame)

            # Вывод результатов
            print(f"Time: {str(attributes.at[0, 'timestamp'])[:-3]}")  # Временная метка кадра
            for (index, data) in contour_data.iterrows():  # Информация о размере и положении контуров
                print(data['center_x'], data['center_y'], data['width'], data['height'])
            print("-" * 50)

            # Прерывание цикла при нажатии клавиши 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Запуск из командной строки cmd:
# Перейти в папку проекта "cd C:\Users\nazarov_vs\PycharmProjects\FrameDetector",
# Запустить скрипт "python test_frame_detector.py --ip 192.168.1.100 --port 554"
# ! Сообщения Keras DeprecationWarning печатаются через С++-логирование до загрузки Python, запуск в CMD команды
# set TF_CPP_MIN_LOG_LEVEL=2 сообщения не скрывает, запуск как строка ниже не работает.
# "set TF_CPP_MIN_LOG_LEVEL=2 python test_frame_detector.py --ip 192.168.1.100 --port 554"