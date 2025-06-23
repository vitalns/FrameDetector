import argparse
import cv2
from video_stream.ethernet_client import EthernetClient
from video_stream.uart_mavlink_client import UARTMavlinkClient
from video_analyzer.frame_analyzer import FrameAnalyzer

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Обработка видео с камеры и отправка данных по UART (MAVLink)"
    )

    # Параметры Ethernet подключения
    ethernet_group = parser.add_argument_group('Ethernet параметры')
    ethernet_group.add_argument(
        '--rtsp-url',
        type=str,
        default='rtsp://192.168.1.100:554',
        help="Полный RTSP URL (например: rtsp://192.168.1.100:554)"
    )

    # Параметры UART
    uart_group = parser.add_argument_group('UART параметры')
    uart_group.add_argument(
        '--uart-port',
        default='COM3',
        help="Порт UART (например /dev/ttyUSB0 или COM3)"
    )
    uart_group.add_argument(
        '--baudrate',
        type=int,
        default=921600,
        help="Скорость передачи (по умолчанию: 921600)"
    )
    uart_group.add_argument(
        '--bytesize',
        type=int,
        choices=[5, 6, 7, 8],
        default=8,
        help="Количество бит данных (5, 6, 7, 8)"
    )
    uart_group.add_argument(
        '--parity',
        choices=['none', 'even', 'odd'],
        default='odd',
        help="Контроль четности (none, even, odd)"
    )
    uart_group.add_argument(
        '--stopbits',
        type=float,
        choices=[1, 1.5, 2],
        default=1,
        help="Количество стоп-бит (1, 1.5, 2)"
    )

    # Параметры MAVLink
    mavlink_group = parser.add_argument_group('MAVLink параметры')
    mavlink_group.add_argument(
        '--mavlink-version',
        type=int,
        choices=[1, 2],
        default=2,
        help="Версия MAVLink (1 или 2)"
    )
    mavlink_group.add_argument(
        '--message-id',
        type=int,
        default=250,
        help="ID кастомного сообщения MAVLink (180-255)"
    )

    return parser.parse_args()

def main():
    """Основная функция"""
    args = parse_args()
    print(f'Параметры подключения по RTSP URL: {args.rtsp_url}')
    print(f'Параметры подключения по UART: \n'
          f'uart-port {args.uart_port} \n'
          f'baudrate {args.baudrate} \n'
          f'bytesize {args.bytesize} \n'
          f'parity {args.parity} \n'
          f'stopbits {args.stopbits} \n')

    try:
        # Инициализация клиентов
        ethernet_client = EthernetClient(rtsp_url=args.rtsp_url)
        uart_client = UARTMavlinkClient(
            port=args.uart_port,
            baudrate=args.baudrate,
            bytesize=args.bytesize,
            parity=args.parity,
            stopbits=args.stopbits,
            mavlink_version=args.mavlink_version,
            message_id=args.message_id
        )

        # Подключение
        ethernet_client.connect()
        uart_client.connect()

        # Основной цикл обработки
        while True:
            success, frame = ethernet_client.get_frame()  # Чтение кадров

            # Обработка кадров
            analyzer = FrameAnalyzer(method='cnn')
            analyzer.clear_directory()

            # Получение информации о кадре и запись в файл frame_data\detector_results\'frame_time_stamp'.csv
            analyzer.get_frame_attributes(frame, success)

            # Выделение контуров на кадре
            processed_frame = analyzer.edge_detection(frame, success)

            # Обработка и распознание контуров
            # (результаты записываются в файл frame_data\frame_attributes\contours_time_stamp'.csv)
            contour_data = processed_frame.process_contours()
            # contour_data = analyzer.edge_detection(frame, ret).process_contours()  # Методы можно использовать последов.

            for (index, data) in contour_data.iterrows():  # Информация о размере и положении контуров
                x, y, w, h = data['center_x'], data['center_y'], data['width'], data['height']
                uart_client.send_contour_data(x, y, w, h)

            # Прерывание цикла при нажатии клавиши 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        ethernet_client.disconnect()
        uart_client.disconnect()

if __name__ == "__main__":
    main()


# Запуск из командной строки cmd:
# - перейти в папку проекта "C:\Users\nazarov_vs\PycharmProjects\FrameDetector",
# - запустить скрипт
# "python frame_detector.py --rtsp-url rtsp://192.168.1.100:554 --uart-port COM3 --baudrate 921600 --bytesize 8"
