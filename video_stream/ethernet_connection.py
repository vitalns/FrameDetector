import cv2
import time


class EthernetVideoStream:
    """Класс для подключения к камере по Ethernet (RTSP)"""
    def __init__(self, rtsp_url, max_retries=0, retry_delay=1):
        """
        Инициализация подключения
        :param rtsp_url: URL видеопотока (rtsp://ip:port)
        :param max_retries: Максимальное количество попыток переподключения
        :param retry_delay: Задержка между попытками (в секундах)
        """
        self.rtsp_url = rtsp_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cap = None
        self. is_connected = False
        self.connect()

    def connect(self):
        """Установка соединения с камерой"""
        self.disconnect()  # Закрываем предыдущее подключение

        self.cap = cv2.VideoCapture(self.rtsp_url)
        if self.cap.isOpened():
            self.is_connected = True
            print(f"Connected to: {self.rtsp_url}")
        else:
            self.is_connected = False
            print(f"Connection failed")

    def disconnect(self):
        """Освобождение ресурсов"""
        if self.cap is not None:
            self.cap.release()
            self.is_connected = False

    def reconnect(self):
        """Процедура переподключения"""
        for i in range(self.max_retries):
            print(f"Attempt {i+1}/{self.max_retries}")
            self.connect()
            if self.is_connected:
                return True
            time.sleep(self.retry_delay)
        return False

    def read(self):
        """Чтение кадров из потока"""
        if not self.is_connected and not self.reconnect():
            return None, False

        success, frame = self.cap.read()

        if not success:
            self.is_connected = False
            if self.reconnect():
                success, frame = self.cap.read()
        return success, frame

    # Для поддержки контекстного менеджера
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# # Использование с контекстным менеджером
# if __name__ == "__main__":
#     rtsp_url = "rtsp://192.168.1.100:554"
#
#     with EthernetVideoStream(rtsp_url) as stream:
#         while True:
#             frame, success = stream.read()
#
#             if not success:
#                 print("Unable to receive frame")
#                 break
#
#             # Обработка кадра
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     cv2.destroyAllWindows()

# # Использование без контекстного менеджера
# if __name__ == "__main__":
#     rtsp_url = "rtsp://192.168.1.100:554"
#
#     stream = EthernetVideoStream(rtsp_url)
#     try:
#         while True:
#             frame, success = stream.read()
#
#             if not success:
#                 print("Unable to receive frame")
#                 break
#
#             # Обработка кадра
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#     finally:
#         stream.disconnect()  # Приходиться закрывать вручную
#
#     cv2.destroyAllWindows()