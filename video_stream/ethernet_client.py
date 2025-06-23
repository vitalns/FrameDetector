import cv2


class EthernetClient:
    """Класс для подключения к камере по Ethernet (RTSP)"""

    def __init__(self, rtsp_url: str):
        """
        Инициализация подключения

        :param rtsp_url: Полный RTSP URL (например: rtsp://192.168.1.100:554)
        """

        self.rtsp_url = rtsp_url
        self.cap = None
        self.is_connected = False

    def connect(self):
        """Установка соединения с камерой"""
        try:
            if not self.rtsp_url:
                raise ValueError("RTSP URL не указан")
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                self.is_connected = False
                raise ConnectionError(f"Не удалось подключиться к RTSP: {self.rtsp_url}")
            self.is_connected = True
            print(f"Подключено к RTSP: {self.rtsp_url}")

        except Exception as e:
            print(f"Ошибка подключения: {e}")
            # raise  # Остановка выполнения

    def get_frame(self):
        """Получение кадра с камеры"""
        try:
            ret, frame = self.cap.read()
            if not ret:
                raise ConnectionError("Ошибка чтения кадра RTSP")
            return ret, frame

        except Exception as e:
            print(f"Ошибка получения кадра: {e}")
            # raise  # Остановка выполнения

    def disconnect(self):
        """Закрытие соединения"""
        if self.cap is not None:
            self.cap.release()
            self.is_connected = False
        print("Соединение с камерой закрыто")
