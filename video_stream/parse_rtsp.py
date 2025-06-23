import argparse


class get_RTSP:
    """Парсинг командной строки и формирование протокола RTSP для подключения к видеопотоку"""
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def generate_rtsp_url(self) -> str:
        """Генерация RTSP URL"""
        rtsp_url = f"rtsp://{self.ip}:{self.port}"
        return rtsp_url

    @classmethod
    def parse_command_line(cls):
        """Создаем зкземпляр из аргументов командной строки"""
        parser = argparse.ArgumentParser(description="Parse RTSP URL for camera")
        parser.add_argument('--ip', type=str, required=True, help='Camera IP address')
        parser.add_argument('--port', type=int, default=554, help='RTSP port')
        args = parser.parse_args()
        return cls(args.ip, args.port)

# Вызов класса
# if __name__ == "__main__":
#     rtsp_url = get_RTSP.parse_command_line().generate_rtsp_url()
#     print(f"Connecting to: {rtsp_url}")

# С одной стороны удобно использовать класс get_RTSP как точку входа,
# а с другой усложняет код и возможно просто обойтись функцией

# Использование функции вместо класса
# def parse_args() -> str:
#     """Парсинг аргументов командной строки"""
#     parser = argparse.ArgumentParser(description="Parse RTSP URL for camera")
#     parser.add_argument('--ip', type=str, required=True, help='Camera IP address')
#     parser.add_argument('--port', type=int, default=554, help='RTSP port')
#     args = parser.parse_args()
#     rtsp_url = f"rtsp://{args.ip}:{args.port}"  # Генерация RTSP URL
#     return rtsp_url

# if __name__ == "__main__":
#     rtsp_url = parse_args()
#     print(f"Connecting to: {rtsp_url}")

# Запуск из командной строки cmd:
# Перейти в папку проекта "cd C:\Users\nazarov_vs\PycharmProjects\test",
# и запустить скрипт "python test_video_stream.py --ip 192.168.1.100 --port 554".
# Возможно потребуется активировать вертуальное окружение PyCharm: ".\venv\Scripts\activate"
# Для активации вертуального окружения Anaconda: "conda activate base" (Список доступных окружений "conda env list")

# При возникновении ошибок:
# Проверить доступность Python из cmd и версию: "python --version";
# Проверить доступность Anaconda из cmd: "conda --version";
# Проверить добавлен ли Python и Anaconda в системный PATH;
# Запустить скрипт указав полный путь к Python, например -
# "С:\ProgramData\anaconda3\python.exe test_video_stream.py --ip 192.168.1.100 --port 554"
