import serial
from pymavlink.dialects.v20 import common as mavlink2

class UARTMavlinkClient:
    """Класс для отправки данных по UART в формате MAVLink"""
    def __init__(self, port, baudrate, bytesize, parity, stopbits, mavlink_version=2, message_id=250):
        """
        Инициализация UART и MAVLink

        :param port: Порт UART (например '/dev/ttyUSB0')
        :param baudrate: Скорость передачи (например 921600)
        :param bytesize: Размер данных (5, 6, 7, 8)
        :param parity: Контроль четности ('none', 'even', 'odd')
        :param stopbits: Стоп-биты (1, 1.5, 2)
        :param mavlink_version: Версия MAVLink (1 или 2)
        :param message_id: ID кастомного сообщения (180-255)
        """
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.mavlink_version = mavlink_version
        self.message_id = message_id

        # Преобразование параметров для pyserial
        self.parity_mapping = {
            'none': serial.PARITY_NONE,
            'even': serial.PARITY_EVEN,
            'odd': serial.PARITY_ODD
        }
        self.bytesize_mapping = {
            5: serial.FIVEBITS,
            6: serial.SIXBITS,
            7: serial.SEVENBITS,
            8: serial.EIGHTBITS
        }

        self.ser = None
        self.mav = None

    def connect(self):
        """Установка соединения по UART и инициализация MAVLink"""
        try:
            # Подключение UART
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize_mapping[self.bytesize],
                parity=self.parity_mapping[self.parity],
                stopbits=self.stopbits,
                timeout=1
            )

            # Инициализация MAVLink
            self.mav = mavlink2.MAVLink(None)

            # Создание кастомного сообщения
            class MAVLink_contour_cord_message(mavlink2.MAVLink_message):
                id = self.message_id
                name = 'CONTOUR_CORD'
                fieldnames = ['time_boot_ms', 'x', 'y', 'w', 'h']
                format = 'IiiII'  # uint32_t, int32_t, int32_t, uint32_t, uint32_t
                crc_extra = 150

            # Регистрация сообщения
            mavlink2.MAVLink_message_classes[self.message_id] = MAVLink_contour_cord_message

            print(f"UART подключен: {self.port}, {self.baudrate} бод, "
                  f"{self.bytesize} бит, {self.parity} четность, {self.stopbits} стоп-бит")
            print(f"MAVLink настроен: версия {self.mavlink_version}, ID сообщения {self.message_id}")

        except Exception as e:
            print(f"Ошибка инициализации UART/MAVLink: {e}")
            # raise

    def send_contour_data(self, x, y, w, h):
        """Отправка данных о контуре через MAVLink"""
        try:
            time_boot_ms = int(time.time() * 1000) % (2 ** 32)
            msg = mavlink2.MAVLink_message_classes[self.message_id](
                time_boot_ms, x, y, w, h
            )
            packet = msg.pack(self.mav, force_mavlink1=(self.mavlink_version == 1))
            self.ser.write(packet)

        except Exception as e:
            print(f"Ошибка отправки данных: {e}")
            # raise

    def disconnect(self):
        """Закрытие соединения"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        print("Соединение UART закрыто")
