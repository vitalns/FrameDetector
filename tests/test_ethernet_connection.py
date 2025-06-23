import cv2
from video_stream.ethernet_connection import EthernetVideoStream
from video_stream.parse_rtsp import get_RTSP

if __name__ == "__main__":
    rtsp_url = "rtsp://192.168.1.100:554"
    # camera = get_RTSP.parse_command_line()
    # rtsp_url = camera.generate_rtsp_url()
    print(f"Connecting to: {rtsp_url}")

    with EthernetVideoStream(rtsp_url) as stream:
        while True:
            frame, success = stream.read()

            if not success:
                print("Unable to receive frame")
                break

            # Обработка кадра

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()