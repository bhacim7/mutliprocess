import socket
import struct
import pickle
import cv2
import threading  # Threading kütüphanesi eklendi


def client_handler(conn, addr):
    """
    Her istemci için ayrı çalışacak olan fonksiyon.
    """
    print(f"[INFO] {addr} için yeni iş parçacığı başlatıldı.")
    payload_size = struct.calcsize("!Q")
    data = b""

    try:
        while True:
            # 1. Mesaj boyutunu al
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet

            if len(data) < payload_size:
                break

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("!Q", packed_size)[0]

            # 2. Asıl frame verisini al
            while len(data) < msg_size:
                packet = conn.recv(4096)
                if not packet: break
                data += packet

            if len(data) < msg_size:
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            try:
                # Veriyi çöz ve görüntüle
                buffer = pickle.loads(frame_data)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

                if frame is not None:
                    # DİKKAT: Her istemcinin pencere adı farklı olmalı!
                    # IP ve Port bilgisini pencere adı yapıyoruz.
                    window_name = f"Kamera - {addr[0]}:{addr[1]}"
                    cv2.imshow(window_name, frame)

                # Her thread kendi penceresi için waitKey kontrolü yapar
                if cv2.waitKey(1) == 27:  # ESC tuşu
                    break

            except Exception as e:
                print(f"[HATA] {addr} Frame işleme hatası: {e}")
                break

    except socket.error as e:
        print(f"[HATA] {addr} Soket hatası: {e}")
    finally:
        conn.close()
        # Pencereyi kapat (sadece bu istemciye ait olanı kapatmak zor olduğu için try-except)
        try:
            cv2.destroyWindow(f"Kamera - {addr[0]}:{addr[1]}")
        except:
            pass
        print(f"[INFO] {addr} bağlantısı sonlandırıldı.")


# --- ANA SUNUCU KURULUMU ---
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", 5000))
    server_socket.listen(5)

    print("[INFO] Sunucu çoklu bağlantı için hazır (Port 5000)...")

    try:
        while True:
            # Ana döngü sadece bağlantı kabul eder
            conn, addr = server_socket.accept()
            print(f"[INFO] Bağlantı kabul edildi: {addr}")

            # Her bağlantı için yeni bir Thread başlat
            t = threading.Thread(target=client_handler, args=(conn, addr))
            t.daemon = True  # Ana program kapanırsa thread'ler de kapansın
            t.start()

    except KeyboardInterrupt:
        print("\n[INFO] Sunucu kapatılıyor...")
    finally:
        server_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_server()