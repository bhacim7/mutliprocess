import socket
import struct
import cv2
import numpy as np
import threading
import queue

# Dictionary to hold the latest frame for each client address
# Format: { "IP:PORT": (frame, window_name) }
shared_frames = {}
shared_frames_lock = threading.Lock()
running = True

def client_handler(conn, addr):
    """
    Her istemci için ayrı çalışacak olan fonksiyon.
    Sadece ağ üzerinden veriyi alır ve shared_frames sözlüğüne koyar.
    """
    print(f"[INFO] {addr} için yeni iş parçacığı başlatıldı.")
    payload_size = struct.calcsize("!Q")
    data = b""
    window_name = f"Kamera - {addr[0]}:{addr[1]}"

    try:
        while running:
            # 1. Mesaj boyutunu al
            while len(data) < payload_size:
                packet = conn.recv(65536)
                if not packet: break
                data += packet

            if len(data) < payload_size:
                break

            packed_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("!Q", packed_size)[0]

            # 2. Asıl frame verisini al
            while len(data) < msg_size:
                packet = conn.recv(65536)
                if not packet: break
                data += packet

            if len(data) < msg_size:
                break

            frame_data = data[:msg_size]
            data = data[msg_size:]

            try:
                # Veriyi çöz
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # GUI thread'i (ana thread) için frame'i paylaş
                    with shared_frames_lock:
                        shared_frames[addr] = (frame, window_name)

            except Exception as e:
                print(f"[HATA] {addr} Frame işleme hatası: {e}")
                break

    except socket.error as e:
        print(f"[HATA] {addr} Soket hatası: {e}")
    finally:
        conn.close()
        # Bağlantı koptuğunda sözlükten çıkar
        with shared_frames_lock:
            if addr in shared_frames:
                del shared_frames[addr]
        print(f"[INFO] {addr} bağlantısı sonlandırıldı.")


def accept_connections(server_socket):
    """
    Arka planda sadece yeni bağlantıları kabul eden thread.
    """
    try:
        while running:
            # Timeout ekliyoruz ki kapatma sinyalini (running=False) yakalayabilelim
            server_socket.settimeout(1.0)
            try:
                conn, addr = server_socket.accept()
                # Timeout'u client için kapatıyoruz
                conn.settimeout(None)
                print(f"[INFO] Bağlantı kabul edildi: {addr}")

                t = threading.Thread(target=client_handler, args=(conn, addr))
                t.daemon = True
                t.start()
            except socket.timeout:
                continue
    except Exception as e:
        if running:
            print(f"[HATA] Sunucu accept hatası: {e}")


# --- ANA SUNUCU KURULUMU ---
def start_server():
    global running
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", 5000))
    server_socket.listen(5)

    print("[INFO] Sunucu çoklu bağlantı için hazır (Port 5000)...")

    # Bağlantı kabul etme işini arka plan thread'ine ver
    accept_thread = threading.Thread(target=accept_connections, args=(server_socket,))
    accept_thread.daemon = True
    accept_thread.start()

    active_windows = set()

    try:
        # ANA THREAD: Sadece GUI çizimi yapar.
        # Bu, OpenCV'nin thread-locking veya waitKey stuttering yapmasını engeller.
        while running:
            frames_to_draw = []

            with shared_frames_lock:
                for addr, (frame, window_name) in list(shared_frames.items()):
                    frames_to_draw.append((window_name, frame))
                    active_windows.add(window_name)

            # Eğer bağlantı kopmuşsa ve sözlükten silinmişse pencereleri kapat
            current_clients = set()
            with shared_frames_lock:
                for addr, (frame, window_name) in shared_frames.items():
                    current_clients.add(window_name)

            windows_to_close = active_windows - current_clients
            for win in windows_to_close:
                try:
                    cv2.destroyWindow(win)
                except:
                    pass
                active_windows.remove(win)

            # Çizim işlemi
            for window_name, frame in frames_to_draw:
                cv2.imshow(window_name, frame)

            # waitKey SADECE ana thread'de çağrılır.
            if cv2.waitKey(1) == 27:  # ESC tuşu
                running = False
                break

    except KeyboardInterrupt:
        print("\n[INFO] Sunucu kapatılıyor...")
        running = False
    finally:
        server_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_server()