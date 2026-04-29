import time
import sys
import os
import datetime
import queue

# Proje ana dizininden modülleri çekebilmek için
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
# Senin orijinal telem.py dosyanı utils klasörüne taşıdığımızı varsayıyoruz
from utils.telem import TelemetrySender, CommandReceiver


def telem_worker(shared_state, command_queue):
    """
    Yer İstasyonu Haberleşme Prosesi.
    GCS'den gelen komutları dinler ve anlık robot durumunu (GPS, Görev, PWM) paketleyip gönderir.
    """
    print("[TELEM_PROCESS] Başlatılıyor...")

    # 1. HABERLEŞME SINIFLARINI BAŞLAT
    sender = TelemetrySender(cfg.TELEM_PORT, cfg.TELEM_BAUD)

    # Yerel dinleyici kuyruğu (CommandReceiver thread'i buraya yazacak)
    local_cmd_queue = queue.Queue()
    receiver = CommandReceiver(sender, local_cmd_queue)
    receiver.start()

    print("[TELEM_PROCESS] Haberleşme döngüsü başlıyor (10 Hz)...")

    # Yayın hızı (Hz)
    hz = 10
    period = 1.0 / hz
    my_id = 1  # İDA kimlik numarası

    # 2. ANA TELEMETRİ DÖNGÜSÜ
    while not shared_state['shutdown']:
        start_time = time.time()

        # A. YER İSTASYONUNDAN GELEN KOMUTLARI İŞLE
        try:
            while True:
                # Kuyruktaki tüm komutları beklemeden çek (Non-blocking)
                cmd = local_cmd_queue.get_nowait()
                command_str = cmd.get("cmd")

                # Acil Durdurma (Yazılımsal E-Stop)
                if command_str == "emergency_stop":
                    print("\n[TELEM_PROCESS] 🚨 YER İSTASYONUNDAN ACİL DURDURMA ALINDI! 🚨")
                    shared_state['shutdown'] = True

                # Otonomdan Manuel Moda Geçiş
                elif command_str == "set_manual":
                    val = bool(cmd.get("value"))
                    shared_state['manual_mode'] = val
                    print(f"[TELEM_PROCESS] Manuel Mod -> {val}")

                # Manuel Sürüş (Joystick) Komutları
                elif command_str == "manual_pwm":
                    if shared_state['manual_mode']:
                        # Nav_process bu değerleri okuyup OrangeCube'a basacak
                        shared_state['motor_pwm_left'] = int(cmd.get("left", 1500))
                        shared_state['motor_pwm_right'] = int(cmd.get("right", 1500))

                # Yeni GPS Noktası Ayarlama (Bunu orkestratör veya nav_process işlesin diye ana kuyruğa atıyoruz)
                elif command_str == "set_gps":
                    command_queue.put(cmd)
                    print(f"[TELEM_PROCESS] Yeni GPS komutu alındı, ana kuyruğa iletildi.")

        except queue.Empty:
            pass

        # B. YER İSTASYONUNA DURUM RAPORU (TELEMETRİ) GÖNDER
        # Paylaşımlı bellekten (shared_state) her zaman EN GÜNCEL veriyi okuruz.
        # Nav_process veya Camera_process ne kadar hızlı güncellerse güncellesin,
        # biz burada saniyede 10 kez o anki son durumu çekip yollarız.
        payload = {
            "id": my_id,
            "t_ms": datetime.datetime.now().strftime('%H:%M:%S'),
            "pwm_L": shared_state['motor_pwm_left'],
            "pwm_R": shared_state['motor_pwm_right'],
            "hdg": round(shared_state['magnetic_heading'], 1),
            "task": shared_state['current_task'],
            "mod": shared_state['manual_mode'],
            "MEVCUT_KONUM": {
                "lat": round(shared_state['gps_lat'], 6),
                "lon": round(shared_state['gps_lon'], 6)
            }
        }

        # İleride tespit edilen objeler (Şamandıra vs.) eklendiğinde payload'a "objects" listesi eklenecek.
        sender.send(payload)

        # C. DÖNGÜ ZAMANLAMASI
        elapsed = time.time() - start_time
        sleep_time = max(0.001, period - elapsed)
        time.sleep(sleep_time)

    # GÜVENLİ KAPANIŞ
    print("[TELEM_PROCESS] Kapanış sinyali alındı. Portlar ve threadler kapatılıyor...")
    receiver.stop()
    sender.close()
    print("[TELEM_PROCESS] Kapandı.")