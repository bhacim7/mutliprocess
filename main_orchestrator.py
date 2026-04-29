import multiprocessing as mp
import time
import sys
import signal


from core.nav_process import nav_worker
from core.telem_process import telem_worker
from core.camera_process import camera_worker

def signal_handler(sig, frame):
    """İşletim sisteminden gelen kill (sonlandırma) sinyallerini yakalar."""
    print("\n[YÖNETİCİ] Kapatma sinyali alındı. Sistem güvenliğe alınıyor...")
    sys.exit(0)


def main():
    print("=" * 50)
    print("🚀 RoboBoat 2026 - IDA System Orchestrator Başlatılıyor...")
    print("=" * 50)

    # İşletim sistemi sinyallerini dinle (Ctrl+C için)
    signal.signal(signal.SIGINT, signal_handler)

    # 1. PAYLAŞILMIŞ BELLEK (SHARED STATE) OLUŞTURMA
    # Tüm proseslerin aynı anda okuyup/yazabileceği ana RAM bloğu
    manager = mp.Manager()
    shared_state = manager.dict({
        'magnetic_heading': 0.0,  # ZED kamerasından gelecek
        'gps_lat': 0.0,  # OrangeCube'dan gelecek
        'gps_lon': 0.0,  # OrangeCube'dan gelecek
        'current_task': 'TASK_1',  # Başlangıç görevimiz (Durum Makinesi)
        'mission_started': True,  # Görev aktif mi?
        'manual_mode': False,  # Yer istasyonu kontrolü
        'motor_pwm_left': 1500,  # Anlık sol motor gücü
        'motor_pwm_right': 1500,  # Anlık sağ motor gücü
        'shutdown': False  # Global acil durdurma ve çıkış bayrağı
    })

    # 2. KUYRUKLAR (QUEUES) OLUŞTURMA
    # Yer istasyonundan gelen komutları (örn: manual override, set_gps) NavProcess'e taşımak için
    command_queue = mp.Queue()

    # 3. PROSESLERİ TANIMLAMA
    # target= çalıştırılacak fonksiyon, args= fonksiyona gidecek parametreler
    processes = []


    p_nav = mp.Process(target=nav_worker, args=(shared_state, command_queue), name="NavProcess")
    p_telem = mp.Process(target=telem_worker, args=(shared_state, command_queue), name="TelemProcess")
    p_cam = mp.Process(target=camera_worker, args=(shared_state,), name="CameraProcess")

    processes.extend([p_nav, p_telem, p_cam])

    # 4. PROSESLERİ AYAĞA KALDIRMA
    print("[YÖNETİCİ] Prosesler ayağa kaldırılıyor...")
    for p in processes:
        p.start()
        print(f"[YÖNETİCİ] Başlatıldı: {p.name} (PID: {p.pid})")

    # 5. WATCHDOG (İZLEYİCİ) DÖNGÜSÜ
    # Ana thread burada bekler ve alt proseslerin çöküp çökmediğini kontrol eder.
    try:
        while True:
            # Eğer sistemin kapatılması istenmişse (Telemetri üzerinden vs.) döngüyü kır
            if shared_state['shutdown']:
                print("[YÖNETİCİ] Sistem kapatma komutu (Shutdown) algılandı!")
                break

            # TODO: İleride çöken prosesi tespit edip yeniden başlatma mantığı eklenebilir.
            time.sleep(1.0)

    except SystemExit:
        # Ctrl+C basıldığında buraya düşer
        shared_state['shutdown'] = True

    # 6. GÜVENLİ KAPANIŞ (GRACEFUL SHUTDOWN)
    print("\n[YÖNETİCİ] Tüm proseslere durma emri gönderildi, kapanmaları bekleniyor...")

    # Proseslerin işlerini bitirmesi için 3 saniye süre tanı
    for p in processes:
        if p.is_alive():
            p.join(timeout=3.0)
            if p.is_alive():
                print(f"[YÖNETİCİ] {p.name} kapanmayı reddetti, zorla kapatılıyor (SIGTERM)!")
                p.terminate()

    print("[YÖNETİCİ] Sistem tamamen kapandı. İyi günler Kaptan.")


if __name__ == '__main__':
    # İşletim sistemine (Ubuntu/Linux) multiprocessing metodunu belirtiyoruz.
    # Jetson için 'spawn' metodu CUDA ve donanım kaynakları çakışmalarını önler.
    mp.set_start_method('spawn')
    main()