import time
import sys
import os
import random

try:
    from rplidar import RPLidar
except ImportError:
    RPLidar = None
import config as cfg


def lidar_worker(shared_state):
    """
    Lidar Process: Reads from LIDAR, downsamples data to avoid IPC freezing,
    and updates the shared memory.
    """
    print("[LIDAR_PROCESS] Başlatılıyor...")

    # 1. INIT LIDAR
    lidar = None
    if RPLidar is not None:
        try:
            LIDAR_PORT_NAME = getattr(cfg, 'LIDAR_PORT', '/dev/ttyUSB1')
            baud = getattr(cfg, 'LIDAR_BAUDRATE', 1000000)
            lidar = RPLidar(LIDAR_PORT_NAME, baudrate=baud, timeout=3)
            print("[LIDAR_PROCESS] Lidar bağlantısı başarılı.")
            lidar.start_motor()
        except Exception as e:
            print(f"[LIDAR_PROCESS] LIDAR başlatılamadı: {e}. MOCK moduna geçiliyor.")
            lidar = None
    else:
        print("[LIDAR_PROCESS] rplidar modülü bulunamadı, MOCK moduna geçiliyor.")

    # Initialize shared lidar array
    shared_state['lidar_map'] = []
    shared_state['lidar_emergency'] = False

    print("[LIDAR_PROCESS] Sensör okuma döngüsü başlıyor...")

    while not shared_state['shutdown']:
        start_time = time.time()

        emergency = False
        lidar_buckets = [10.0] * 72 # Varsayılan: Her yön boş (10m)

        if lidar is not None:
            try:
                # iter_scans yerine her 0.1 saniyede tamponlanmış okuma mantığı
                scan_iterator = lidar.iter_scans(max_buf_meas=500, min_len=5)
                # Sadece sıradaki yield olan *ilk* tam taramayı (1 frame) çekiyoruz, döngüyü bozmuyoruz.
                scan = next(scan_iterator)
                for quality, angle, distance in scan:
                    dist_m = distance / 1000.0
                    if dist_m > 0 and dist_m < 15.0:
                        bucket_idx = int(angle / 5) % 72
                        if dist_m < lidar_buckets[bucket_idx]:
                            lidar_buckets[bucket_idx] = dist_m
            except StopIteration:
                pass
            except Exception as e:
                print(f"[LIDAR_PROCESS] Veri okuma hatası: {e}. Lidar resetleniyor...")
                time.sleep(1)

            # Ön taraf kontrolü (Acil durum) - Merkez (-15 ile +15 derece)
            # Kovalar: 0,1,2 ve 69,70,71
            front_buckets = lidar_buckets[0:3] + lidar_buckets[69:72]
            lidar_limit = getattr(cfg, 'LIDAR_ACIL_DURMA_M', 1.2)
            if any(d < lidar_limit for d in front_buckets):
                emergency = True

        else:
            # Mock Data
            lidar_buckets = [round(random.uniform(0.5, 10.0), 2) for _ in range(72)]
            if any(d < 0.8 for d in lidar_buckets[0:3] + lidar_buckets[69:72]):
                 emergency = True

        shared_state['lidar_map'] = lidar_buckets
        shared_state['lidar_emergency'] = emergency

        # Loop frequency ~10Hz
        elapsed = time.time() - start_time
        sleep_time = max(0.001, 0.1 - elapsed)
        time.sleep(sleep_time)

    print("[LIDAR_PROCESS] Kapanış sinyali alındı. Lidar durduruluyor...")
    if lidar is not None:
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except:
            pass
    print("[LIDAR_PROCESS] Kapandı.")
