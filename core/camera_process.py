import time
import sys
import os
import pyzed.sl as sl

# Ana dizinden utils klasörüne erişim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import kamera
from utils.utilities import KalmanFilter
from utils.kamera import TimestampHandler


def camera_worker(shared_state):
    """
    Kamera ve Sensör (Göz/Denge) Prosesi.
    ZED2i kamerasını başlatır, IMU/Manyetometre verilerini okur,
    Kalman filtresinden geçirir ve paylaşımlı belleğe yazar.
    """
    print("[CAMERA_PROCESS] Başlatılıyor...")

    # 1. ZED KAMERA VE TAKİP SİSTEMİNİ BAŞLAT
    try:
        zed = kamera.initialize_camera()

        # Orijinal kodundaki konumsal takip başlatma fonksiyonunu çağırıyoruz
        kamera.initialize_positional_tracking(zed)
        print("[CAMERA_PROCESS] ZED Kamera ve Konumsal Takip (Positional Tracking) başlatıldı!")

    except Exception as e:
        print(f"[CAMERA_PROCESS] KRİTİK HATA! ZED Kamera açılamadı: {e}")
        shared_state['shutdown'] = True
        return

    # 2. YARDIMCI SINIFLAR (Filtreler ve Zamanlayıcılar)
    ts_handler = TimestampHandler()

    # Orijinal kodundaki Kalman Filtresi nesnesi (process ve measurement varyansları ile)
    magnetic_filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)

    sensors_data = sl.SensorsData()

    print("[CAMERA_PROCESS] Sensör okuma döngüsü başlıyor...")

    # 3. SENSÖR OKUMA DÖNGÜSÜ
    # ZED IMU verileri çok yüksek hızlarda (örneğin 400Hz) akabilir.
    while not shared_state['shutdown']:
        start_time = time.time()

        # Sensör verilerini anlık (CURRENT) olarak çek
        if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:

            # Veri yeniyse işle (TimestampHandler ile kontrol)
            if ts_handler.is_new(sensors_data.get_imu_data()):
                # ZED'den ham manyetik pusula verisini al
                raw_heading = sensors_data.get_magnetometer_data().magnetic_heading

                # Kalman filtresinden geçirerek yumuşat
                filtered_heading = magnetic_filter.update(raw_heading)

                # Manyetik sapma vb. için ufak düzeltme (Orijinal kodundaki +6 derece vb. ayarlar)
                final_heading = (filtered_heading - 6) % 360

                # Paylaşımlı belleğe anında yaz (nav_process buradan okuyacak)
                shared_state['magnetic_heading'] = final_heading

        # İşlemciyi (Orin'i) %100 kullanmamak için çok ufak bir uyku
        # (Sensör polleme hızını ~100Hz civarında tutmak idealdir)
        elapsed = time.time() - start_time
        sleep_time = max(0.001, 0.01 - elapsed)
        time.sleep(sleep_time)

    # 4. GÜVENLİ KAPANIŞ
    print("[CAMERA_PROCESS] Kapanış sinyali alındı. ZED SDK güvenlice kapatılıyor...")
    try:
        kamera.temiz_kapat(zed)
    except Exception as e:
        print(f"[CAMERA_PROCESS] ZED Kapatma Hatası: {e}")

    print("[CAMERA_PROCESS] Kapandı.")