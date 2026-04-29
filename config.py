import os

# =============================================================================
# 1. DONANIM VE HABERLEŞME AYARLARI
# =============================================================================
# OrangeCube (Pixhawk) Bağlantı Ayarları
ORANGE_PORT = os.getenv("ORANGE_PORT", "/dev/ttyACM0")
ORANGE_BAUD = int(os.getenv("ORANGE_BAUD", "57600"))

# Yer İstasyonu (Telemetri) Bağlantı Ayarları
TELEM_PORT = os.getenv("TELEM_PORT", "/dev/ttyUSB0")
TELEM_BAUD = int(os.getenv("TELEM_BAUD", "57600"))

# OrangeCube / Pixhawk Servo Çıkış Pinleri
SOL_MOTOR = 1
SAG_MOTOR = 3

# Röle ve Güç Yönetimi
MOTOR_RELAY_PIN = 15
ESC_INIT_DELAY = 3.0  # ESC'lerin dıt-dıt sesini bitirmesi için bekleme süresi

# =============================================================================
# 2. MOTOR VE SÜRÜŞ (PWM) AYARLARI
# =============================================================================
# T200 vb. thrusterlar için standart PWM aralıkları
BASE_PWM = 1500        # Motorlar duruyor
MIN_PWM_LIMIT = 1100   # Maksimum geri
MAX_PWM_LIMIT = 1900   # Maksimum ileri

# İleri Sürüş Hızları (1500 + CRUISE_PWM olarak uygulanır)
CRUISE_PWM = 100       # Standart seyir hızı (1600 PWM)

# Navigasyon Dönüş Katsayıları (P-Kontrolcü için)
Kp_HEADING = 1.5       # Hedef açıyla aramızdaki farkı PWM'e çeviren katsayı
MAX_TURN_PWM = 150     # Dönüşlerde uygulanacak maksimum eksi/artı PWM farkı

# Görev Geçiş Toleransı
WAYPOINT_TOLERANCE_M = 2.0  # Hedefe kaç metre kala "Görev Tamamlandı" sayılacak?

# =============================================================================
# 3. GÖREV (WAYPOINT) KOORDİNATLARI
# =============================================================================
# RoboBoat 2026 parkuru için 6 ardışık hedef noktası.
# State Machine sırayla bu noktalara gidecek.

# GÖREV 1: Giriş Kapısı
T1_LAT = 40.8630501
T1_LON = 29.2599517

# GÖREV 2: Kapı Ortası
T2_LAT = 40.8630000  # Örnek ara nokta
T2_LON = 29.2599300

# GÖREV 3: Çıkış Kapısı
T3_LAT = 40.8629223
T3_LON = 29.2599123

# GÖREV 4: Engel Sahası (Debris) Girişi
T4_LAT = 40.8091600
T4_LON = 29.2619150

# GÖREV 5: Engel Sahası Ortası
T5_LAT = 40.8090500  # Örnek ara nokta
T5_LON = 29.2619200

# GÖREV 6: Engel Sahası Bitişi
T6_LAT = 40.8089552
T6_LON = 29.2619292

# =============================================================================
# 4. GÖREV DURUM MAKİNESİ (STATE MACHINE) HARİTASI
# =============================================================================
# nav_process.py bu sözlüğü kullanarak hangi görevde hangi hedefe gideceğini bilecek.
TASK_WAYPOINTS = {
    'TASK_1': (T1_LAT, T1_LON),
    'TASK_2': (T2_LAT, T2_LON),
    'TASK_3': (T3_LAT, T3_LON),
    'TASK_4': (T4_LAT, T4_LON),
    'TASK_5': (T5_LAT, T5_LON),
    'TASK_6': (T6_LAT, T6_LON)
}