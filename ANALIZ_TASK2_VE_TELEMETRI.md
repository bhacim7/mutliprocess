# IDA USV — Task 2 Manevra Bozukluğu ve Telemetri Kaybı Analizi

**Tarih:** 2026-08-11
**Kaynak veriler:** `task2Claude.mp4` (66,99 s · 1064×698 · 30 fps · 2009 kare), `final_costmap.png`,
tekne terminal logu, RFD900x ayar ekran görüntüsü, klasördeki kod tabanı (değiştirilmedi).

**İnceleme yöntemi:** Video 2 fps'te bütün olarak, kritik pencereler (23:47:57–23:48:15) 4–5 fps'te
kare kare incelendi. HUD'daki her sayısal alan (`HIZ`, `HDG`, `HEDEF_HDG`, `ACI_FARKI`,
`KONTROL_HATA`, `HEDEFE_MESAFE`, `IDA_KONUM`, `HEDEF_KONUM`, 5 kanal PWM, `FPS`) okunup
kod satırlarıyla eşleştirildi.

> **DURUM (2026-08-11):** Bölüm 6'daki **11 manevra maddesi** ve bölüm 7'deki
> **T1–T4 telemetri maddeleri** kod tabanına **uygulandı**. Uygulama sonuçları ve ölçülen
> doğrulama değerleri için → [Bölüm 10](#10-uygulama-durumu-ve-doğrulama-sonuçları).
> **T5** (GCS dönüşümlü polling) ve **12/13/14** (turuncu sınır kısıtı, pusula doğrulama,
> derinlik eleme) kullanıcı kararıyla **ertelendi**.

---

## İÇİNDEKİLER

1. [Yönetici özeti](#1-yönetici-özeti)
2. [Videodan kare kare kanıtlar](#2-videodan-kare-kare-kanıtlar)
3. [Manevra bozukluğunun kök nedenleri (A–G)](#3-manevra-bozukluğunun-kök-nedenleri)
4. [final_costmap değerlendirmesi](#4-final_costmap-değerlendirmesi)
5. [Telemetri kaybının kök nedeni](#5-telemetri-kaybının-kök-nedeni)
6. [Manevra düzeltmeleri — 11 madde, detaylı](#6-manevra-düzeltmeleri--11-madde-detaylı)
7. [Telemetri düzeltmeleri — T1–T5 + RF](#7-telemetri-düzeltmeleri)
8. [Ek bulgular](#8-ek-bulgular)
9. [Uygulama sırası ve doğrulama planı](#9-uygulama-sırası-ve-doğrulama-planı)

---

## 1. YÖNETİCİ ÖZETİ

### Manevra
Tekne **engel görmediği sürece kusursuz** gidiyor (düzeltmeler ±15 PWM). Şamandıralar
costmap'e girip A\* devreye girdiği anda kontrol **bang-bang doyuma** giriyor ve tekne
360°'den fazla dönerek daire çiziyor.

Tek cümlelik sebep:
> Pure Pursuit kazancı aktüatör aralığına göre ~4 kat yüksek, ileri bakış mesafesi
> kalıcı olarak 0,8 m'ye çakılı, ve `find_lookahead_point` yoldan sapıldığında
> teknenin **yanındaki/arkasındaki** yol düğümüne yapışıyor. Üçü birleşince
> 27°'lik bir hata bile pervaneleri tam doyuma sokuyor.

Belirleyici kanıt — **23:47:59**:
```
ACI_FARKI: 4          <- hedef tam burnunun ucunda
ARKA_SOL_PWM: 1100    <- tam sol doyum
ARKA_SAG_PWM: 1900
DUMEN_PWM:    1100
```
Hedef 4° ileride ama komut tam sol. Bu ancak Pure Pursuit'in iç hatasının ≈ −73° olmasıyla
mümkün. Aradaki **77°'lik fark hedeften değil, yapışılan yol düğümünden** geliyor.

### Telemetri
RF'te, menzilde veya telsiz ayarlarında sorun **yok**. Tek sebep:

> `write_timeout = 0.5 s` @ 57600 baud = **2880 bayt tavan**.
> Paketiniz 10 nesnede **2960 bayt** oluyor → `ser.write()` yarıda kesiliyor →
> `\n` gitmiyor → GCS `json.loads`'ta patlatıp **sessizce çöpe atıyor**.

Komutların (40–500 B) sorunsuz gitmesi, telemetrinin (3–7 KB) hiç gelmemesi bu teşhisin
ispatıdır: RF sorunu olsaydı iki yön de bozulurdu.

---

## 2. VIDEODAN KARE KARE KANITLAR

### 2.1 Zaman çizelgesi

| Zaman | Durum | Gözlem |
|---|---|---|
| 23:47:32–23:47:45 | `TASK2_START` | **Sakin.** `SOL 1672 / SAG 1687`, `DUMEN 1488–1520`. `ACI_FARKI −1…−2`. Mesafe 18,0 → 3,7 m. **Kadrajda hiç tespit yok.** |
| 23:47:45 | → `TASK2_GO_TO_MID` | Hedef `40.809401, 29.262145`, mesafe 25,6 m. Hâlâ sakin. |
| 23:47:46–47 | — | `IDA_KONUM` ~1,5 s boyunca `40.809610, 29.262216`'da **donuk**; `HEDEFE_MESAFE` 24,0'da takılı. PWM'ler de birebir tekrar ediyor. |
| 23:47:57 | — | İlk şamandıralar kadraja giriyor: `Yellow 3.3m`, `Yellow 8.3m`, `Orange 9.3m`. |
| **23:47:59** | **kırılma anı** | `ACI_FARKI 4` iken `1100/1900`, `DUMEN 1100` → **tam sol doyum** |
| 23:47:59–23:48:03 | 1. yarım tur | HDG 156→141→133→122→105→46→20→15→357. `ACI_FARKI` +4→+17→+26→+41→+58→+122→+153. Komut hep `1100/1900`. **Tekne hedeften aktif olarak kaçıyor.** |
| 23:48:03–23:48:08 | 2. yarım tur | HDG 269→254→243→229→216→190→181→165→141. `HEDEF_HDG 182` iken 141'e iniyor → **41° aşım**. Toplam dönüş **> 360°**. |
| 23:48:08 | bang-bang | Aşağıdaki tablo |
| 23:48:20–22 | `TASK2_GO_TO_END` | `ACI_FARKI 58`, `1900/1100`, `DUMEN 1900`. Tam sağ komutuna rağmen HDG 167→142'ye düşmeye devam (≈1 s tekne ataleti). |
| 23:48:30–32 | — | `Yellow 6.8 → 6.1 → 5.4 → 4.8 m` kadraj ortasında. `ACI_FARKI −17→−26` yavaşça kayıyor. |

### 2.2 Bang-bang ölçümü (23:48:08, 0,25 s aralıkla)

| Zaman | ACI_FARKI | ARKA_SOL | ARKA_SAG | DUMEN |
|---|---|---|---|---|
| 23:48:08 | 48 | **1100** | **1900** | **1100** |
| +0,25 s | 62 | 1831 | 1528 | 1602 |
| +0,50 s | 71 | **1900** | **1109** | **1893** |

Çeyrek saniyede her pervanede **~800 PWM ters yönde sıçrama**. Hiçbir hız sınırlaması yok.
`MAX_PWM_CHANGE = 60` config'de tanımlı ama **kodun hiçbir yerinde kullanılmıyor**
(grep sonucu: yalnızca `config.py:63`).

### 2.3 Doyum eşiği hesabı

`BASE_PWM 1500 + CRUISE_PWM 180 = 1680`
`diff_multiplier = 1 + (1850 − 1680)/350 = 1,486`  (`nav_process.py:54`)

| Aktüatör | Doyum koşulu | `correction` | `KP=5.5` ile **hata** |
|---|---|---|---|
| Arka pervaneler (1900 tavan) | `diff ≥ 220` | ≥ 148 | **≥ 26,9°** |
| Arka pervaneler (1100 taban) | `diff ≤ 580` | ≥ 390 | **≥ 70,9°** |
| Dümen servosu | `\|corr\| ≥ 400` | ≥ 400 | **≥ 72,7°** |

Ölçülen `1100/1900` değerleri ≈ 71° iç hataya karşılık geliyor — tablo ile birebir uyumlu.

**Karşılaştırma:** Task 1/3'te kullanılan `DIRECT_DRIVE_KP = 1.5` ile aynı doyum 98,7°'de
başlıyor. **Task 2, diğer görevlerden 3,7 kat agresif.** "Sadece Task 2'de saçmalıyor"
gözleminin sayısal karşılığı budur.

### 2.4 Diğer HUD gözlemleri

- **`HIZ: 0.0`** — videonun tamamında, istisnasız.
- **`FPS`** — 30 → 25 → 21 → 19 → **17**. En düşük değerler (23:47:59–23:48:01)
  dairenin başladığı anla **birebir çakışıyor**. CPU doyduğunda sistem kararsızlaşıyor.
- **Aynı şamandıra için tutarsız mesafeler**: aynı kadrajda `Orange 9.3m / 11.4m / 12.2m / 13.5m`;
  23:48:03'te `Orange 11.4m`, 23:48:05'te `Orange 3.3m`.

---

## 3. MANEVRA BOZUKLUĞUNUN KÖK NEDENLERİ

### A. Pure Pursuit kazancı aktüatör aralığına göre ~4 kat yüksek — *birincil*

27°'den büyük her sapma arka pervaneleri tam doyuma sokuyor (§2.3). Doyumdaki bir
kontrolcü artık orantılı değildir; aç/kapa (bang-bang) davranır ve tekne ataletiyle
birleşince **kaçınılmaz olarak limit-çevrimi** üretir.

📍 `config.py:78-79`, `utils/planner.py:459-466`

### B. Lookahead kalıcı olarak 0,8 m'ye çakılı

`MainSystem2.py:112` → `get_horizontal_speed()` **`LOCAL_POSITION_NED`** mesajını okuyor.
ArduRover bunu varsayılan stream setinde yayınlamıyor → hep `0.0` → HUD'daki `HIZ: 0.0`.

Sonuç (`planner.py:434`):
```python
lookahead_dist = clip(0.0 * 1.0, 0.8, 1.5) = 0.8 m
```
1,8 m/s giden bir teknede 0,8 m ileri bakmak **0,45 saniyelik ufuk** demek.
Hız-uyarlamalı davranış tamamen ölü.

📍 `hardware/MainSystem2.py:110-117`

### C. `find_lookahead_point` yol düğümüne yapışıyor — *dairenin asıl mekaniği*

```python
# utils/planner.py:399-400
if min_d2 > lookahead_dist * lookahead_dist:
    return path[closest_idx], closest_idx
```

`string_pulling` sonrası düğümler metrelerce aralıklı. Lookahead 0,8 m'ye çakılı olduğu için
tekne yoldan 0,8 m'den fazla saptığı anda — ki bu neredeyse her döngüde oluyor — çemberle
kesişim yerine **"en yakın düğüm"** döndürülüyor. O düğüm teknenin yanında, hatta arkasında
olabilir → anlık 90–180° hata → tam doyum → tekne dönmeye başlıyor.

Ayrıca `start_idx` tekdüze artırıldığı için bu dal son düğümü seçerse, PP ömrünün geri
kalanında **yolun sonuna beeline yapar** ve kaçınmak istediği engelin içinden geçer.

Bu, **23:47:59'daki "hedef 4° önde ama komut tam sol"** çelişkisinin tek makul açıklamasıdır.

📍 `utils/planner.py:369-409`

### D. Engel şişirmesi konum gürültüsünden küçük → kaçınma yönü rastgele

| Bileşen | Değer |
|---|---|
| Engel dairesi (`cv2.circle(..., 6, 0, -1)`) | 0,6 m |
| Şişirme `(0.4 + 0.1)/0.10 = 5 px` | 0,5 m |
| **Toplam açıklık** | **1,1 m** |

Şamandıra konumu `GPS + magnetic_heading + ZED derinlik` ile hesaplanıyor; videodaki
saçılma **1–2 m** mertebesinde. **Konum belirsizliği açıklıktan büyük olunca A\*'ın
"soldan mı sağdan mı" kararı her replanda (~0,1–0,3 s) işareti değiştiriyor.**
23:48:08'deki 0,25 saniyelik tam sol → tam sağ dönüşü tam olarak budur.

> `planner.py:147`'deki yorum hâlâ *"2.1 m of inflation"* diyor — config artık 0,5 m
> veriyor. Yorum bayat, yanıltıcı.

📍 `core/nav_process.py:414-421`, `utils/planner.py:43-48`, `config.py:14-15`

### E. A\* zaman bütçesi çok dar → bayat yol takip ediliyor

`time_budget_s = 0.06` ile saf-Python A\* ancak ~1000–2000 düğüm açabilir. 400×400 ızgarada
engel etrafında bu sık sık yetmez ve `None` döner.

```python
# core/nav_process.py:905
if new_path and len(new_path) >= 2:
    current_path = new_path
```
`None` dönünce **eski yol temizlenmiyor**; tekne saniyeler önce başka konumdan planlanmış
yolu takip etmeye devam ediyor, `path_progress_idx` de sıfırlanmıyor. (C) maddesiyle
birleşince tekne kendi arkasındaki bir düğümü kovalar.

📍 `core/nav_process.py:890-907`, `utils/planner.py:206`

### F. Determinizm yok — "bazen çok güzel, bazen berbat"

Üç jitter kaynağı:

1. **`shared_state` Manager dict**: nav döngüsü 50 Hz'de ~20 erişim = saniyede ~1000 IPC
   turu. `vision_detected_objects` her turda 10–30 dict'lik listeyi pickle/unpickle ediyor.
2. **FC komut seli**: döngü başına 5 × `set_servo` × 50 Hz = **saniyede 250 `COMMAND_LONG`**.
   ArduPilot her birine `COMMAND_ACK` üretir → link ve FC scheduler dolar. Videoda GPS'in
   23:47:46–47 arasında ~1,5 s donduğu ölçüldü.
3. **`MAV_DATA_STREAM_ALL @ 5 Hz`** (`MainSystem2.py:77-83`) — konum zaten 5 Hz geliyor,
   ama 50 Hz kontrol döngüsü buna körü körüne güveniyor.

Hepsi `loop_dt`'yi oynatıyor; `dt_scale` D terimini 0,2–5,0 arası ölçekliyor
(`planner.py:18-21`) → **efektif sönümleme her turda değişiyor**.
FPS'in 30→17'ye düştüğü an ile dairenin başladığı an çakışması bunun gözlemsel kanıtı.

### G. Task 2'de doyum için emniyet freni yok

PID dalında `SPOT_TURN_THRESHOLD = 30°` var: hata büyükse ileri itki kesilip yerinde
dönülüyor (`nav_process.py:969-974`). **Pure Pursuit dalında bu yok.** Tekne 150° hatayla
bile `1680` seyir gazıyla ilerlemeye devam ediyor + maksimum diferansiyel →
**geometrik olarak daire çizmek zorunda**.

📍 `core/nav_process.py:918-952`

---

## 4. `final_costmap` DEĞERLENDİRMESİ

Turuncu bulut gerçek bir problemin göstergesi. Sebepleri:

1. **1 Hz güncelleme + hiç silmeme.** `CostmapRecorder.update()` saniyede bir çalışıp
   hafızadaki *tüm* nesneleri yeniden çiziyor. Alfa filtresiyle kayan konumlar üst üste
   birikip bulut yapıyor. Tekne yolu da 1 Hz örneklendiği için (1,8 m/s'te ~1,8 m'lik
   parçalar) daire küçük bir kement gibi görünüyor.
2. **Mükerrer ID üretimi.** `ObjectMemoryManager` eşleşme için `type` **ve** `color`
   birebir tutuyor (`camera_process.py:138`). YOLO ters ışıkta aynı şamandırayı bir kare
   sarı bir kare turuncu görürse **yeni ID açılıyor** ve ikisi de 5 s hafızada kalıyor.
   Task 2 filtresi `cid in [1,3]` olduğu için **tek şamandıra iki ayrı engel** oluyor.
3. **Derinlik gürültüsü.** Derinlik, bbox'ın alt %15'inden 5×5 medyan ile alınıyor
   (`camera_process.py:429-447`). Güneş parlaması ve su yansımalarında bu ROI suya düşüyor.

### Önemli nüans
A\*'ın kullandığı costmap her döngüde `fill(127)` ile **sıfırlanıyor**
(`nav_process.py:399`), yani `final_costmap` gibi kümülatif değil. **A\* o bulutu görmüyor.**
Ama **kare-başına saçılma aynı** — ve (D) maddesindeki 1,1 m'lik açıklıkla birleşince asıl
zararı o veriyor.

> `final_costmap`, sorunun kendisi değil; sorunun çok iyi bir **röntgeni**.

---

## 5. TELEMETRİ KAYBININ KÖK NEDENİ

### 5.1 Ölçülen paket boyutu

`telem_process.py:65-87` yapısı birebir kurulup ölçüldü:

| Bileşen | Boyut |
|---|---|
| Sabit alanlar | 447 B |
| `GÖREV_NOKTALARI` (her pakette tekrar!) | **513 B** |
| Her nesne | **200 B** |

| Nesne sayısı | Paket | Seri portta süre (57600) | `write_timeout=0.5` |
|---|---|---|---|
| 0–5 | 1 460 – 1 960 B | 254–340 ms | ✅ |
| **10** | **2 960 B** | **521 ms** | ❌ **kıl payı patlıyor** |
| 20 | 4 960 B | 861 ms | ❌ |
| 30 | 6 960 B | 1208 ms | ❌ |

**Tavan: 57600 baud 8N1 → 5760 B/s → 0,5 s'de en fazla 2880 bayt.**

`ObjectMemoryManager` nesneleri 5 s tutuyor (`camera_process.py:124`) ve `memory_objects`
listesinin **tamamı** pakete giriyor. Videoda kare başına 4–7 etiket sayıldı; 5 saniyelik
hafıza + renk oynamasından doğan mükerrer ID'lerle 10–30 nesne son derece normal.

### 5.2 Kırpılma zinciri

1. `ser.write()` `SerialTimeoutException` fırlatır — **ama teslim edilmiş baytlar zaten
   yola çıkmıştır**, kalan atılır. Havada giden satırın **sonunda `\n` yoktur**.
2. GCS `\n`'e göre bölüp son parçayı `_rx_buffer`'da bekletir (`GCSv1000.py:370-376`).
   Bir sonraki paketin kırpılmış başı **arkasına yapışır**.
3. Nihayet bir `\n` geldiğinde satır `{...}{...` şeklindedir → `json.JSONDecodeError` →
   `pass` (`GCSv1000.py:390-391`) → **sessizce çöpe**.
4. Yeşil LED yalnızca `on_packet()` içinde tetiklenir (`GCSv1000.py:3027-3029`), o da yalnızca
   `json.loads` başarılı olursa. **"Yeşil hiç yanmıyor" = geçerli tek bir JSON satırı bile
   gelmiyor.**

### 5.3 Asimetri — teşhisin ispatı

| Yön | Paket | Sonuç |
|---|---|---|
| GCS → Tekne `report_status` | ~40 B | ✅ her zaman |
| GCS → Tekne 8 GPS noktası | ~500 B | ✅ → **terminalde görülüyor** |
| Tekne → GCS telemetri | 2 960 – 6 960 B | ❌ kırpılıyor |

Komutların sorunsuz gitmesi **RF/menzil sorunu olmadığının kanıtıdır.**

### 5.4 RFD900x ayarlarının değerlendirmesi

Ekran görüntüsündeki ayarlar (`RFD SiK 3.57 on RFD900X2`):

| Ayar | Değer | Değerlendirme |
|---|---|---|
| **Air Speed** | 224 | ✅ Yüksek. Ham ~28 KB/s. **Hava darboğaz değil.** |
| **Baud (seri)** | 57 (57600) | ⚠️ **5,76 KB/s — en dar halka bu** |
| **Mavlink** | Off | ✅ Ham JSON için **doğru** |
| **ECC** | kapalı | ✅ Verim için doğru |
| **RTS CTS** | kapalı | ⚠️ Akış kontrolü yok (aşağıya bkz.) |
| Net ID | 26 (üçünde aynı) | — |
| Max Window | 131 ms | Paketler küçülünce 40–50 ms daha iyi olabilir |
| Duty Cycle / LBT | 100 / 0 | — |

**Kritik düzeltme:** Hava linki (224 kbps) seri porttan (57600) **~5 kat hızlı**. Yani
darboğaz telsizin havası değil, **Jetson ile telsiz arasındaki seri kablo**. Bu, seri
portun taşmadığı anlamına da gelir — `RTS CTS` kapalı olması bu yüzden düşünüldüğü kadar
kritik değil. **Tek gerçek sebep `write_timeout` tavanıdır.**

### 5.5 3 telsiz meselesi (my_id şeması korunarak)

Kimliklemeyi uygulama katmanında `my_id`/`target_id` ile yapmak SiK P2P firmware'de
**doğru ve tek uygulanabilir çözümdür**. Sorun kimlikleme değil, **eşzamanlı konuşma**:

SiK TDMA iki düğüm için tasarlanmıştır. Üçüncü düğüm eklendiğinde tekne ve drone aynı
sırayı kendilerinin sanıp aynı anda basar → çarpışma.

Çarpışma olasılığı **havada geçirilen süreyle** orantılıdır. 224 kbps'te:

| Paket | Havada kalma |
|---|---|
| 3 000 B (şu anki) | **~107 ms** |
| 254 B (hedef) | **~9 ms** |

> **Paketi küçültmek çarpışma penceresini 12 kat daraltır.** Tek bir yazılım
> düzeltmesi hem kırpılmayı hem 3-düğüm çarpışmasının büyük kısmını çözer.
> Firmware değiştirmeye gerek yok.

**Yapısal ek iyileştirme:** Drone şu anda **hiç sorgulanmıyor** — kendi kendine yayın
yapıyor (`GCSv1000.py:3012`'deki yorum: *"drone her karede paket yolluyordu"*). Tekne ise
sorguya cevap veriyor. İkisi birbirinden habersiz. GCS drone'u da sorgulayıp tekne ile
**dönüşümlü** yaparsa (t=0 tekne, t=250 ms drone, …), aynı anda asla iki araç konuşmaz.
Bu, radyonun TDMA'sının üstüne kurulmuş bir **uygulama katmanı TDMA'sıdır**.

> **LED uyarısı:** Yanıp sönen aktivite ışığı yalnızca RF enerjisinin alışverişte olduğunu
> gösterir — paketlerin CRC'yi geçtiğini **göstermez**. Çarpışan 3 düğümlü bir ağda LED'ler
> neşeyle yanıp söner, geçerli paket sayısı düşüktür.

---

## 6. MANEVRA DÜZELTMELERİ — 11 MADDE, DETAYLI

Öncelik: **1, 2, 3** tek başına dairelerin büyük kısmını keser. **5, 6, 7** kalan
belirsizliği azaltır. **8, 9** determinizmi getirir. **4, 10, 11** destekleyicidir.

---

### ✦ 1 — Pure Pursuit kazancını aktüatör aralığına oturt
**📍 `config.py:78-79`**

**Sorun:** `PURE_PURSUIT_KP = 5.5` ile pervaneler 27°'de doyuyor (§2.3).
`PURE_PURSUIT_KD = 2.5` ise yol değiştiğinde hatanın tek döngüde 80° sıçramasını
`80 × 2.5 = 200` düzeltmeye çevirip doyumu garantiliyor.

**Hedef:** Doyum ~70°'de başlasın.
`correction = 148` doyum eşiği; `148 / 70 ≈ 2.1`

**Değişiklik:**
```python
PURE_PURSUIT_KP = 2.0      # 5.5'ten
PURE_PURSUIT_KD = 0.8      # 2.5'ten
```

**Risk:** Çok düşük KP dar koridorda yavaş tepki verir. Havuz testinde 1,5–2,5 arası
tarayıp en az salınım veren değeri seçin.

**Doğrulama:** `ACI_FARKI` 30° iken `ARKA_SOL/SAG` **doymamalı** (1100/1900 görmemelisiniz).

---

### ✦ 2 — `MAX_PWM_CHANGE` slew-rate limitini gerçekten uygula
**📍 `core/nav_process.py` (`apply_motor_mixer`, satır 28-93) + `config.py:63`**

**Sorun:** `MAX_PWM_CHANGE = 60` tanımlı ama **kullanılmıyor**. Ölçülen: 0,25 s'de
800 PWM ters yönde sıçrama (§2.2).

**Değişiklik:** `apply_motor_mixer` içinde her kanal için bir önceki komutu saklayıp
değişimi sınırla. Fonksiyon şu anda saf (state'siz) olduğu için kalıcı bir sözlük
gerekir — `USVController.pwms` zaten son değeri tutuyor, oradan okunabilir.

```
yeni = clip(hedef, onceki - MAX_PWM_CHANGE, onceki + MAX_PWM_CHANGE)
```

**Kalibrasyon:** 50 Hz'de `MAX_PWM_CHANGE = 60` → saniyede 3000 PWM, yani tam aralığı
0,27 s'de tarar. Bu hâlâ hızlı; **20–30** daha uygun olabilir.
**Not:** Kontrol döngüsü hızını değiştirirseniz (madde 8) bu değeri de ölçekleyin.

**Kritik istisna:** Acil durum dalı (`nav_process.py:786-815`) ve SIGTERM nötrleme
(`nav_process.py:202-213`) slew limitini **atlamalı** — orada anında tepki gerekir.

---

### ✦ 3 — `find_lookahead_point`'i düzelt
**📍 `utils/planner.py:369-409`**

**Sorun:** §3.C. Yoldan 0,8 m'den fazla sapınca "en yakın düğüm" döndürülüyor; o düğüm
yanda/arkada olabiliyor.

**Değişiklik:** "En yakın düğüm" mantığı yerine **yol poligonuna dik izdüşüm + yay uzunluğu
boyunca ilerleme**:

1. `start_idx`'ten itibaren her segmente teknenin **dik izdüşümünü** hesapla; en yakın
   izdüşüm noktasını ve bulunduğu segmenti bul (düğümü değil, **segment üzerindeki noktayı**).
2. Hedef noktayı o izdüşümden itibaren yol boyunca `lookahead_dist` kadar **ileri** yürüyerek
   belirle (segment sınırlarını aşarak devam et).
3. Yoldan sapma büyükse (`> 2 × lookahead`) hedefi yine ileri al ama ileri gazı kıs
   (madde 5) — asla geriye dönük bir nokta seçme.

**Neden kritik:** Bu, teknenin arkasındaki noktayı hedeflemesini **yapısal olarak imkânsız**
kılar. Madde 1 ve 2 semptomu bastırır; bu madde sebebi ortadan kaldırır.

---

### ✦ 4 — Yer hızını doğru MAVLink mesajından oku
**📍 `hardware/MainSystem2.py:110-117`**

**Sorun:** `LOCAL_POSITION_NED` gelmiyor → `HIZ: 0.0` → lookahead kalıcı 0,8 m.

**Değişiklik:**
```
1. tercih: VFR_HUD.groundspeed        (m/s, doğrudan)
2. tercih: GPS_RAW_INT.vel / 100.0    (cm/s → m/s)
3. tercih: GLOBAL_POSITION_INT.vx,vy  (cm/s → hypot/100)
son çare : 0.0
```
Üçü de yoksa `None` dönüp çağıran tarafın bunu "veri yok" olarak ayırt edebilmesi daha iyi.

**Etki:** `lookahead = clip(hız × K_SPEED, 0.8, 1.5)`. 1,8 m/s'te 1,5 m'ye çıkar — daha
yumuşak takip. `PURE_PURSUIT_MAX_LOOKAHEAD`'ı 2,5–3,0'a çıkarmayı da değerlendirin;
1,5 m hâlâ kısa.

**Doğrulama:** HUD'da `HIZ` sıfırdan farklı ve makul (0–2,5 m/s) olmalı.

---

### ✦ 5 — PP dalına gaz kısma / spot-turn emniyeti ekle
**📍 `core/nav_process.py:918-952`**

**Sorun:** §3.G. PID dalında `SPOT_TURN_THRESHOLD` var, PP dalında yok. Tekne 150° hatayla
seyir gazında ilerlerken maksimum dönüş uyguluyor → daire.

**Değişiklik — iki kademeli:**

*Kademe 1 — orantılı gaz kısma (tercih edilen, yumuşak):*
```
gaz_carpani = clip(cos(heading_err), 0.25, 1.0)
forward = 1500 + CRUISE_PWM * gaz_carpani
```
45°'de gaz %71'e, 90°'de %25'e iner. Tekne dönerken yanal kaymaz.

*Kademe 2 — histerezisli spot turn:*
```
|err| > 60°  -> forward = 1500 (yerinde dön)
|err| < 35°  -> normal seyre dön
```
Histerezis şart; tek eşik kullanılırsa eşik civarında yeni bir çatırdama doğar.

**Not:** Eşiği PP'nin iç `heading_err`'ine bağlayın, `aci_farki`'ya değil — yol takibinde
doğru referans odur.

---

### ✦ 6 — Engel şişirmesini konum gürültüsüne göre boyutlandır
**📍 `config.py:14-15`, `core/nav_process.py:421`, `utils/planner.py:43-48`**

**Sorun:** §3.D. Toplam açıklık 1,1 m, konum gürültüsü 1–2 m.

**Önce yapısal düzeltme:** `cv2.circle(costmap_img, p_virtual, 6, 0, -1)` içindeki **6 px
sabit kodlanmış**. Bu değer **şamandıranın fiziksel yarıçapını** temsil etmeli
(~0,2–0,3 m = 2–3 px), tüm emniyet payı **şişirmeye** gitmeli. Şişirme geometrik olarak
doğru uygulanır; sabit daire ise değil. `BUOY_RADIUS_M` diye config'e alın.

**Sonra boyutlandırma — koridor genişliğine bağlı:**
```
serbest_geçiş = koridor_genişliği − 2 × (BUOY_RADIUS_M + ROBOT_RADIUS_M + INFLATION_MARGIN_M)
serbest_geçiş ≥ 1.0 m olmalı
```

| Koridor genişliği | Maks. `INFLATION_MARGIN_M` (BUOY=0.25, ROBOT=0.4 ile) |
|---|---|
| 3,0 m | 0,35 m |
| 4,0 m | 0,85 m |
| 5,0 m | 1,35 m |

> ⚠️ **Uyarı:** Şişirmeyi körü körüne büyütmek koridoru kapatır ve A\* hiç yol bulamaz
> hale gelir — bu, mevcut durumdan **daha kötüdür** (obstacle-blind PID'e düşer).
> **Önce parkurdaki şamandıra aralığını ölçün.**

**Alternatif (daha iyi):** Şişirmeyi büyütmek yerine **konum gürültüsünü düşürün**
(madde 11 + derinlik filtresi). Gürültü 0,5 m'ye inerse 1,1 m açıklık zaten yeterli olur.

**Ek:** `planner.py:147`'deki bayat "2.1 m" yorumunu güncelleyin.

---

### ✦ 7 — A\* başarısızlığını doğru ele al + planlamayı kontrolden ayır
**📍 `core/nav_process.py:890-907`, `utils/planner.py:206`**

**Sorun A:** A\* `None` dönünce eski yol temizlenmiyor → bayat yol takip ediliyor.
```python
if new_path and len(new_path) >= 2:
    current_path = new_path
# else: hiçbir şey yapılmıyor  <-- BUG
```
**Değişiklik:** `else` dalında `current_path = None; path_progress_idx = 0`.
Böylece kontrol, obstacle-blind ama **yumuşak** olan PID'e düşer — daireden iyidir.

**Sorun B:** `time_budget_s = 0.06` çok dar; saf-Python A\* ~1000–2000 düğüm açabiliyor.

**Değişiklik:** Planlama hızını kontrol hızından ayırın:
- Kontrol (Pure Pursuit) **50 Hz** — son geçerli yol üzerinde
- Planlama (A\*) **5 Hz**, `time_budget_s = 0.12`

Şu anda ikisi de aynı döngüde (`plan_timer > 4` → ~10 Hz) ve A\* çalışan tur kontrolü
geciktiriyor. Ayırmanın en temiz yolu A\*'ı ayrı bir process/thread'e almak; daha basit
yol `plan_timer > 9` yapıp bütçeyi artırmak.

**Sorun C:** Yol yaşı denetimi yok. `current_path` üretildiği zamanı saklayıp
**0,5 s'den eskiyse** kullanmayın.

---

### ✦ 8 — FC komut selini kes (250 msg/s → ~60 msg/s)
**📍 `hardware/MainSystem2.py:142-161`, `core/nav_process.py:1058-1059`**

**Sorun:** §3.F.2. Döngü başına 5 `set_servo` × 50 Hz = **saniyede 250 `COMMAND_LONG`**.
ArduPilot her birine `COMMAND_ACK` üretir. Videoda GPS'in 1,5 s donduğu ölçüldü.

**Değişiklik — üç seçenek, birlikte kullanılabilir:**

1. **Kontrol döngüsünü 25 Hz'e indir** (`nav_process.py:1059`, `0.02` → `0.04`).
   Teknenin yaw dinamiği ~1–2 s zaman sabitinde; 25 Hz zaten 12–25 kat aşırı örnekleme.
   Tek satırda IPC, MAVLink ve CPU yükünü **yarıya** indirir.
2. **Değişim eşiği:** `set_servo` içinde PWM önceki değerden `< 3` farklıysa gönderme.
   Madde 2'nin slew limiti ile birlikte ön pervanelerde büyük tasarruf sağlar.
3. **Sensör okuma ile komut yazmayı ayır:** sensörleri 50 Hz oku, motorları 20 Hz yaz.

**Beklenen sonuç:** GPS donmaları biter, `loop_dt` jitteri azalır, `FPS` 30'da kalır.

---

### ✦ 9 — `shared_state` IPC yükünü azalt
**📍 `core/nav_process.py:382` ve döngü genelinde**

**Sorun:** §3.F.1. `shared_state` bir `mp.Manager().dict()` — **her erişim bir soket üzerinden
pickle/unpickle turudur**. `vision_detected_objects` 10–30 dict'lik liste ve **50 Hz'de
okunuyor**.

**Değişiklik:**
1. `vision_detected_objects`'i **10 Hz'de** oku, yerel değişkende sakla:
   ```
   if now - last_vision_read > 0.1: cache = shared_state.get(...); last_vision_read = now
   ```
2. Yazma tarafında (`nav_process.py:1034-1038`) 5 PWM'i **tek bir tuple** olarak yaz
   (5 IPC turu → 1).
3. Uzun vadede: nesne listesini `mp.Array` + basit binary paketleme ile taşıyın.

**Beklenen sonuç:** Nav döngüsünün IPC yükü ~%80 düşer; `loop_dt` stabilleşir; `dt_scale`
sabitlenir → **sönümleme her turda aynı olur** (bu, "bazen iyi bazen kötü" sorununun
doğrudan çözümüdür).

---

### ✦ 10 — Telemetri paketinden `objects`'i çıkar
**📍 `core/telem_process.py:80`**

**Sorun:** Hem telemetriyi öldürüyor (§5) hem de `telem_process`'in her turda Manager dict
üzerinden büyük listeyi çekmesine sebep oluyor (madde 9 ile aynı kök).

Detaylı çözüm §7'deki **T1**'de.

**Manevra tarafına faydası:** `shared_state['vision_detected_objects']` üzerindeki
okuyucu sayısı azalır, nav döngüsünün Manager kilidinde beklemesi düşer.

---

### ✦ 11 — Mükerrer nesne ID üretimini durdur
**📍 `utils/../camera_process.py:113-181` (`ObjectMemoryManager`)**

**Sorun:** §4.2. Eşleşme `dist < 2.5 AND type == AND color ==` şartına bağlı
(`camera_process.py:138`). Renk bir karede değişirse yeni ID doğar; ikisi de 5 s yaşar;
Task 2 filtresi `cid in [1,3]` olduğu için **tek şamandıra iki engel** olur.

**Değişiklik:**
1. Eşleşmeyi **konum + tip** üzerinden yap; **rengi eşleşme şartından çıkar**.
2. Rengi izlenen nesnenin **oylanan özelliği** yap: her görüşte renk sayacını artır,
   baskın rengi kullan. Böylece tek kare hatası kimliği bölmez.
3. `MERGE_DISTANCE = 2.5` m — konum gürültüsü 1–2 m olduğu için makul; renk şartı
   kalkınca zaten daha etkili çalışacak.
4. **Ek öneri:** Derinlik ölçümünü sağlamlaştırın (`camera_process.py:429-447`).
   ROI'yi bbox alt %15 yerine **bbox merkezinin biraz altı**na alın ve geçerli örnek
   sayısı 5'ten azsa tespiti **tamamen atın** (şu anda `inf` dönüp `0.5 < dist < 15`
   filtresine takılıyor, ama az sayıda geçerli örnekle hesaplanan medyan de güvenilmez).

**Beklenen sonuç:** Costmap'teki engel sayısı gerçek şamandıra sayısına yaklaşır,
saçılma düşer, A\*'ın kaçınma yönü kararlı hale gelir (madde 6'ya alternatif/tamamlayıcı).

---

## 7. TELEMETRİ DÜZELTMELERİ

### Aşama 1 — Yazılım (sorunu çözen kısım, RF'e dokunmadan)

| # | Yer | Değişiklik |
|---|---|---|
| **T1** | `telem_process.py:80` | `objects`'i ana paketten çıkar. Gerekiyorsa **en yakın 6–8 nesneyi**, kısaltılmış alanlarla (`{i,c,la,lo,d}` = **53 B**) ve **ayrı, seyrek** bir pakette gönder |
| **T2** | `telem_process.py:86` | `GÖREV_NOKTALARI`'nı her pakete koyma — sadece değiştiğinde veya GCS istediğinde |
| **T3** | `telem_process.py:65-87` | Float'ları `round()` ile kırp (koordinat **7 hane** = 1 cm, açı 1, mesafe 1). Anahtarları kısalt. `json.dumps(..., separators=(',',':'))` |
| **T4** | `utils/telem.py:20-26` | Boyut koruması: paket 1500 B'ı aşarsa nesneleri düşürüp tekrar dene. **Asla yarım satır yollama.** İdeali: uzunluk önekli çerçeve (`struct.pack('!H', n) + payload`) |
| **T5** | `GCSv1000.py:3493` | Drone'u da sorgula, tekne ile **dönüşümlü** (§5.5) |

**Neden `round()` önemli:** Python `json.dumps` float'ları `repr` ile yazıyor:
`40.809465000000004` — **17 hane**. Tek başına koordinat baytlarının ~%40'ını siliyor.

### Ölçülen sonuç

| Aksiyon | Paket | Seri süre | 2 Hz'de doluluk |
|---|---|---|---|
| Şu anki (12 nesne) | ~3 360 B | 583 ms | **117 %** ❌ |
| T1 (objects çıkar) | 960 B | 167 ms | 33 % ✅ |
| T1+T2 | 447 B | 78 ms | 16 % ✅ |
| **T1+T2+T3** | **254 B** | **44 ms** | **8,8 %** ✅✅ |
| T1+T2+T3 + 8 nesne ayrı | 698 B | 121 ms | 24 % ✅ |

254 B ile **5 Hz'e bile çıkabilirsiniz** (%22 doluluk) — şu anki 2 Hz'den akıcı bir GCS,
üstelik komutlara bol boşluk kalarak. Havada kalma süresi 107 ms → **9 ms**.

### Aşama 2 — İsteğe bağlı, marj için

- **Seri baud'u 115200'e çıkar.** Artık mantıklı, çünkü hava (224 kbps) seri porttan hızlı.
  `write_timeout` tavanı 2880 → **5760 bayt** olur.
  Değiştirilecek yerler: **üç telsizde de** `Baud` alanı + `config.py:35` (`SERIAL_BAUD`) +
  `GCSv1000.py:2310` (combo varsayılanı).
  ⚠️ RFD Tools'taki **"Copy required to remote"** düğmesini unutmayın; üç radyoda da aynı olmalı.
- **RTS CTS**: **yalnızca** Jetson↔telsiz kablosunda CTS/RTS hatları fiziksel olarak bağlıysa
  açın. Bağlı değilse açmak linki tamamen kilitler. Aşama 1'den sonra zaten gerekmeyecek.
- **Max Window 131 ms → 40–50 ms**: paketler küçüldükten sonra gidiş-dönüş gecikmesini
  düşürür, tekne/drone daha sık sıra değiştirir. Test ederek karar verin.
- **Air Speed 224**: menzil açısından en agresif ayar (alıcı hassasiyeti en düşük).
  Paketler 254 B'a inince 64 kbps'te bile havada sadece ~32 ms tutar. Sahada menzil/RSSI
  sıkıntısı yaşarsanız düşürmek ciddi marj kazandırır.

### Ölçüm önerisi

Ekran görüntüsündeki RSSI satırı `pkts: 0`, `L/R RSSI: 0/0` gösteriyor — bu **masa başı,
karşı taraf kapalıyken** alınmış bir okuma, ondan sonuç çıkarılamaz.

Asıl bakılması gereken, **tekne çalışırken** RFD Tools → RSSI sekmesindeki:
- `pkts` → alınan paket sayısı
- `rxe` → **alım hatası** (3-düğüm çarpışmasının doğrudan göstergesi)
- `L/R RSSI` ve `L/R noise` → link marjı

`rxe` yüksekse çarpışmalar gerçekten canınızı yakıyor demektir; düşükse **Aşama 1 tek
başına yeter**.

> GCS uygulaması ile RFD Tools aynı COM portunu paylaşamaz. Ölçümü GCS kapalıyken,
> tekne yayın yaparken yapın.

---

## 8. EK BULGULAR

### 8.1 `T2_ZONE_MID` ile `T2_ZONE_END` birebir aynı — gizli hata
```python
# config.py:94-97
T2_ZONE_MID_LAT = 40.8095851 ;  T2_ZONE_MID_LON = 29.2622612
T2_ZONE_END_LAT = 40.8095851 ;  T2_ZONE_END_LON = 29.2622612   # AYNI
```
GCS'ten nokta gönderdiğiniz için şu anda maskeleniyor. Ama GCS'ten gönderilmezse
`TASK2_GO_TO_END` durumu **anında tamamlanır** (3 m kabul yarıçapı, `nav_process.py:458`)
ve Task 2 atlanır. Config değerlerini gerçek parkurla eşitleyin veya en azından farklı yapın.

### 8.2 Task 2'de kırmızı/yeşil şamandıralar engel sayılmıyor
```python
# nav_process.py:404
if "TASK2" in mevcut_gorev and obj.get('cid') not in [1, 3]:
    continue
```
`cid` 0 (kırmızı) ve 4 (yeşil) **tamamen yok sayılıyor**. Parkurunuz sarı+turuncu ise
doğru; ama giriş/çıkış kapısı şamandıraları varsa onlara çarpma riski var. Bilinçli bir
tercihse sorun yok, değilse gözden geçirin.

### 8.3 `hdg` alanı string, diğerleri sayı
```python
# telem_process.py:74
"hdg": f"{heading:.0f}" if heading is not None else "0",
```
Tutarsız tipleme. GCS tarafında `int()`/`float()` dönüşümü gerektiriyor. T3 ile birlikte
sayıya çevirin.

### 8.4 `LIDAR_KORIDOR_KP`, `HYBRID_STEP_DIST`, `NAV_MODE`, `CAM_RES`, `SHOW_LOCAL_WINDOW`
Config'de tanımlı, kodda **hiç kullanılmıyor** (grep ile doğrulandı). Kafa karışıklığı
yaratıyor; ya kullanın ya silin. `MAX_PWM_CHANGE` de bu gruptaydı — onu madde 2 ile
kullanıma alıyoruz.

### 8.5 `CommandReceiver` ile ana telemetri döngüsü aynı `Serial` nesnesini paylaşıyor
`utils/telem.py:53-64` ayrı thread'den `in_waiting`/`readline()` çağırırken ana döngü
`write()` yapıyor. pyserial `Serial` **thread-safe değildir**. Linux'ta okuma/yazma ayrı
yönler olduğu için genelde sorun çıkmaz, ama T4'teki uzunluk önekli çerçevelemeye
geçilirse bir kilit (`threading.Lock`) eklemek gerekir.

### 8.6 `_expand_image_if_needed` içinde eksik `dx` ataması
`utils/costmap_recorder.py:63-65` — `elif x_px > w - margin` dalında `new_w` büyütülüyor
ama `dx` **ayarlanmıyor** (0 kalıyor). Sağa/aşağı taşmada görüntü büyüyor ama origin
kaymadığı için bu yön doğru çalışıyor; sola/yukarı taşmada `dx`/`dy` doğru. Yani mevcut
haliyle çalışıyor, ama asimetrik ve kırılgan. Düşük öncelik.

---

## 9. UYGULAMA SIRASI VE DOĞRULAMA PLANI

### Grup 1 — Telemetri (bağımsız, düşük risk, hemen yapılabilir)
**T1 → T2 → T3 → T4** (tekne tarafı, `telem_process.py` + `telem.py`)

*Doğrulama:* GCS'te **yeşil LED yanıp sönmeli**, tekne konumu haritada akıcı güncellenmeli,
teknenin terminalinde `Write timeout` **tamamen kesilmeli**.

Sonra **T5** (GCS polling dönüşümlü) → drone LED'i de kararlı yanmalı.

---

### Grup 2 — Manevra çekirdeği (dairelerin kaynağı)
**3 → 1 → 2** bu sırayla. (3 sebebi, 1 ve 2 semptomu kapatır.)

*Doğrulama (karada, motorlar sökülü veya suda düşük gaz):*
- `ACI_FARKI` 30° iken PWM **doymamalı**
- Yol değiştiğinde PWM'de basamak sıçraması **olmamalı**
- Şamandıra yanından geçerken `DUMEN_PWM` 1100/1900'e **hiç gitmemeli**

---

### Grup 3 — Kararlılık ve determinizm
**8 → 9 → 7**

*Doğrulama:*
- HUD'da `FPS` **sürekli 30** kalmalı (17'ye düşmemeli)
- `IDA_KONUM` **donmamalı** (§2.1'deki 1,5 s'lik donmalar bitmeli)
- Aynı senaryo **art arda 3 denemede benzer** davranmalı

---

### Grup 4 — Algı kalitesi ve emniyet payı
**11 → 4 → 5 → 6**

> **6'yı en sona bırakın** ve **önce parkurdaki şamandıra aralığını ölçün.**
> 11 numara gürültüyü düşürürse 6'ya hiç gerek kalmayabilir.

*Doğrulama:*
- `final_costmap`'te turuncu bulut yerine **ayrık noktalar** görülmeli
- HUD'da `HIZ` sıfırdan farklı ve makul olmalı
- Büyük hata anlarında ileri gaz **kısılmalı**

---

### Grup 5 — Temizlik
**10** (Grup 1 ile birlikte yapılmış olacak), **8.1, 8.3, 8.4**

---

## ÖZET TABLO

| # | Konu | Dosya | Öncelik |
|---|---|---|---|
| 3 | `find_lookahead_point` düğüme yapışması | `planner.py:369-409` | 🔴 Kritik |
| 1 | PP kazancı (KP 5.5→2.0, KD 2.5→0.8) | `config.py:78-79` | 🔴 Kritik |
| 2 | `MAX_PWM_CHANGE` slew limiti | `nav_process.py:28-93` | 🔴 Kritik |
| T1–T4 | Telemetri paketi küçültme | `telem_process.py`, `telem.py` | 🔴 Kritik |
| 8 | FC komut seli (250→60 msg/s) | `MainSystem2.py:142`, `nav_process.py:1059` | 🟠 Yüksek |
| 9 | Manager dict IPC yükü | `nav_process.py:382, 1034` | 🟠 Yüksek |
| 7 | A\* `None` → bayat yol + bütçe | `nav_process.py:890-907` | 🟠 Yüksek |
| T5 | GCS dönüşümlü polling | `GCSv1000.py:3493` | 🟠 Yüksek |
| 11 | Mükerrer nesne ID | `camera_process.py:113-181` | 🟡 Orta |
| 4 | Yer hızı MAVLink kaynağı | `MainSystem2.py:110-117` | 🟡 Orta |
| 5 | PP dalında gaz kısma | `nav_process.py:918-952` | 🟡 Orta |
| 6 | Engel şişirmesi (ölçüme bağlı) | `config.py:14-15`, `nav_process.py:421` | 🟡 Orta |
| 8.1 | `T2_MID == T2_END` | `config.py:94-97` | 🟢 Düşük |
| 8.3 | `hdg` string tipi | `telem_process.py:74` | 🟢 Düşük |
| 8.4 | Kullanılmayan config anahtarları | `config.py` | 🟢 Düşük |

---

## 10. UYGULAMA DURUMU VE DOĞRULAMA SONUÇLARI

### 10.1 Uygulanan maddeler

| # | Madde | Dosyalar | Durum |
|---|---|---|---|
| 1 | PP kazancı 5.5→2.0 / 2.5→0.8 | `config.py` | ✅ |
| 2 | `MAX_PWM_CHANGE` slew limiti | `nav_process.py` (`_slew`, `apply_motor_mixer`) | ✅ |
| 3 | `find_lookahead_point` yeniden yazıldı | `planner.py` | ✅ |
| 4 | Yer hızı `VFR_HUD.groundspeed` | `MainSystem2.py` | ✅ |
| 5 | PP gaz kısma + histerezisli spot turn | `nav_process.py` | ✅ |
| 6 | `BUOY_RADIUS_M` ayrıştırıldı | `config.py`, `nav_process.py` | ✅ |
| 7 | A\* `None` → yol at + yaş denetimi + 5 Hz/0.12 s | `nav_process.py`, `planner.py`, `config.py` | ✅ |
| 8 | Döngü 25 Hz + `set_servo` dedup | `config.py`, `nav_process.py`, `MainSystem2.py` | ✅ |
| 9 | Vision 10 Hz cache, yayın 10 Hz, lidar okuması atlandı | `nav_process.py` | ✅ |
| 10 | Telemetriden `objects` çıkarıldı | `telem_process.py` | ✅ |
| 11 | Renk oylaması (`cid_votes`) | `camera_process.py` | ✅ |
| T1 | `objects` kaldırıldı | `telem_process.py` | ✅ |
| T2 | Waypoint bloğu sadece değişince/10 s'de bir | `telem_process.py`, `config.py` | ✅ |
| T3 | Float yuvarlama + kompakt ayraç | `telem_process.py`, `telem.py` | ✅ |
| T4 | Boyut koruması + resync | `telem.py`, `config.py` | ✅ |
| — | `mission_points` paylaşımlı bellek köprüsü | `nav_process.py`, `telem_process.py`, `main_orchestrator.py` | ✅ |

**Ertelenen:** T5 (GCS dönüşümlü polling), 12 (turuncu sınır çizgi kısıtı),
13 (pusula doğrulama), 14 (derinlik eleme).

### 10.2 Ölçülen doğrulama sonuçları

**Madde 3 — geriye hedef verme (asıl daire sebebi):**

| Tekne konumu | ESKİ kod | YENİ kod |
|---|---|---|
| (1.2, 1.0) | **−140,2°** ← geriye dönüş | −51,3° |
| (2.5, −1.5) | — | +61,9° |
| (0.3, 0.9) | — | −48,4° |

Eski kod `min_d2 (2,44) > lookahead² (0,64)` olduğu için `path[0]`'ı, yani **teknenin
arkasındaki düğümü** döndürüyordu. Yeni kodda hiçbir örnekte hedef geride değil.

**Madde 3 + 4 birlikte** (hız düzeltilince lookahead 0,8 → 1,8 m):

| | la = 0,8 | la = 1,8 |
|---|---|---|
| En kötü yön hatası | 61,9° | **53,8°** |

**Madde 1 — doyum eşiği:**

| Kazanç | Doyum |
|---|---|
| KP = 5,5 (eski) | **26,9°** |
| KP = 2,0 (yeni) | **74,0°** |
| Direct-Drive PID KP = 1,5 (referans) | 98,7° |

En kötü gerçekçi hata 53,8° < 74,0° doyum eşiği → **normal yol takibinde doyuma
girilmiyor**, ve 53,8° < 60° spot-turn eşiği → **gereksiz yerinde dönüş tetiklenmiyor**.

**Madde 2 — slew limiti:** `MAX_PWM_CHANGE = 60 @ 25 Hz = 1500 PWM/s`.
1100→1900 tam geçiş artık **0,53 s** sürüyor (ölçülen arıza: 0,25 s'de ~800 PWM ters dönüş).

**Telemetri:**

| Paket | Boyut | 2 Hz doluluk | Havada (@224 kbps) |
|---|---|---|---|
| ESKİ (12 nesne) | 3 360 B | %117 ❌ | 107 ms |
| YENİ (waypoint'li) | **708 B** | %24,6 ✅ | 25,3 ms |
| YENİ (waypoint'siz) | **332 B** | %11,5 ✅ | 12 ms |

`write_timeout` bütçesi 2880 B, `TELEM_MAX_PAYLOAD_B` 1500 B — ikisinin de altında.

**T4 davranış testi (sahte seri port):**
- Normal paket → gönderildi, `\n` ile bitiyor ✅
- 3 250 B'lık eski paket → **porta hiç yazılmadı**, yarım satır gitmedi ✅
- Zorlanmış timeout → `reset_output_buffer()` + tek `\n` yazıldı, GCS tamponu resenkronize ✅

### 10.3 ⚠️ Sahaya çıkmadan önce mutlaka kontrol edin

**Koridor genişliği.** `INFLATION_MARGIN_M = 0.55` ile toplam açıklık **1,20 m**
(eskiden 1,10 m). Ölçülen serbest geçiş:

| Koridor | Serbest geçiş | Durum |
|---|---|---|
| **3,0 m** | **0,60 m** | ❌ **A\* yol bulamaz** |
| 4,0 m | 1,60 m | ✅ |
| 5,0 m | 2,60 m | ✅ |

Parkurunuzdaki şamandıra aralığı **4 m'den darsa** `config.py`'de
`INFLATION_MARGIN_M`'i düşürün:
- koridor 3,0 m → `INFLATION_MARGIN_M = 0.35`
- koridor 3,5 m → `INFLATION_MARGIN_M = 0.60`

A\*'ın hiç yol bulamaması, engel-kör PID'e düşmek demektir — mevcut durumdan **daha kötüdür**.

### 10.4 Sahada doğrulama sırası

1. **Karada, motorlar suda değilken:** sistemi başlat, GCS'i bağla.
   → GCS'te **yeşil LED yanıp sönmeli**, terminalde `Write timeout` **hiç çıkmamalı**.
   → HUD'da **`HIZ` sıfırdan farklı** olmalı (madde 4).
2. **GCS'ten 8 GPS noktası gönder.** → Haritada yeşil noktalar **gönderdiğin konumlarda**
   görünmeli (eskiden config varsayılanlarını gösteriyordu).
3. **Düşük gazda suda:** `ACI_FARKI` 30° iken PWM'ler **1100/1900'e gitmemeli**.
4. **Şamandıra yanından geçiş:** `DUMEN_PWM` uçlara **hiç dayanmamalı**.
5. **Aynı senaryoyu 3 kez tekrarla:** `FPS` sürekli 30 kalmalı, `IDA_KONUM` donmamalı,
   üç denemede de **benzer** iz çizilmeli.
