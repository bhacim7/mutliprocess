# drone_color.py — Plaket Rengi Tespiti Analizi ve Düzeltmeler

**Tarih:** 2026-08-11

**Sistem:** Jetson Orin Nano (uçuş kontrolünden bağımsız) · Logitech Brio 4K ·
~10 m irtifa · kamera 2–3° eğimli (neredeyse nadir) · **beton zemin** ·
50×50 cm RAL 9005 / RAL 3026 / RAL 6037 plaketler · sonuç RFD900x ile GCS'e.

> **Önemli kısıt:** Plaketler elde olmadığı için **kalibrasyon yapılamadı**. Aşağıdaki
> eşikler ölçüm değil, RAL referans değerleri + beton albedosu + sRGB gama modellemesiyle
> **hesaplanmış** değerlerdir. Yön güvenilir, tam sayılar sahada doğrulanmalı.

---

## 1. EN KRİTİK BULGU

**Siyah plaket muhtemelen hiç tespit edilmiyordu.**

Betona göre pozlanmış bir sahnede RAL 9005'in beklenen değeri **V ≈ 54–76**,
eski eşik ise `V ≤ 45` idi.

Sentetik sahne testiyle doğrulandı:

| | Siyah maskesinde bulunan piksel |
|---|---|
| **Eski eşik (V ≤ 45)** | **0** ← plaket görünmüyor |
| Yeni eşik (V ≤ 85) | 2 025 |

---

## 2. ÇÖZÜNÜRLÜK — beklenen kaldıraç değildi

`min_area = roi_alanı × 0.005` **göreli** bir eşikti. Çözünürlük artınca eşik de aynı
oranda büyüdüğü için menzil değişmiyordu:

| Yapılandırma | min_area | Maks. irtifa |
|---|---|---|
| Eski (640×480 → 320×240) | 188 px | **9,0 m** |
| 1920×1080 | 5 080 px | 9,5 m |
| 3840×2160 | 20 321 px | 9,5 m |

Ayrıca `cv2.resize(frame, (320,240))` kameradan geleni anında çöpe atıyordu.

**Düzeltme sonrası** (mutlak eşik `MIN_BLOB_AREA_PX = 400`, resize kaldırıldı, 1280×720):

| İrtifa | Plaket | Durum |
|---|---|---|
| 5 m | 90,7 px | ✅ |
| **10 m** | **45,3 px** | ✅ (uçuş irtifanız) |
| 15 m | 30,2 px | ✅ |
| 22 m | 20,6 px | ✅ sınır |
| 25 m | 18,1 px | ❌ |

**Menzil 9,0 m → 22 m.**

---

## 3. RENK ARALIKLARI

RAL referanslarının birden çok sRGB karşılığı test edildi:

| Renk | Hesaplanan H | Hesaplanan S | Eski aralık | Değerlendirme |
|---|---|---|---|---|
| RAL 3026 kırmızı | **0–5** | 222–255 | H 0–10 / 170–179 | ✅ rahat, sarmalama doğru |
| RAL 6037 yeşil | **70–78,5** | 255 | H 40–85 | ⚠️ üstte 6,5° marj, altta 30° boşuna |
| RAL 9005 siyah | — | 0–32 | V ≤ 45 | ❌ plaket V≈54–76 |

### Pozlama modeli (sRGB gama dahil)

`V_yüzey = V_beton × (Y_yüzey / Y_beton)^(1/2.2)`

| Yüzey | Oto-pozlama (beton V=128) | Kilitli (beton V=180) |
|---|---|---|
| RAL 3026 kırmızı | 128 | 180 |
| RAL 6037 yeşil | **106** | 150 |
| **RAL 9005 siyah** | **54** | **76** |
| Gölgedeki beton | 62 | 87 |

İki sonuç:
- Yeşil, oto-pozlamada V≈106 → eski 110 tabanının **altında**, kaçırılıyordu
- Siyah, her koşulda 45 tavanının **üstünde**, hiç görülmüyordu

### Uygulanan aralıklar

| | Eski | Yeni | Gerekçe |
|---|---|---|---|
| Yeşil H | 40–85 | **55–95** | Pencereyi RAL 6037'nin (78,5) üstüne kaydır |
| Kırmızı/yeşil S tabanı | 140 / 120 | **150** | Beton S≈20–60, bedava marj |
| Kırmızı/yeşil V tabanı | 110 | **60** | Bulut/gölge payı |
| Siyah V tavanı | 45 | **85** | Plaketi içeri al |

---

## 4. SİYAH — eşikle çözülemeyen kısım

Siyah tavanını 85'e çıkarmak plaketi içeri alır, **ama gölgeli betonu da** (V 62–87).

Bunlar eşikle ayrılamaz: betonun albedosu (~0,30) ile RAL 9005'inki (~0,045) arasındaki
oran, güneş/gölge oranıyla neredeyse aynıdır. Fizik gereği çakışırlar.

Kamera 2–3° eğimli, yani **nadir** → güneş tepedeyken **drone'un kendi gölgesi** ROI'ye
düşer ve 10 m'de ~45 px, yani plaketle **aynı boyuttadır**.

### Çözüm: siyah maskesine iki ucuz şekil kontrolü

Yalnızca siyaha uygulanır. Kırmızı ve yeşil betonda doygunlukla zaten temiz ayrıldığı için
onlara dokunulmadı.

**a) Extent** = `kontur_alanı / sınır_kutusu_alanı`

| | Extent |
|---|---|
| Kare plaket | **0,96** (ölçüldü) |
| Drone gölgesi (gövde + 4 kol + pervane halkaları) | **0,31** |

Eşik: `BLACK_MIN_EXTENT = 0.75`

**b) Kadraj kenarına değme reddi** — bina/direk/operatör gölgeleri genelde taşar,
ortadaki plaket taşmaz. `BLACK_REJECT_BORDER = True`

### Hangisi neyi yakalıyor

| Yanlış pozitif | Extent | Kenar |
|---|---|---|
| Drone'un kendi gölgesi | ✅ | ❌ |
| Bina / direk / ağaç gölgesi | kısmen | ✅ |
| Operatör gölgesi | ✅ | çoğunlukla ✅ |
| **Betonda kompakt kare koyu leke** | ❌ | ❌ |

> **Artık risk:** Son satır kalıyor. Kalibrasyon olmadan tamamen çözülemez.

---

## 5. KAMERA — AWB ve pozlama kilidi

Eski hâli:
```python
cap.set(cv2.CAP_PROP_AUTO_WB, 1)           # AWB AÇIK
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # V4L2'de 0.75 = otomatik
```

- **AWB** sahne içeriğine göre tonları kaydırır
- **Oto-pozlama** siyah plaket kadrajı doldurunca parlaklığı artırıp onu griye çevirir

### Uygulanan çözüm: otomatik ayarla, sonra kilitle

Sabit bir pozlama değeri kodlamak yerine (o değer sadece seçildiği ışıkta doğru olurdu):

1. AWB + oto-pozlama **açık** başlat, `AE_SETTLE_S = 2.0` saniye oturmasını bekle
2. Sürücünün yakınsadığı değerleri **geri oku**
3. Manuel moda geç ve o değerleri **kilitle**

Böylece günün ışığına uyum sağlar ama **uçuş sırasında kaymaz**.

Ayrıca: `FOURCC=MJPG` (720p+ için USB bant genişliği şart), `BUFFERSIZE=1` (gecikme).
V4L2'de sıralama önemli — FOURCC çözünürlükten, `AUTO_*` manuel değerden önce gelmeli.

---

## 6. DAYANIKLILIK

| Sorun | Düzeltme |
|---|---|
| Seri portta `write_timeout` yok → yazma tıkanırsa döngü sonsuza kilitlenir | `write_timeout=0.5` |
| `if not ok: break` → tek kare aksarsa görev biter | 15 başarısız okumada kamerayı **yeniden aç** (pozlama kilidi de yeniden uygulanır) |
| Kayıt yok | `color_log.csv` + renk değişiminde kare kaydı (`captures/`, `MAX_CAPTURES` sınırlı) |
| `last_detected_conf` sadece renk *değişince* güncelleniyordu → ekrandaki güven bayat | Her karede güncelleniyor |
| Terminal ~100 Hz yeniden çiziliyordu | `DISPLAY_INTERVAL_S = 0.2` |

### Arka plan taraması — kalibrasyonun yapılabilen yarısı

Plaketler yok ama **beton ve gölge sahada olacak**. ROI'nin medyan H/S/V değeri 2 saniyede
bir loglanıyor. Bu, *neyin plaket olmadığını* kesin söyler ve eşiklerin neyi dışarıda
bırakması gerektiğini verir. Plaket gerektirmez.

---

## 7. DOĞRULAMA

Sentetik sahne testi (1280×720 beton + taneli gürültü, pozlama betona kilitli):

| Senaryo | Beklenen | Bulunan | Extent |
|---|---|---|---|
| Boş beton | BELIRSIZ | BELIRSIZ | — |
| RAL 3026 kırmızı 45 px | KIRMIZI | KIRMIZI | — |
| RAL 6037 yeşil 45 px | YESIL | YESIL | — |
| RAL 9005 siyah 45 px | SIYAH | SIYAH | 0,96 |
| Sadece drone gölgesi | BELIRSIZ | BELIRSIZ | eleme |
| Bina gölgesi (kenardan) | BELIRSIZ | BELIRSIZ | eleme |
| Siyah plaket + drone gölgesi | SIYAH | SIYAH | 0,96 |
| Yeşil plaket + drone gölgesi | YESIL | YESIL | eleme |

**8/8 geçti.**

> Test, `drone_color.py` bir betik olduğu (import etmek sonsuz döngüyü başlatır) için
> kaynağı "Main Loop" başlığına kadar kesip ayrı modül olarak yürütür. Dosyaya dokunmaz.

---

## 8. YAPILMAYANLAR

Kullanıcı kararıyla kapsam dışı:

| | Neden |
|---|---|
| **Kalibrasyon betiği** | Plaketler elde yok |
| **RC darbe ölçümü sağlamlaştırma** | Sahada sorun yaşanmıyor |
| **Lab + ΔE sınıflandırma** | Köklü çözüm ama en çok iş; HSV eşikleri şimdilik yeterli |
| **Tam şekil analizi (4 köşe)** | Extent + kenar kontrolü yeterli ve daha dayanıklı |

---

## 9. SAHADA DOĞRULAMA

1. **Kalkıştan önce** terminalde `[CAM] ... exposure locked at N | WB locked at M`
   satırını görün — görmüyorsanız sürücü manuel modu kabul etmemiştir.
2. **Beton üzerinde** `Scene median HSV` satırını not edin. Bu, eşiklerin dışarıda
   bırakması gereken değerdir.
3. **Siyah için vekil test:** mat siyah mukavva / koyu düz bir yüzey. RAL 9005 değil ama
   "koyu, nötr, kare" davranışını doğrular. `Black extent` satırı 0,75'in üstünde olmalı.
4. **Drone gölgesini kadraja alın** (güneş tepedeyken) — SIYAH raporlanmamalı.
5. **Uçuş sonrası** `color_log.csv` ve `captures/` klasörünü inceleyin. Eşikler burada
   gerçek veriyle düzeltilir.

> Kalibrasyon olmadan **siyah zayıf halka olarak kalıyor.** Yapılanlar onu
> "hiç çalışmıyor"dan "muhtemelen çalışıyor"a taşıdı; kesinlik sahada doğrulanmalı.
