# IDA USV — Task 2 Navigasyon ve Telemetri: Analiz ve Değişiklik Kaydı

**Son güncelleme:** 2026-08-12
**Durum:** Uygulandı ve sahada doğrulandı. Task 2 çarpmadan ve parkurdan çıkmadan tamamlanıyor.
**Sıradaki test:** GPS noktaları parkurun daha da kenarına atılarak koridor kapağının sınavı.

**Kardeş belge:** Drone plaket rengi tespiti için → [`ANALIZ_DRONE_COLOR.md`](ANALIZ_DRONE_COLOR.md)

---

## İÇİNDEKİLER

1. [Nasıl buraya geldik](#1-nasıl-buraya-geldik)
2. [Sorun 1 — Task 2'de daire çizme](#2-sorun-1--task-2de-daire-çizme)
3. [Sorun 2 — Telemetri GCS'e ulaşmıyor](#3-sorun-2--telemetri-gcse-ulaşmıyor)
4. [Sorun 3 — Parkurdan çıkma](#4-sorun-3--parkurdan-çıkma)
5. [Şişirme değeri — üç kez yanlış ayarlandı](#5-şişirme-değeri--üç-kez-yanlış-ayarlandı)
6. [Yaklaşma noktası](#6-yaklaşma-noktası)
7. [Costmap recorder](#7-costmap-recorder)
8. [Mevcut ayarlar](#8-mevcut-ayarlar)
9. [Bilinen sınırlar ve ertelenenler](#9-bilinen-sınırlar-ve-ertelenenler)
10. [Saha kontrol listesi](#10-saha-kontrol-listesi)
11. [Değişiklik günlüğü](#11-değişiklik-günlüğü)

---

## 1. NASIL BURAYA GELDİK

Başlangıç şikâyeti: *"Task 2'de tekne bazen çok güzel gidiyor, bazen bir anda daire çizmeye başlıyor, her denemem aynı değil."* Ayrıca telemetri GCS'e akmıyordu ve `final_costmap` okunamaz haldeydi.

Yöntem: her adımda **ölçüm**. Ekran kayıtları kare kare incelendi, HUD'daki her sayısal alan okunup kodla eşleştirildi, değişiklikler sentetik ve gerçek veri üzerinde doğrulandı.

Bu belgede **ne yapıldığı kadar neyin yanlış yapıldığı** da yazılı — özellikle şişirme değeri iki kez yanlış ayarlandı ve ikisinin de sebebi kayıtlı.

---

## 2. SORUN 1 — TASK 2'DE DAİRE ÇİZME

**Kaynak:** `task2Claude.mp4` (66,99 s · 2009 kare), 2 fps tam tarama + kritik pencerelerde 4–5 fps.

### Belirleyici kanıt — 23:47:59

```
ACI_FARKI: 4          <- hedef tam burnunun ucunda
ARKA_SOL_PWM: 1100    <- tam sol doyum
ARKA_SAG_PWM: 1900
DUMEN_PWM:    1100
```

Hedef 4° ileride ama komut tam sol. Bu ancak Pure Pursuit'in **iç hatası ≈ −73°** ise mümkün — yani 77°'lik fark hedeften değil, **takip edilen yol düğümünden** geliyordu.

Devamında HDG 156 → 141 → 105 → 46 → 357 → 269 → 141: toplam **360°'den fazla dönüş**. Ve 23:48:08'de 0,25 saniye içinde `1100/1900 → 1831/1528 → 1900/1109` — pervane başına **800 PWM ters sıçrama**.

### Kök nedenler ve düzeltmeler

| # | Sorun | Düzeltme | Dosya |
|---|---|---|---|
| 1 | `find_lookahead_point`, yoldan lookahead'den fazla sapınca **en yakın düğümü** döndürüyordu. Düğüm teknenin arkasında olabiliyordu (ölçüldü: **−140,2°**). | Dik izdüşüm + yay boyu ilerleme. Geriye hedef geometrik olarak imkânsız. | `planner.py` |
| 2 | `PURE_PURSUIT_KP = 5.5` → pervaneler **26,9°**'de doyuyordu (Task 1/3'teki PID: 98,7°). | KP → 2.0, KD → 0.8. Doyum **74°**. | `config.py` |
| 3 | `MAX_PWM_CHANGE` config'de vardı, **hiçbir kod kullanmıyordu**. | `apply_motor_mixer`'da uygulandı. Acil fren / SIGTERM / disarm muaf. | `nav_process.py` |
| 4 | `get_horizontal_speed()` `LOCAL_POSITION_NED` okuyordu; ArduRover yayınlamıyor → `HIZ: 0.0` → lookahead kalıcı 0,8 m. | `VFR_HUD.groundspeed` + yedekleri. | `MainSystem2.py` |
| 5 | PP dalında `SPOT_TURN_THRESHOLD` karşılığı yoktu → 150° hatayla tam seyir gazı = **tanım gereği daire**. | Kosinüs gaz kısma + histerezisli spot-turn (60°/35°). | `nav_process.py` |
| 6 | A\* `None` dönünce eski yol temizlenmiyordu. | Yol atılıyor + yaş denetimi; planlama 5 Hz'e ayrıldı. | `nav_process.py` |
| 7 | 5 kanal × 50 Hz = **250 `COMMAND_LONG`/s**; GPS 1,5 s dondu, FPS 30→17. | Döngü 25 Hz, `set_servo` dedup, IPC kısıldı. | `MainSystem2.py`, `nav_process.py` |
| 8 | `ObjectMemoryManager` renk eşleşmesini **birebir** arıyordu → tek şamandıra iki engel. | Renk **oylanan** özellik. | `camera_process.py` |

### Doğrulama

| | Eski | Yeni |
|---|---|---|
| Tekne (1.2, 1.0), 3 düğümlü yol | **−140,2°** | −51,3° |
| Lookahead 1,8 m'de en kötü hata | 61,9° | **53,8°** |
| Doyum eşiği | 26,9° | **74,0°** |
| 1100→1900 tam geçiş | 0,25 s | **0,53 s** |

53,8° < 74° doyum **ve** < 60° spot-turn → normal takipte ne doyum ne gereksiz yerinde dönüş.

**Sahada doğrulandı:** 2026-08-12 koşusunda `ACI_FARKI 2…5`, `DUMEN 1457–1522`. Daire yok.

---

## 3. SORUN 2 — TELEMETRİ GCS'E ULAŞMIYOR

### Tek sayı: 2880 bayt

`write_timeout = 0.5 s` @ 57600 baud → 0,5 saniyede en fazla **2880 bayt**.

Ölçülen paket: sabit alanlar 447 B + `GÖREV_NOKTALARI` **513 B** (her pakette tekrar!) + nesne başına **200 B**.

| Nesne sayısı | Paket | Sonuç |
|---|---|---|
| 5 | 1 960 B | ✅ |
| **10** | **2 960 B** | ❌ kıl payı aşıyor |
| 20 | 4 960 B | ❌ |

Yazım yarıda kesiliyor → `\n` gitmiyor → GCS parçayı sonraki pakete yapıştırıp `JSONDecodeError` → **ikisini birden** atıyor. Yeşil LED sadece başarılı parse'ta yandığı için hiç yanmıyordu.

**Asimetri teşhisin ispatıydı:** komutlar (40–500 B) hep gidiyordu, telemetri (3–7 KB) hiç gelmiyordu. RF sorunu olsaydı iki yön de bozulurdu.

### Düzeltmeler

- **T1** `objects` paketten çıkarıldı — en büyük alandı ve **GCS bu anahtarı hiç okumuyor**
- **T2** Görev noktaları sadece değişince + 10 s tazeleme
- **T3** Float yuvarlama (`json.dumps` `repr` ile 17 hane yazıyordu) + kompakt ayraç
- **T4** `TelemetrySender.send()` **bitiremeyeceği yazımı başlatmıyor**; timeout olursa `reset_output_buffer()` + tek `\n` ile alıcıyı resenkronize ediyor
- **Ek** GCS yoklama 500 → **200 ms**

| Paket | 2 Hz doluluk | Havada (@224 kbps) |
|---|---|---|
| Eski (12 nesne) 3 360 B | %117 ❌ | 107 ms |
| Yeni 708 B (waypoint'li) | %24,6 ✅ | 25 ms |
| Yeni 332 B | %11,5 ✅ | 12 ms |

### RFD900x ayarları — sorun yoktu

| Ayar | Değer | Değerlendirme |
|---|---|---|
| Air Speed | 224 | ✅ hava darboğaz değil (~28 KB/s) |
| Baud (seri) | 57600 | en dar halka buydu |
| Mavlink | Off | ✅ ham JSON için doğru |
| ECC | kapalı | ✅ |

**3 telsiz:** SiK P2P firmware iki düğüm için tasarlı, üçüncü çarpışma yaratır. Ama paket 107 ms → 25 ms'ye inince çarpışma penceresi **4 kat daraldı**; firmware değişikliği gerekmedi.

### Yol boyunca çıkan gizli hata

`set_gps` nav_process'in `cfg` modülünü değiştiriyor, ama `mp.set_start_method('spawn')` yüzünden telem_process'in **kendi kopyası** var. Tekne operatörün noktalarına giderken GCS'e **config varsayılanları** yollanıyordu. `shared_state['mission_points']` köprüsüyle çözüldü.

### Sonuç (ölçüldü)

| | Değişim | Medyan aralık | En uzun donma |
|---|---|---|---|
| GCS paneli | 199 / 88 s = **2,3 Hz** | 367 ms | 2,10 s (tek) |
| Kamera HUD | 22 Hz | 33 ms | 0,97 s |

Paket kaybı bitti. 2,3 Hz zaten yoklama hızıydı; 200 ms'ye çekilerek ~5 Hz'e çıkarıldı.

---

## 4. SORUN 3 — PARKURDAN ÇIKMA

**Kaynak:** `IDAparkurdanÇıkanDeneme.mp4`. GPS noktaları bilerek parkur kenarına yakın verildi.

### Ne oldu

02:17:34–46 arası kamera: `Yellow 1.6m` solda, `Orange 3.6m` sağda. Tekne **sarı engel ile turuncu sınır arasında sıkıştı** ve dışarıdan dolaştı.

Sebep basit: **A\* için turuncu, sarıdan farksız bir nokta engel.** İki nokta engelin arasından geçmek yerine dışarıdan dolaşmak hem kısa hem tamamen boş.

### Elenen yaklaşımlar

| | Neden elendi |
|---|---|
| **Düz hat + sabit bant** (T2 ilk→son çevresinde ±X m) | Parkur kavisli olabilir; bant gerçek koridorun içine girip **meşru kaçışı engeller** |
| **Turuncuları birleştirip duvar örmek** | Tek kaçırılan tespit **duvarda delik** açar, A\* tam oradan çıkar. Ayrıca hangi şamandıranın hangi zincire ait olduğunu çözmek gürültülü tespitlerle zor |

### Uygulanan: yanal kapak

> **Hiçbir turuncu şamandıranın dış tarafından geçme.**

Sert duvar değil **maliyet gradyanı** — koridor gerçekten kapalıysa kilitlenmek yerine çıkabilir.

**Eksen = T2 ilk nokta → T2 son nokta.** Bu, komitenin verdiği **tek garanti** (*"bu hat parkurdan çıkmaz"*) ve tekneyle birlikte dönmediği için çapraz girişte sol/sağ ayrımı bozulmaz.

Üç koruma:

1. **Boylamsal pencere** — sadece teknenin civarındaki turuncular kapak koyar (−5 m … +15 m). 25 m ilerideki biri şu anki yanal hareketi kısıtlamaz.
2. **Bin bazlı profil** — kapak eksen boyunca dilimlere ayrılıyor, **kavis otomatik takip ediliyor** ve koridor genişliği parametresi gerekmiyor; şamandıralardan okunuyor.
3. **Çift taraf şartı** — kapak ancak **hem solda hem sağda** onaylı turuncu varsa kurulur. Çapraz girişte veya bir zincir kaybolduğunda hiçbir kısıt uygulanmaz.

**Onay:** şamandıra `CORRIDOR_CONFIRM_SIGHTINGS` (3) kez görülmüş ve son 1,5 s içinde görülmüş olmalı. Sahte bir turuncu koridoru daraltır, dar koridor da yok koridor kadar zararlı.

**Yedek basamak:** kapakla yol bulunamazsa **kapaksız** tekrar denenir. Dışarı çıkan ama engelden kaçan bir yol, engel-**kör** PID'e düşmekten iyidir.

### Doğrulama

| Test | Sonuç |
|---|---|
| Gerçekçi 12 senaryo (rastgele sarılar) | **12/12 yol bulundu, 5 ms, sınır ihlali 0** |
| Tek taraf görünür (çapraz giriş) | Kapak **kurulmuyor** ✅ |
| İki taraf görünür | İçeri maliyet 0, dışarı 3,0 ✅ |
| Yedek basamak | Kapaklı yol yokken kapaksız buluyor ✅ |

**Sahada:** 2026-08-12, noktalar ortada → parkurdan çıkma yok. **Kenar testi henüz yapılmadı.**

---

## 5. ŞİŞİRME DEĞERİ — ÜÇ KEZ YANLIŞ AYARLANDI

Bu bölüm ayrı duruyor çünkü **iki kez ters yöne ayarlandı** ve sebepleri öğretici.

### Geometri tek satırda

> **gövde–şamandıra boşluğu = `INFLATION_MARGIN_M` − konum hatası**

A\* teknenin **merkezini** haritadaki konumdan `BUOY + ROBOT + INFLATION` uzakta tutar; gerçek şamandıra `hata` kadar daha yakın olabilir.

### Kronoloji

| Değer | Sonuç | Sebep |
|---|---|---|
| **0,55** | 1,5 m geçitler **görünmez** | Boşluk negatif → tek çıkış dışarıdan dolaşmak → parkurdan çıkma |
| **0,25** | İki sarıya **çarptı** | **Yanlış istatistikten** ayarlandı: iz-içi saçılma (0,11 m) *kesinlik*, konum doğruluğu değil |
| **0,45** | **Sürttü** | Boşluk −0,05 m |
| **0,50** | ✅ Çarpma yok, parkurdan çıkma yok | Boşluk 0,00 m |

### Kritik ders: kesinlik ≠ doğruluk

Recorder iki ayrı sayı basıyor:

```
per-track scatter  : 0.22 m   <- KESINLIK (bir izin kendi tutarlılığı)
between-track      : 0.50 m   <- DOGRULUK (aynı şamandıranın izleri arası fark)
```

`ObjectMemoryManager` bir şamandırayı 5 s görmezse izi atıp yeni iz açıyor. 126 iz ↔ 39 fiziksel şamandıra. **Doğruluk için izler arası fark bakılmalı** — dar geçit dokunup dokunamayacağını belirleyen sayı bu.

### Çözülemeyen çelişki

| | Gereken |
|---|---|
| Çarpmamak | `INFL` > 0,50 |
| 1,5 m geçitleri geçmek | `INFL` ≤ 0,35 |

**0,50 m konum hatasıyla bağdaşmıyorlar.** 0,50 uzlaşmadır: 1,8 m'den geniş geçitler açık, daha darları kapalı — ve kapalı olduğunda koridor kapağı dışarı çıkmayı engelliyor.

Bunu kapatmanın tek yolu **konum hatasını düşürmek** (0,50 m ≈ 10 m'de 2,9° başlık hatası). Bkz. bölüm 9.

---

## 6. YAKLAŞMA NOKTASI

### Ne çözüyor

Tekne koridorun **yanında ve ilerisinde** olduğunda, giriş noktasına en kısa yol sınır zincirinin **1. ve 2. şamandırası arasından** geçebilir (aralık ~4 m, şişirmeden sonra ~1,8 m boşluk). Tekne parkura **ağzından değil yan duvarından** girer.

### Ne yapıyor

Giriş noktasının **eksende 12 m gerisine** sanal nokta koyar; tekne önce oraya, sonra eksene hizalı olarak ağızdan içeri girer.

### Ne zaman devreye giriyor

Sadece eksenden yanal sapma > `TASK2_APPROACH_LATERAL_M` iken. Ağzın önünde hizalıysanız hiç devreye girmez.

### İki kez yanlış yazıldı

| Sürüm | Hata |
|---|---|
| İlk | Bırakma eşiği **girişe** olan mesafeye bakıyordu (8 m) ama nokta 12 m gerideydi → varınca koşul doğru kalıyor, **kendi altındaki noktayı hedefliyor**. Tekne Task 2'ye hiç başlayamadı. Kayıt: `HEDEFE_MESAFE` 0,6–3,5 m arası salınım, `HEDEF_HDG` 143→273→296→177→100→256 |
| İkinci | "Yaklaşma noktasına 3 m kaldıysa bırak" → nokta geçilince yarıçaptan çıkıp **geri dönüyordu** (salınım) |
| Nihai | **Yanal sapmaya** bağlı. Yaklaştıkça sapma monoton azaldığı için anahtar **bir kez kapanır, açılmaz** |

### Doğrulama

| Senaryo | Sonuç |
|---|---|
| Eksen üzerinde önden yaklaşma | Doğrudan girişe, dolambaç yok |
| 45° oblik, 30 m → 3 m | Hizalanana kadar yaklaşma, sonra giriş |
| Yaklaşma → giriş | **Hedef değişimi = 1**, salınım yok |
| Koridorun içinde | Giriş, geriye çekmiyor |

`TASK2_APPROACH_LATERAL_M` **3,0 → 8,0** yapıldı (kullanıcı tercihi): koruma kalıyor ama günlük kullanımda görünmüyor.

---

## 7. COSTMAP RECORDER

Yalnızca teşhis; kontrole etkisi yok. API değişmedi.

### İlk hâlindeki sorunlar

- `update()` her 1 Hz turda hafızadaki *tüm* nesneleri yeniden çiziyor, **hiçbir şey silinmiyordu** → 60 s'de ~900 daire, tek şamandıra **14,4 m** lekeye dönüşüyordu
- Çizim yarıçapı 3 px × 0,5 m/px = **3 m çap** (gerçek şamandıra 0,5 m)
- Yol 1 Hz örnekleniyordu → 10 saniyelik daire **10 köşeli çokgen**
- Tuval sabit 500 m, parkur ~55 m → PNG'nin **%99,7'si boş siyah**

### Yeni tasarım

Metre cinsinden veri biriktirilip `save()` anında bir kez render — artımlı tuval büyütme mantığı tamamen kalktı.

| Katman | |
|---|---|
| `track` | Tekne izi, 5 Hz |
| `objects` | id → son konum, **konuma göre kümelenmiş** (126 iz → 39 şamandıra) |
| `observations` | Ham gözlemler, soluk bulut |

Çözünürlük 0,2 m/px, tuval otomatik sığdırma, başlangıç/bitiş işaretleri, ölçek çubuğu, her nesneye **RMS saçılma halkası**.

### Terminal çıktısı — asıl kazanç

```
[COSTMAP]   track points: 864   tracks: 126   buoys (clustered): 39   raw sightings: 1079
[COSTMAP]   between-track spread per buoy (ACCURACY):
[COSTMAP]     orange  n=12  mean 0.50 m  max 0.85 m
[COSTMAP]   position scatter (RMS), A* clearance = 0.90 m:
[COSTMAP]     orange  n=94  mean 0.22 m  max 1.70 m
```

**İkinci blok doğruluk, üçüncü blok kesinlik.** Şişirme ayarı ikincisine bakılarak yapılır.

### Bir uyarı

Recorder **proses ömrü boyunca** biriktirir. Kodu kapatmadan 8-10 koşu yaparsanız hepsi üst üste biner. Kötü görünen bir harita her zaman kötü kod demek değil — bir koşuda tekne yerinde döndüyse harita da onu doğru gösterir.

---

## 8. MEVCUT AYARLAR

```
INFLATION_MARGIN_M        0.50      BUOY_RADIUS_M             0.25
ROBOT_RADIUS_M            0.40      A* yasak yarıçapı         1.15 m
PURE_PURSUIT_KP           2.0       PURE_PURSUIT_KD           0.8
PURE_PURSUIT_MAX_LOOKAHEAD 2.5      MAX_PWM_CHANGE            60
NAV_LOOP_HZ               25        A_STAR_GOAL_PROJECTION_M  10.0
A_STAR_PLAN_DIVISOR       5         A_STAR_TIME_BUDGET_S      0.12
ENABLE_TASK2_CORRIDOR     True      CORRIDOR_PENALTY          3.0
CORRIDOR_WINDOW_BACK_M    5.0       CORRIDOR_WINDOW_AHEAD_M   15.0
CORRIDOR_CONFIRM_SIGHTINGS 3        CORRIDOR_REQUIRE_BOTH_SIDES True
TASK2_APPROACH_OFFSET_M   12.0      TASK2_APPROACH_LATERAL_M  8.0
HEADING_SOURCE            'ZED'     GCS polling               200 ms
```

**Türetilen değerler (0,50 m konum hatasıyla):**
- Gövde–şamandıra boşluğu: **0,00 m** (sürtme sınırı)
- Geçilebilen en dar yüzey boşluğu: **1,80 m**

---

## 9. BİLİNEN SINIRLAR VE ERTELENENLER

### Kapatılamayan çelişki

0,50 m konum hatasıyla "çarpmamak" ve "1,5 m geçit geçmek" aynı anda sağlanamıyor. Şu an 1,8 m altı geçitler kapalı; koridor kapağı bunun parkurdan çıkmaya dönüşmesini engelliyor.

Konum hatasını düşürmenin üç yolu, **hiçbiri uygulanmadı**:

| Yol | Not |
|---|---|
| **Pusula kaynağını değiştirmek** (`HEADING_SOURCE` → `'FC'`/`'FUSED'`) | 0,50 m ≈ 10 m'de 2,9° başlık hatası. Cube manyetometresi kalibre edilmediği için ertelendi |
| **İz ömrünü uzatmak** (`ObjectMemoryManager` 5 s → 20-30 s) | Şamandıralar sabit; alfa filtresi ortalamaya devam ederse etkin hata 0,50 → 0,22'ye yaklaşır. **En ucuz seçenek, henüz denenmedi** |
| **Yakın mesafe tepkisel katman** | Kamera kerterizi ~1° hassas ve pusuladan/GPS'ten bağımsız. Kullanıcı tarafından reddedildi: A\*'ın akıcılığını bozar |

### Diğer ertelenenler

| | Neden |
|---|---|
| **GCS'in drone'u dönüşümlü yoklaması** | 3 düğümlü RFD çarpışmaları; paket küçülünce risk azaldı |
| **Turuncu sınır duvarı (zincir birleştirme)** | Yanal kapak daha sağlam; tespit gürültüsü duvarda delik açar |
| **A\* `nav_map` + yol dökümü** | Planlayıcının ne gördüğünü izlemek; kullanıcı istemedi |

### Gizli hatalar (düzeltilmedi)

- `T2_ZONE_MID` ile `T2_ZONE_END` config'de **birebir aynı** — GCS'ten gönderildiği için maskeleniyor
- Task 2'de kırmızı (cid 0) ve yeşil (cid 4) şamandıralar **engel sayılmıyor**
- `LIDAR_KORIDOR_KP`, `HYBRID_STEP_DIST`, `NAV_MODE`, `CAM_RES`, `SHOW_LOCAL_WINDOW` tanımlı ama kullanılmıyor

---

## 10. SAHA KONTROL LİSTESİ

### Kalkıştan önce
- [ ] GCS'te yeşil LED yanıp sönüyor, terminalde `Write timeout` yok
- [ ] HUD'da `HIZ` sıfırdan farklı
- [ ] GPS noktaları gönderildikten sonra haritada **gönderdiğiniz** konumlarda görünüyor

### Koşu sırasında
- [ ] `ACI_FARKI` 30° iken PWM'ler **doymuyor** (1100/1900 görülmemeli)
- [ ] `FPS` sürekli 30, `IDA_KONUM` donmuyor

### Koşudan sonra — terminal
```
[NAV_PROCESS] Corridor cap left no route - replanning without it.
```

| Bu satır | Parkurdan çıkma | Yorum |
|---|---|---|
| yok | yok | ✅ kapak çalıştı |
| **var** | var | Koridor gerçekten kapalıydı → `INFLATION_MARGIN_M` düşür |
| yok | **var** | Kapak hiç kurulmadı → turuncular onaylanmamış → `CORRIDOR_CONFIRM_SIGHTINGS` 3 → 2 |

### Koşudan sonra — costmap
- [ ] `between-track spread` satırı: 0,50 m civarında mı, düştü mü?
- [ ] `buoys (clustered)` sayısı gerçek şamandıra sayısına yakın mı?

### Ayar kuralları
| Belirti | Yapılacak |
|---|---|
| Şamandıraya çarpıyor | `INFLATION_MARGIN_M` +0,05 |
| Dar geçitleri hiç denemiyor, dolaşıyor | `INFLATION_MARGIN_M` −0,05 |
| Girişte gereksiz dolambaç | `TASK2_APPROACH_LATERAL_M` yükselt |
| Yan duvardan giriyor | `TASK2_APPROACH_LATERAL_M` düşür |

---

## 11. DEĞİŞİKLİK GÜNLÜĞÜ

| Commit | Konu |
|---|---|
| `a8bccec` | Task 2 doyumu + telemetri kırpılması (11 madde + T1–T4) |
| `38f54ab` | Costmap recorder yeniden yazımı |
| `fe5cf45` | drone_color: RAL eşikleri, çözünürlük, dayanıklılık |
| `da0ae38` | drone_color: yakın plakette alan sınırı regresyonu |
| `f99243b` | drone_color: gerçek HSV raporu |
| `0f00a17` | drone_color: siyah=kırmızı düzeltmesi + video kaydı |
| `663d1c7` | drone_color: ölçülmüş eşikler + seviye tabanlı RC trigger |
| `3012ac7` | drone_color: siyah şekil kapısı |
| `e67dd76` | **Task 2 koridor kapağı** + şişirme 0,25 + yaklaşma noktası + GCS 200 ms + recorder kümeleme |
| `90c2657` | Yaklaşma noktası kilitlenmesi düzeltildi |
| `f12b191` | Şişirme 0,25 → 0,45 (yanlış istatistikten ayarlanmıştı) |
| *(bu)* | Şişirme 0,45 → **0,50**, `TASK2_APPROACH_LATERAL_M` 3 → **8** (sahada doğrulandı) |

---

## KAPANIŞ NOTU

Bu süreçte en çok tekrar eden hata, **yanlış istatistiğe bakmak** oldu: şişirme değeri iz-içi saçılmadan ayarlandı ve tekne şamandıralara çarptı. Doğru sayı (izler arası fark) ancak recorder'a kümeleme eklenince ortaya çıktı.

İkinci tekrar eden hata, **durum makinesi değişikliğini sahada test etmek**: yaklaşma noktası iki kez yanlış yazıldı, ikisi de ancak sudayken görüldü. Artık konum konum simülasyonla doğrulanıyor.

Her ayar değişikliğinin gerekçesi `config.py` içinde yorumda duruyor — bir değeri değiştirmeden önce oradaki hesabı okuyun.
