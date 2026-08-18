# IDA USV — Task 2 Navigasyon ve Telemetri: Analiz ve Değişiklik Kaydı

**Son güncelleme:** 2026-08-12
**Durum:** Uygulandı ve sahada doğrulandı. Task 2 çarpmadan ve parkurdan çıkmadan tamamlanıyor.
**Sıradaki test (hiçbiri suda denenmedi):**
(a) GPS noktaları parkurun kenarına atılarak koridor kapağının sınavı ·
(b) GCS röle butonları — kumanda etkileşimi ·
(c) drone renk zinciri (`DRONE_ACTIVE` yeni açıldı) ·
(d) iz ömrü 20 s — costmap'te `tracks:` sayısı ·
(e) costmap videosu — `.npz` düşüyor mu, `mp4v` kodlayıcısı teknede var mı

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
12. [GCS'ten röle kontrolü](#12-gcsten-röle-kontrolü--uygulandı)
13. [Telemetri akışını hızlandırma (öneri)](#13-telemetri-akışını-hızlandırma--öneri-uygulanmadi)
14. [Drone renk zinciri](#14-drone-renk-zinciri--uygulandı)
15. [Costmap videosu](#15-costmap-videosu--uygulandı)

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

Konum hatasını düşürmenin üç yolu:

| Yol | Durum |
|---|---|
| **İz ömrünü uzatmak** (`ObjectMemoryManager`) | ✅ **uygulandı** — 5 s → 20 s, hız korumalarıyla birlikte. Ayrıntı aşağıda. Suda denenmedi |
| **Pusula kaynağını değiştirmek** (`HEADING_SOURCE` → `'FC'`/`'FUSED'`) | ❌ 0,50 m ≈ 10 m'de 2,9° başlık hatası. Cube manyetometresi kalibre edilmediği için **ertelendi** |
| **Yakın mesafe tepkisel katman** | ❌ Kamera kerterizi ~1° hassas ve pusuladan/GPS'ten bağımsız. **Kullanıcı reddetti**: A\*'ın akıcılığını bozar |

#### İz ömrü — neyin düzeltildiği (ve neyin düzelmediği)

**Belgenin önceki hâli yanlıştı**, düzeltiliyor. İki iddia hatalıydı:

*"Tek sayı değişikliği"* — değil. Eşleştirme, izi **hızıyla ileri taşıyarak** yapıyor ve o hız
tek kare aralığından (~0,1 s) hesaplanıyor. 0,2 m konum gürültüsünü 0,1 s'ye bölmek,
demirli bir şamandıraya **2 m/s** görünür hız atfediyor; yumuşatma sonrası **0,43 m/s**
kalıcı sahte hız ölçüldü. 20 saniyeye yayılınca tahmin şamandıradan metrelerce uzağa
düşüyor, 2,5 m eşleşme yarıçapını aşıyor, iz yine ikizleniyor — üstelik bayat kopya
4 kat uzun yaşıyor.

*"Etkin hata 0,50 → 0,22'ye yaklaşır"* — yaklaşmaz. `x ← 0.8x + 0.2z` bir **üstel hareketli
ortalama**; zaman sabiti ~5 örnek, yani 10 Hz'de yarım saniye. Koşunun tamamını ortalamıyor,
**son okumaları takip ediyor**. Ömrü uzatmak bunu değiştirmez.

Uygulanan üç değişiklik:

| | Ne | Config |
|---|---|---|
| a1 | Hız sadece yeterli aralıkta güncellenir | `OBJECT_VEL_MIN_DT_S = 0.5` |
| a2 | İleri tahmin sınırlandı | `OBJECT_VEL_MAX_PREDICT_S = 1.0` |
| b | İz ömrü 5 s → 20 s | `OBJECT_MEMORY_S = 20.0` |

Simülasyon — tek **sabit** şamandıra, 8 s aralarla 6 kez görünüyor, 0,2 m gürültü,
doğru cevap **1 kimlik**:

| | Kimlik | Sahte hız |
|---|---|---|
| Eski (5 s, sınırsız hız) | 6 | 0,43 m/s |
| **Sadece (b)** — 20 s, sınırsız hız | **4** | 0,43 m/s |
| Yeni (20 s + a1 + a2) | **1** ✅ | 0,01 m/s |

Orta satır önemli: ömrü tek başına uzatmak işin ancak bir kısmını kurtarıyor.

**Asıl kazanç doğruluk değil, kararlılık.** İz geri dönüşümü bittiği için A\* haritasındaki
engel artık ~0,50 m zıplamıyor — 2026-08-12 koşusunda 292 kez zıplama fırsatı vardı.

**Bedeli:** yanlış tespit 5 s yerine 20 s hayalet kalır. Arkadakiler yerel costmap'ten
kendiliğinden çıkar (20 s'de 36 m), ama öndeki sahte engel planlayıcıyı daha uzun dolaştırır.
`seen` sayacı hazır lever olarak duruyor.

**Alfa filtresine bilerek dokunulmadı (c).** Düz koşan ortalama, uzaktan yapılan yüksek
hatalı tespitlere yakındakilerle eşit ağırlık verirdi; mevcut EMA son (yakın, daha iyi)
ölçümleri takip ederek kazara doğru olanı yapıyor. Düzgünü **mesafeye göre ağırlıklandırmak**,
o da önce ölçüm istiyor.

**Ölçüm tuzağı:** başarılı olursa `between-track spread` satırı **seyrelir veya kaybolur** —
o satır sadece birden fazla ize sahip şamandıralar için basılıyor. **Kaybolması sonuçtur,
eksik veri değil.** Bakılacak satır:

```
[COSTMAP]   tracks: 292   buoys (clustered): 30
                    ▲                     ▲
              30-60'a düşmeli        aynı kalmalı
```

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
| `2cb2a49` | Şişirme 0,45 → **0,50**, `TASK2_APPROACH_LATERAL_M` 3 → **8** (sahada doğrulandı) |
| `60bdaf8` | **GCS'ten röle kontrolü** (§12) |
| `1538042` | **Drone renk zinciri** — kapalı çevrim, paket kurtarma, `DRONE_ACTIVE` (§14) |
| `3653cc7` | **Costmap videosu** + telemetri önerileri (§13, §15) |
| `f31b409` | **İz ömrü 20 s + hız korumaları** (§9) |

---

## 12. GCS'TEN RÖLE KONTROLÜ — **uygulandı**

**Durum:** Kod yazıldı, mantık test edildi, **sahada denenmedi.**
**Firmware:** ArduRover **4.6.2** → `RELAY_STATUS` mevcut, gerçek durum okunabiliyor.

### Eklenenler

| Yer | Ne |
|---|---|
| `GCSv1000.py` | **⚡ RÖLE AÇ** ve **⛔ RÖLE KAPAT** butonları (onaysız) + gerçek durum etiketi |
| `telem_process.py` | Pakete `relay` alanı (−1 = bilinmiyor) |
| `nav_process.py` | `set_relay` komutu, `target_id` filtresi, teyitli tekrar |
| `MainSystem2.py` | `set_relay()` / `get_relay_state()` + `RELAY_STATUS` isteği @2 Hz |
| `config.py` | `RELAY_INSTANCE`, `RELAY_COMMAND_RETRIES`, `RELAY_RETRY_INTERVAL_S`, `VEHICLE_ID` |

Mevcut **ACİL DURDUR butonuna dokunulmadı.**

### Kritik ayrıntılar

**`MAV_CMD_DO_SET_RELAY`'in instance parametresi 0 tabanlı.** `RELAY1` için `param1 = 0`. `1` göndermek var olmayan ikinci röleyi adresler ve sessizce hiçbir şey olmaz. Test edildi.

**Durum "son gönderilen komut" değil, aracın kendi raporu.** Kumanda röleyi bizden habersiz değiştirebildiği için `RELAY_STATUS`'tan okunuyor. Mesaj gelmezse GCS **"bilinmiyor"** yazıyor — tahmin etmiyor.

**Tekrar mekanizması.** `command_long_send` ACK beklemiyor; kaybolan güvenlik komutu fark edilmez. En fazla 5 kez, 0,25 s arayla, `RELAY_STATUS` teyit edince erken durur. Nav döngüsü bloklanmıyor.

**`target_id` filtresi.** `CommandReceiver` telsizde duyduğu her satırı kuyruğa koyuyor — drone'a (id 3) giden röle komutunu tekne de uygulardı. Artık `VEHICLE_ID` ile filtreleniyor. `target_id` olmayan eski komutlar geriye uyumlu şekilde işleniyor.

### Cube parametreleri — hiçbiri değişmedi

```
SERVO9_FUNCTION = -1     AUX1 GPIO oldu          } röleyi TANIMLAR
RELAY1_PIN      = 50     röle 1 = AUX1           } (kim komut verirse versin)
RELAY1_FUNCTION = 1      röle aktif              }
RELAY1_DEFAULT  = 0      açılışta KAPALI  <- güvenli, dokunmayın
RC7_OPTION      = 28     kumanda anahtarı          <- sadece kumanda yolu
```

İlk üçü röleyi tanımlıyor; `RC7_OPTION` bir *girdi*, MAVLink `DO_SET_RELAY` başka bir girdi. **İkisi ArduPilot içindeki aynı duruma yazıyor, son yazan kazanıyor.**

### Neden `nav_process`, `telem_process` değil

Bir seri portu aynı anda tek proses açabilir. Cube'a giden `/dev/ttyACM0` **`nav_process`'in elinde**; `telem_process` telsize (`/dev/ttyUSB0`) bağlı ve Cube'a hiç erişimi yok.

- `telem_process` = **tercüman** (GCS JSON'u ↔ iç dünya)
- `nav_process` = **uygulayıcı** (Cube'a dokunan her şey)

`set_gps`, `set_manual`, `set_target_color`, `emergency_stop` de aynı yolu izliyor. Röle yeni bir desen değil, listeye yeni bir üye.

**Bedeli:** `nav_process` takılırsa GCS röle komutu Cube'a ulaşmaz. Bu yüzden **kumanda anahtarı asıl acil durdurma olarak kalıyor** — Jetson'dan, Python'dan, telsizden bağımsız donanım yolu.

### Doğrulama (mantık düzeyinde)

| Test | Sonuç |
|---|---|
| `DO_SET_RELAY` instance 0, state 0/1 | ✅ |
| `RELAY_STATUS` yok → `None` (bilinmiyor) | ✅ |
| `present`/`on` bitmask çözümleme | ✅ 3/3 |
| Teyit gelince tekrar durur | ✅ 2 gönderimde durdu |
| Teyit gelmezse 5 kez dener, 0,25 s arayla | ✅ |
| Drone'a (id 3) giden komut | ✅ yok sayıldı |
| `target_id` olmayan eski komut | ✅ işlendi |
| Hızlı AÇ→KAPAT | ✅ son komut kazandı |

Telemetri paketi `relay` alanıyla **342 B** → 5 Hz'de %29,7 seri doluluk. Röle komutu 48 B.

### Sahada bakılacaklar

1. **RÖLE AÇ/KAPAT** → motor gücü kesiliyor/veriliyor mu
2. Durum etiketi **"bilinmiyor"da takılıyor mu** → takılırsa `RELAY_STATUS` gelmiyordur, `MAV_CMD_SET_MESSAGE_INTERVAL` isteği çalışmamış olabilir
3. **Kumandadan** röleyi değiştirin → GCS etiketi birkaç saniye içinde güncellenmelidir
4. GCS'ten kapatıp **kumanda anahtarını oynatın** → kumanda kazanmalı
5. Terminalde `Relay command not confirmed after 5 attempts` uyarısı çıkıyor mu

### Test edilmemiş üç durum

| Durum | Beklenen |
|---|---|
| Cube yeniden başlarsa | `RELAY1_DEFAULT = 0` → kapalı gelir |
| RC sinyali kesilip dönerse | ArduPilot anahtarları yeniden değerlendirebilir, sürüme göre değişir |
| Aynı anda iki komut | Son yazan kazanır (mantıken; sahada doğrulanmadı) |

---

## 13. TELEMETRİ AKIŞINI HIZLANDIRMA — **öneri, UYGULANMADI**

**Durum:** 2026-08-16 itibarıyla rafta. Aşağıdakilerin hiçbiri koda girmedi.

### Gözlem

Kapalı ortamda, üç telsiz yan yanayken GCS tekne paneli **2-3 saniyede bir** yenileniyor.
Tasarım 3,3 Hz (300 ms) yoklama, yani **10 kat yavaş**.

> Ölçüm uyarısı: bu gözlem paketteki `t_ms` alanına bakılarak yapıldı ve o alan
> `"%H:%M:%S"` biçiminde — **çözünürlüğü 1 saniye**. 300 ms ile 900 ms'yi ayırt edemez.
> Yani "2-3 s" kaba bir okuma; işe başlamadan önce gerçek ölçüm gerekiyor (D5).

### Gecikme bütçesi (hesaplandı)

| Adım | Süre |
|---|---|
| Yoklama aralığı | 300 ms |
| Hava gidiş-dönüş | ~50 ms |
| nav_process (25 Hz, kuyruk her turda boşaltılıyor) | 40 ms |
| telem_process (20 Hz) | 50 ms |
| 369 B @ 57600 seri | 64 ms |
| **Toplam en kötü** | **~500 ms** |

**500 ms ile 2-3 s arasındaki 5 kat fark açıklanamadı.** Çarpışma hesabı da yetersiz:
havadaki toplam meşguliyet saniyede ~51 ms, yani %5. %5 meşguliyet %55 kayıp üretmez.

### Kök neden (kısmen): üç telsiz, iki düğümlük TDMA

SiK firmware zamanı **iki** tarafa bölüyor. Üçüncü telsizin kendi dilimi yok, mecburen
birininkini paylaşıyor; ikisi aynı anda konuşursa **iki paket birden** ölüyor. Aynı
`NETID`'deki telsizler birlikte frekans atladığı için kazara ayrılmaları da mümkün değil.

Bu yazılımla çözülemez — firmware seviyesinde. Trafiği azaltmak etkisini hafifletir.

### Açıklanamayan farkın adayları

| # | Aday | Nasıl ayırt edilir |
|---|---|---|
| A | TDMA dilim bekleme | gönderilen yoklama = gelen cevap, ama geç |
| B | `Manager` sözlüğü tıkanması (4 işlem aynı anda) | teknede "komut geldi → paket çıktı" süresi uzun |
| C | Gerçek paket kaybı | gönderilen yoklama ≫ gelen cevap |
| D | Telsiz tampon / akış kontrolü | cevaplar kümeler hâlinde geliyor |

### Öneriler

**D5 — Ölçüm göstergesi.** GCS kendi saatiyle: gerçek Hz, en uzun boşluk, gönderilen
yoklama / gelen cevap, cevap gecikmesi. Teknede: `report_status` → paket çıkışı süresi.
**Dört adayı birbirinden ayıran deney bu.** Önce bu yapılmalı.

**D1 — Yoklamayı kaldır, tekne kendisi yayınlasın.** Şu an her güncelleme için havada
**iki** iletim gerekiyor ve ikisi de varmalı. Tek yönlü yayında bir tane.

| yön başına başarı | yoklama (p²) | yayın (p) |
|---|---|---|
| 0,90 | %81 | %90 |
| 0,70 | %49 | %70 |
| 0,55 | %30 | %55 |

Ayrıca havadaki iletim sayısı 9,6/s → 5,0/s'ye iner (az trafik = az çarpışma = `p` yükselir),
ve şu adımlar telemetri yolundan tamamen silinir: yoklama iletimi, `CommandReceiver`,
`mp.Queue`, `nav_process`, `Manager` sözlüğü. ArduPilot/MAVLink zaten böyle çalışır.

**D2 — Seri hız 57600 → 115200.** Seri hat şu an havadan yavaş: 5,76 KB/s'ye karşı
~12,5 KB/s etkin. 369 B paket havada 13 ms, kabloda **64 ms**. Darboğaz telsiz değil kablo.
Üç telsizin ayarı + `config.py` + GCS birlikte değişmeli.

**D3 — Paketi küçült (369 → ~250 B).** `"hlth":"GOOD"` sabit dize (15 B), görev adı uzun
metin yerine sayı, alan adları kısaltılabilir (GCS aynı anda güncellenir). %32 daha az hava
süresi.

**D4 — Drone'u sadece gerektiğinde hızlandır.** Sabit 3 Hz yerine: renk değişince 3 s
boyunca 3 Hz, sonra 1 Hz + GCS eşiği 6 s. Marj yine 6 paket, hava kullanımı ⅓'e iner.
(3 Hz sabit kararı teknenin kanalından çaldı.)

**D6 — Dördüncü telsiz (donanım, kesin çözüm).** GCS'e ikinci telsiz, drone ayrı
`NETID`/kanala. Tekne↔GCS temiz iki-düğüm bağlantısı olur, çarpışma imkânsızlaşır.
GCS kodunda `worker_2` altyapısı zaten duruyor, kullanılmıyor. Yedek telsiz varsa bu tek
başına konuyu kapatır ve D1-D4'e gerek kalmaz.

### Menzil endişesi — yersiz

40-50 m açık suda **şu ankinden kötü olması beklenmiyor**. RFD900x 1 W ile kilometrelerce
gidiyor. Üstelik kapalı ortam çok yollu sönümleme (duvar yansımaları) yüzünden daha kötü
olabilir. Sorun menzil değil, üç telsizin tek kanalda çarpışması — ve **bu mesafeyle
değişmiyor**.

### Yarış gerçeği

Task 2 ve 3 sırasında drone inmiş, tetik kapalı → **drone hiç konuşmuyor**, kanal tamamen
teknenin. Üçü birden aktifken yapılan test **en kötü durum**, sürekli hâl değil.

### Önerilen sıra

`D5 (ölç) → D1 (yayın) → ölçüme göre D2/D3/D4`. Yedek telsiz varsa `D6` hepsinin önüne geçer.

---

## 14. DRONE RENK ZİNCİRİ — **uygulandı**

**Durum:** Kod yazıldı, mantık doğrulandı, **uçurulmadı.**

### Zincir

```
drone_color.py        →  {"id":3, "drone_color":"SIYAH"}
GCSv1000.on_packet()  →  {"target_id":1, "cmd":"set_target_color", "color":"black"}
nav_process           →  shared_state['drone_target_color'] = "black"
Task 3                →  if cfg.DRONE_ACTIVE:  target_color = shared_state[...]
```

### Korunan üç kural

Bunlar baştan doğru tasarlanmıştı, hiçbir değişiklik bunlara dokunmadı:

1. **Sadece renk değişince gönder** — heartbeat aynı rengi tekrarlarken hat boş kalır
2. **`BELIRSIZ` iletilmez** — drone anlık kararsızlaşınca teknenin elindeki renk silinmez
3. **Son gelen kazanır, kilit yok** — `siyah → kırmızı` senaryosu kendi kendini düzeltir

### Kapatılan delikler

| Delik | Neydi | Çözüm |
|---|---|---|
| **`DRONE_ACTIVE = False`** | Tekne rengi alıyor, saklıyor, `Drone target color updated to ...` logluyor — **ama kullanmıyordu**. Task 3 `TASK3_KAMIKAZE_COLOR` ile gidiyordu. Zincir son adımda kapalıydı | `True` |
| **Kaybolan komut asla tekrarlanmıyor** | GCS bir kez gönderip kendi defterine "gitti" yazıyordu. O tek paket kaybolursa drone dakikalarca KIRMIZI raporlar, GCS "değişmedi" deyip **susardı** | Kapalı çevrim (aşağıda) |
| **Tekne yeniden başlarsa renk gider** | `shared_state` sıfırlanır, GCS yine susar | Aynı kapalı çevrim |
| **Drone paneli tekneden 4 kat hassas** | 1 Hz heartbeat + 2500 ms eşik = **3** ardışık kayıpta "koptu". Tekne 5 Hz yoklamada 12 gerektiriyordu | 3 Hz + 4000 ms → **12** |
| **Poz ayarı drone'u susturuyor** | `calibrate_exposure()` ana döngüde 1,5-4 s kare okuyor; o sürede hiç paket gitmiyordu. Eşikten uzun → **her poz ayarında garantili "Bağlantı Koptu"**, telsiz sağlamken | `link_keepalive` geri çağrısı okuma döngülerine geçirildi |
| **Bozuk tekne paketi drone'unkini yutuyor** | GCS satır sonuna göre bölüyor. Havada bozulan tekne paketinin satır sonu gelmeyince yarım satır tamponda kalıyor, sıradaki paket ona yapışıyor, `json.loads` patlıyor, **ikisi de çöpe** gidiyordu. Tekne çok daha sık yayınladığı için kurban genelde **sapasağlam gelen drone paketi** oluyordu | `_parse_or_recover()` |
| **GCS seri portunda `write_timeout` yok** | `flush()` akış kontrolünde süresiz bloklar; aynı thread okuma da yaptığı için **iki LED birden** kırmızıya döner | `write_timeout=0.5` |

### Kapalı çevrim — röledeki ilkenin aynısı

Tekne telemetrisi artık **elinde tuttuğu rengi ve kaynağını** taşıyor (~27 B):

| Alan | Değerler |
|---|---|
| `tcol` | rengin kendisi, boş = henüz gelmedi |
| `tsrc` | `"cfg"` = `DRONE_ACTIVE` kapalı, config karar veriyor · `"drone"` = drone karar veriyor |

GCS ikisini karşılaştırıyor; uyuşmazsa komutu **tekrar gönderiyor** (2 s hız sınırıyla).
Her şey yolundayken **tamamen sessiz** — "sadece değişince gönder" kuralı bozulmuyor.

> *Son gönderdiğin komuta değil, aracın bildirdiği gerçek duruma bak.*

### GCS'te TASK3 RENK göstergesi

İHA panelinde, drone'un gördüğü rengin **altında** — üstteki drone'un gördüğü, alttaki
**teknenin elindeki**. İkisi aynı şey değil ve fark tam olarak mesele:

| Durum | Ekranda | Renk |
|---|---|---|
| `DRONE_ACTIVE = False` | `SIYAH (config)` | gri |
| `True`, renk gelmedi | **`RENK YOK`** | turuncu |
| Renk ulaştı | `KIRMIZI (drone)` | yeşil |

Suda "siyahı kovalıyor çünkü drone öyle dedi" ile "siyahı kovalıyor çünkü drone hiç
ulaşmadı, config öyle diyor" **birebir aynı görünür** ve çok farklı şeylerdir.

### Yol boyunca çıkan gizli tehlike

`CommandReceiver` telsizde duyduğu **her satırı** kuyruğa koyuyordu ve `nav_process`
`target_id`'ye bakmadan işliyordu. **Drone'a gönderilen bir röle komutunu tekne de
uygulardı.** Artık `VEHICLE_ID` ile filtreleniyor; `target_id` taşımayan eski komutlar
çalışmaya devam ediyor.

### Reddedilen öneri

**Boştayken heartbeat** (tetik kapalıyken de "buradayım, boştayım" yollamak) — GCS'in
"koptu" ile "boşta"yı ayırmasını sağlardı, maliyeti %0,4. **Kullanıcı reddetti:** drone
sadece 1. görevde birkaç dakika uçuyor, tetik elle yönetiliyor, hat bütçesi buna
harcanmasın.

### Doğrulama (mantık düzeyinde)

Paket kurtarma **8/8** — yarım tekne + tam drone, `MEVCUT_KONUM` ortasından kesik, üç paket
zinciri, kurtarılamayan çöp. **İç içe süslü parantez tuzağı geçti:** sadece paket başlangıcı
aranıyor, körü körüne süslü parantez aransaydı `{"lat":...,"lon":...}` geçerli JSON olarak
kabul edilip **id'siz bir paket** olarak arayüze verilirdi (ve sessizce tekneninki sayılırdı).

Kapalı çevrim **5/5** — uyuşmazlıkta tekrar, 2 s içinde sessiz, onayda susma, tekne yeniden
başlayınca tekrar.

### Hat bütçesi

| | Önce | Sonra |
|---|---|---|
| Hat doluluğu | %33,8 | **%25,3** |
| Drone kopma marjı | 3 ardışık | **12** |
| Tekne kopma marjı | 12 | 8 (etkisiz: %4 kayıpta 8 ardışık ≈ 10⁻¹¹) |
| Yoklama | 200 ms | 300 ms |

### Sahada bakılacaklar

1. Otonom komutu → **`RENK YOK`** yazmalı (artık `SIYAH (config)` değil)
2. Trigger aç, plaket göster → **yeşil** `KIRMIZI (drone)`
3. Drone LED'i poz ayarı sırasında artık kırmızıya düşmemeli
4. Durum çubuğunda `tekrar gönderildi` çıkarsa → paket kaybı var **ama kendini onarıyor**
5. `DRONE_ACTIVE` açık: drone hiç uçmazsa Task 3 yine `TASK3_KAMIKAZE_COLOR` ile gider
   (güvenli geri düşüş) — ama GCS `RENK YOK` yazacağı için farkında olursunuz

---

## 15. COSTMAP VİDEOSU — **uygulandı**

**Durum:** Kod yazıldı, **teknede çalıştırılmadı.**

### Neden PNG'den video üretilemez

PNG **tek kare**: koşunun bittiği andaki son hâl. İçinde zaman yok. Betik PNG'yi değil,
**ayrı bir ham veri dosyasını** okuyor.

### Eksik olan tek şey zamandı

Kayıt üç katman tutuyordu ve ikisinde zaman bilgisi yoktu:

| Katman | Zaman | Not |
|---|---|---|
| `track` | dolaylı | 5 Hz düzenli, sıra ≈ zaman |
| `observations` | **yoktu** | 1 Hz toplu ekleniyor, **turlar arası ayraç yok**, tur boyu değişken → sıradan zaman çıkarılamaz |
| `objects` | **yoktu** | sözlük, sadece **en son** konum |

Çözüm: gözleme zaman damgası + ize ilk görülme anı.

```
observations: (x, y, cid)  →  (x, y, cid, t)
objects[id]:  + 't0'
```

10 dakikalık koşuda ~70 KB. `render()` ve `_scatter_radii()` içindeki iki açma satırı
güncellendi; **üretilen PNG piksel piksel aynı.**

### Şamandıralar nihai konumlarıyla belirir

Kümeleme sonda **bir kez** yapılır; her kare "ilk görülme anı ≤ T olan kümeleri" çizer.
Şamandıralar **keşif sırasıyla belirir ve bir daha kıpırdamaz.**

Alternatifi (her karede yeniden kümeleme) daha dürüst ama nokta titrer, bazen iki küme
birleşip biri kaybolur — jüriye **hata gibi** görünür. Ve 600 kare × yeniden kümeleme.

**Kaybedilen bilgi yok:** tahminin nasıl yakınsadığı zaten PNG'de — soluk gözlem bulutu ve
saçılım halkaları tam olarak bunu gösteriyor.

> **İş bölümü:** PNG **belirsizliği** gösterir, video **kronolojiyi**.

Küçük incelik: "ilk görülme" tek bir gürültülü tespitle erkene kaymasın diye **3'ten az
gözlemi olan izler** bu hesaba katılmıyor.

### Render neden pahalı olurdu

`_scatter_radii()` her gözlemi aynı sınıftaki her nesneyle karşılaştırıyor — kare başına
~0,5-1 s. 600 kare = **10 dakika.** İki katmanlı çözüm:

- **Saçılım halkaları videodan çıkarıldı** — istatistiksel teşhis, PNG'de kalıyor. Pahalı kısım buydu
- **Kümeleme bir kez** → kare başına O(1)
- Gözlem bulutu ve iz **artımlı** çiziliyor: her kare öncekinin üstüne ekleniyor, sıfırdan değil

### Örnekleme ≠ oynatma

```
ÖRNEKLEME  1 Hz   verinin çözünürlüğü
OYNATMA   15 fps  izleme hızı
```

İkisi de 1 olsa 10 dakikalık koşu = 10 dakikalık video. 15 fps ile **~40 saniye**.
Her karede geçen süre yazılı, gerçek zamanlama okunabilir kalıyor.

### Kapanma yolu — asıl risk

`save()` **`atexit` ve sinyal işleyicisinden** çağrılıyor. Ctrl+C'ye basınca program hemen
kapanmıyor, önce `save()` çalışıyor. Bugün ~1 s, videoyla 15-30 s.

Asıl tehlike süre değil, **yeniden girme**:

```python
self.save()                          # ← 30 saniye burada
signal.signal(sig, signal.SIG_DFL)   # ← varsayılan davranış ANCAK BURADA geri geliyor
```

O 30 saniye boyunca SIGINT hâlâ aynı işleyiciye gidiyor ve `_saved` bayrağı henüz
kurulmadığı için **ikinci Ctrl+C render'ı en baştan başlatıyor.** Program Ctrl+C ile
kapanmaz hâle gelir. Bugün pencere ~1 s olduğu için isabet ettirmek imkânsıza yakın;
30 saniyede **kaçınmak** imkânsız olurdu — kullanıcının doğal tepkisi tam olarak budur.

Ayrıca MP4'ün dizini (`moov atom`) `release()` ile yazılır. Süreç ondan önce ölürse dosyada
kareler vardır ama dizin yoktur → *"yarım video"* değil, **hiç açılmayan dosya**.

**Korunma:** PNG **önce** yazılır, `_saved` **hemen sonra** kurulur, video en sonda.
İkinci Ctrl+C artık `if self._saved: return` ile anında dönüyor — video yarım kalır ama
PNG garanti ve program **temiz kapanır**.

### Neden ayrı betik

Ham veri her zaman yazılıyor (milisaniyeler, risksiz). Video isteğe bağlı.

| | Kapanışta render | Ayrı betik |
|---|---|---|
| Kapanma süresi | +30 s | **değişmez** |
| Ctrl+C riski | var | **yok** |
| Ayarı beğenmezseniz | koşuyu tekrarlayın | **betiği tekrar çalıştırın** |
| Nerede çalışır | Jetson | **laptop (hızlı)** |
| Kodlayıcı yoksa | veri kayıp | **veri elinizde** |

### Teknede kontrol edilecek

Kapanışta `.npz` düşüyor mu, ve `mp4v` kodlayıcısı var mı — ikincisi tek satırla sınanır
(`VideoWriter` açılmalı ve dosya sıfır bayttan büyük olmalı). Açılmazsa `MJPG` + `.avi`
kullanılır: dosya büyük ama her yerde çalışır.

### Küçük ama canını yakacak ayrıntılar

- **Sabit tuval:** MP4 sabit boyut ister. Sınırlar tüm veriden bir kez hesaplanır — yan etkisi güzel: harita boş çerçeveye **doğru büyür**
- **Çift sayı zorunluluğu:** `mp4v` tek sayılı boyutlarda sessizce bozuk dosya üretir
- **Çözünürlük:** 0,2 m/px'te tipik parkur ~500×300 px, video için küçük → ölçekleme gerekiyor

---

## KAPANIŞ NOTU

Bu süreçte en çok tekrar eden hata, **yanlış istatistiğe bakmak** oldu: şişirme değeri iz-içi saçılmadan ayarlandı ve tekne şamandıralara çarptı. Doğru sayı (izler arası fark) ancak recorder'a kümeleme eklenince ortaya çıktı.

İkinci tekrar eden hata, **durum makinesi değişikliğini sahada test etmek**: yaklaşma noktası iki kez yanlış yazıldı, ikisi de ancak sudayken görüldü. Artık konum konum simülasyonla doğrulanıyor.

Üçüncü tekrar eden hata, **tek atışa güvenmek**: röle komutu da renk komutu da bir kez
gönderilip "gitti" sayılıyordu. İkisi de aynı ilkeyle düzeltildi — *son gönderdiğin komuta
değil, aracın bildirdiği gerçek duruma bak.* Kayıp bir paketin sessizce yanlış bir duruma
kilitlemesi, telsiz hattında istisna değil kuraldır.

Dördüncüsü, **"tek sayı değişikliği" sanmak**: iz ömrünü 5 s'den 20 s'ye çıkarmak tek
başına işin ancak yarısını kurtarıyordu, çünkü eşleştirme gürültüden doğan sahte bir hızla
ileri tahmin yapıyordu. Bu belgede o iddia bir süre **yanlış** olarak durdu; simülasyon
düzeltti.

Her ayar değişikliğinin gerekçesi `config.py` içinde yorumda duruyor — bir değeri
değiştirmeden önce oradaki hesabı okuyun.

**Suda denenmemiş olanlar** (2026-08-17): röle kumanda etkileşimi, drone renk zinciri,
iz ömrü 20 s, costmap videosu, kenar GPS noktalarıyla koridor kapağı sınavı. §13'teki
telemetri önerilerinin **hiçbiri uygulanmadı**.
