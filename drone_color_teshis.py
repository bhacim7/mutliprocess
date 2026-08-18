# -*- coding: utf-8 -*-
"""
drone_color teshis araci - TEK BASINA calisir.

Amac: "hicbir renk tespit edilmiyor" sorununun KAMERADAN mi ESIKLERDEN mi
geldigini kesin olarak soylemek. GPIO ve seri port kullanmaz, sadece kamera.

Drone'daki Jetson'a kopyalayip calistirin:

    python3 drone_color_teshis.py                 # merkezdeki 100x100 kutuyu olcer
    python3 drone_color_teshis.py --box 200       # olcum kutusunu buyut
    python3 drone_color_teshis.py --no-lock       # pozlama/WB kilidini denemeden calis
    python3 drone_color_teshis.py --shots 5       # 5 kare ortalamasi

Olcmek istediginiz yuzeyi (plaket, mukavva, beton) kadrajin ORTASINA getirin.
Betik ekrana ciziyor: kameranin gercekte ne verdigi, olcum kutusundaki gercek
HSV degerleri, ve bu degerlerin ESKI/YENI esiklerin hangisinden gectigi.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

# --- Ayni ayarlar: drone_color.py ile birebir ---
CAM_INDEX = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
AE_SETTLE_S = 2.0

ESKI = {
    "red1": ((0, 140, 110), (10, 255, 255)),
    "red2": ((170, 140, 110), (179, 255, 255)),
    "green": ((40, 120, 110), (85, 255, 255)),
    "black": ((0, 0, 0), (179, 70, 45)),
}
YENI = {
    "red1": ((0, 150, 60), (10, 255, 255)),
    "red2": ((170, 150, 60), (179, 255, 255)),
    "green": ((55, 150, 60), (95, 255, 255)),
    "black": ((0, 0, 0), (179, 80, 85)),
}


def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)) if v else "?"


def open_camera(lock=True):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        return None, {}

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    t0 = time.time()
    while time.time() - t0 < AE_SETTLE_S:
        cap.read()

    settled_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    settled_wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)

    locked = False
    if lock:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if settled_exp and settled_exp > 0:
            cap.set(cv2.CAP_PROP_EXPOSURE, settled_exp)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        if settled_wb and settled_wb > 0:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, settled_wb)
        # Gercekten uygulandi mi? Surucu sessizce yok sayabilir.
        locked = abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - 0.25) < 0.01

    info = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "ae_mode": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "auto_wb": cap.get(cv2.CAP_PROP_AUTO_WB),
        "wb": cap.get(cv2.CAP_PROP_WB_TEMPERATURE),
        "settled_exp": settled_exp,
        "settled_wb": settled_wb,
        "locked": locked,
    }
    return cap, info


def mask_hits(hsv, ranges):
    """Her renk icin olcum kutusundaki piksel yuzdesi."""
    n = hsv.shape[0] * hsv.shape[1]
    red = cv2.inRange(hsv, ranges["red1"][0], ranges["red1"][1]) \
        + cv2.inRange(hsv, ranges["red2"][0], ranges["red2"][1])
    green = cv2.inRange(hsv, ranges["green"][0], ranges["green"][1])
    black = cv2.inRange(hsv, ranges["black"][0], ranges["black"][1])
    return {
        "KIRMIZI": 100.0 * int(red.sum() / 255) / n,
        "YESIL": 100.0 * int(green.sum() / 255) / n,
        "SIYAH": 100.0 * int(black.sum() / 255) / n,
    }


def why_rejected(h, s, v, ranges):
    """Olcum kutusunun MEDYAN rengi hangi esikte takiliyor?"""
    out = []
    for name in ("KIRMIZI", "YESIL", "SIYAH"):
        if name == "KIRMIZI":
            lo1, hi1 = ranges["red1"]
            lo2, hi2 = ranges["red2"]
            hue_ok = (lo1[0] <= h <= hi1[0]) or (lo2[0] <= h <= hi2[0])
            reasons = []
            if not hue_ok:
                reasons.append(f"H={h} aralik disi (0-{hi1[0]} / {lo2[0]}-179)")
            if s < lo1[1]:
                reasons.append(f"S={s} < {lo1[1]}")
            if v < lo1[2]:
                reasons.append(f"V={v} < {lo1[2]}")
        elif name == "YESIL":
            lo, hi = ranges["green"]
            reasons = []
            if not (lo[0] <= h <= hi[0]):
                reasons.append(f"H={h} aralik disi ({lo[0]}-{hi[0]})")
            if s < lo[1]:
                reasons.append(f"S={s} < {lo[1]}")
            if v < lo[2]:
                reasons.append(f"V={v} < {lo[2]}")
        else:
            lo, hi = ranges["black"]
            reasons = []
            if s > hi[1]:
                reasons.append(f"S={s} > {hi[1]}")
            if v > hi[2]:
                reasons.append(f"V={v} > {hi[2]}")
        out.append((name, reasons))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=100, help="merkezdeki olcum kutusu (px)")
    ap.add_argument("--shots", type=int, default=3, help="ortalanacak kare sayisi")
    ap.add_argument("--no-lock", action="store_true", help="pozlama/WB kilidini deneme")
    args = ap.parse_args()

    cap, info = open_camera(lock=not args.no_lock)
    if cap is None:
        print("HATA: kamera acilamadi (cv2.VideoCapture(%d))." % CAM_INDEX)
        print("  -> 'ls /dev/video*' ile cihazi, 'v4l2-ctl --list-devices' ile adini kontrol edin.")
        sys.exit(1)

    print("=" * 66)
    print("1) KAMERA GERCEKTE NE VERIYOR")
    print("=" * 66)
    print(f"  cozunurluk    : {info['w']}x{info['h']}   (istenen {CAM_WIDTH}x{CAM_HEIGHT})")
    print(f"  FOURCC        : {info['fourcc']}   (MJPG olmali)")
    print(f"  FPS           : {info['fps']:.0f}")
    print(f"  AE modu       : {info['ae_mode']:.2f}   (0.25=manuel, 0.75=otomatik)")
    print(f"  pozlama       : {info['exposure']:.0f}   (oto-oturma sonrasi: {info['settled_exp']:.0f})")
    print(f"  AWB           : {info['auto_wb']:.0f}   (0=kapali/kilitli)")
    print(f"  WB sicakligi  : {info['wb']:.0f}   (oto-oturma sonrasi: {info['settled_wb']:.0f})")
    if not args.no_lock:
        print(f"  KILIT         : {'BASARILI' if info['locked'] else 'BASARISIZ - surucu manuel modu kabul etmedi'}")
    if info["w"] != CAM_WIDTH or info["h"] != CAM_HEIGHT:
        print("  !! Kamera istenen cozunurlugu vermedi. USB bant genisligi veya MJPG destegi.")

    frames = []
    for _ in range(max(1, args.shots)):
        ok, f = cap.read()
        if ok:
            frames.append(f.astype(np.float32))
        time.sleep(0.05)
    cap.release()

    if not frames:
        print("\nHATA: kare okunamadi.")
        sys.exit(1)

    frame = np.mean(frames, axis=0).astype(np.uint8)
    H, W = frame.shape[:2]
    b = max(10, min(args.box, min(H, W) // 2))
    cx, cy = W // 2, H // 2
    box = frame[cy - b // 2: cy + b // 2, cx - b // 2: cx + b // 2]
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)

    med = np.median(hsv.reshape(-1, 3), axis=0).astype(int)
    p5 = np.percentile(hsv.reshape(-1, 3), 5, axis=0).astype(int)
    p95 = np.percentile(hsv.reshape(-1, 3), 95, axis=0).astype(int)
    bgr_med = np.median(box.reshape(-1, 3), axis=0).astype(int)

    print()
    print("=" * 66)
    print(f"2) MERKEZDEKI {b}x{b} KUTUNUN GERCEK RENGI")
    print("=" * 66)
    print(f"  BGR medyan : B={bgr_med[0]:3d}  G={bgr_med[1]:3d}  R={bgr_med[2]:3d}")
    print(f"  HSV medyan : H={med[0]:3d}  S={med[1]:3d}  V={med[2]:3d}")
    print(f"  HSV %5-%95 : H={p5[0]}-{p95[0]}  S={p5[1]}-{p95[1]}  V={p5[2]}-{p95[2]}")

    print()
    print("=" * 66)
    print("3) BU RENK HANGI ESIKTEN GECIYOR")
    print("=" * 66)
    for tag, ranges in (("ESKI", ESKI), ("YENI", YENI)):
        hits = mask_hits(hsv, ranges)
        print(f"  --- {tag} esikler ---")
        for name, reasons in why_rejected(med[0], med[1], med[2], ranges):
            pct = hits[name]
            if not reasons:
                print(f"    {name:<8} GECER   (kutunun %{pct:.1f}'i eslesti)")
            else:
                print(f"    {name:<8} KALIR   (%{pct:.1f}) <- {'; '.join(reasons)}")

    out = "teshis_kare.jpg"
    vis = frame.copy()
    cv2.rectangle(vis, (cx - b // 2, cy - b // 2), (cx + b // 2, cy + b // 2), (0, 255, 255), 2)
    cv2.imwrite(out, vis)
    print()
    print(f"Kare kaydedildi: {os.path.abspath(out)}  (sari kutu = olcum alani)")
    print()
    print("NASIL OKUNUR:")
    print("  * Cozunurluk/FOURCC yanlissa veya KILIT BASARISIZ ise  -> sorun KAMERADA")
    print("  * Kamera dogru ama renk 'KALIR' diyorsa                -> sorun ESIKTE,")
    print("    yukaridaki H/S/V medyanini bana gonderin, esikleri ona gore ayarlayalim")


if __name__ == "__main__":
    main()
