# -*- coding: utf-8 -*-
# Gerekli kütüphaneleri içe aktar
import cv2
import numpy as np
import time
import datetime
import csv
import queue
import serial
import json
import os
import threading
import Jetson.GPIO as GPIO

# ==========================================================================
# TUNABLES - hepsi burada, sahada tek yerden ayarlanır
# ==========================================================================

# --- GPIO and Telemetry Settings ---
# FİZİKSEL pin numaralandırması (GPIO.BOARD) kullanılıyor; 40-pin header diyagramındaki
# numaraların aynısı.
# Kaynak: Orange Cube AUX1 (SERVO9), ArduPilot relay olarak yapılandırıldı.
#   SERVO9_FUNCTION = -1   (AUX1'i GPIO yap)
#   RELAY1_PIN      = 50   (50 = AUX1)
#   RC7_OPTION      = 28   (Relay On/Off)
# Kablolama: Cube AUX1 "S" -> Jetson pin 16, Cube AUX1 "-" -> Jetson pin 14 (GND).
# AUX1 "+" hattı BAĞLANMAZ. Pin ile GND arasına harici 10k pull-down direnç konur.
#
# Bu bir SEVİYE sinyali (relay), servo darbesi değil. Önceki sürüm kenar yakalayıp darbe
# genişliği ölçüyordu; relay çıkışında ölçülecek darbe olmadığı için tetik hiç çalışmıyordu.
TRIGGER_PIN = 16

# Relay HIGH iken sistemin aktif olmasını istiyoruz. Kumandada kanal 7 ters çalışıyorsa
# (tuş indirilince ch7 düşüyorsa) burayı 0 yapmak yeterli, başka yeri değiştirme.
ACTIVE_LEVEL = 1  # 1 = GPIO.HIGH, 0 = GPIO.LOW

# Gürültüye karşı basit yumuşatma: art arda bu kadar okuma aynı olmadan seviye değişmiş
# sayılmaz. Döngü ~100 Hz döndüğü için 3 okuma ≈ 30 ms.
DEBOUNCE_SAMPLES = 3
TIMER_THRESHOLD = 2.0   # seviye bu kadar süre sabit kalınca durum değişir

# RFD900x veya benzeri telemetri modülü USB üzerinden bağlıysa ttyUSB0 kalabilir.
# Eğer doğrudan Jetson'un TX/RX pinlerine (UART) bağlıysa "/dev/ttyTHS0" veya "/dev/ttyTHS1" olabilir.
TELEMETRY_PORT = "/dev/ttyUSB0"
TELEMETRY_BAUD = 57600

# --- Camera ---
CAM_INDEX = 0
# 640x480 -> 1280x720. At 10 m with the Brio's ~70 deg horizontal FOV a 50x50 cm plaque is
# about 45 px across here, versus 12 px at the old 320x240 processing size. 12 px is too
# small to judge shape and barely survives the morphology step.
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30

# Exposure handling. See calibrate_exposure() - the camera is MEASURED, not trusted.
AE_SETTLE_S = 1.5            # let the driver's own auto exposure settle before measuring
TARGET_V = 120               # aim for roughly mid grey
V_TOO_BRIGHT = 235           # above this the frame is blown out and every colour desaturates
V_TOO_DARK = 25              # below this nothing is distinguishable
# Manual exposure candidates, swept high to low. Units differ between drivers (usually
# 100 us steps), which is exactly why the value is chosen by measurement rather than
# calculation.
EXPOSURE_CANDIDATES = [1250, 640, 320, 160, 80, 40, 20, 10, 5, 3]
# If the picture goes bad mid-run, recalibrate - but only a few times, so a camera that
# simply cannot be controlled does not spin forever printing messages.
MAX_RECALIBRATIONS = 3
RECALIBRATE_AFTER_S = 5.0

# --- Detection geometry ---
# Centre window that is actually evaluated. Mapping where each colour appeared across the
# test footage settled this: the false positives live at the edges and the targets do not.
#     orange bench edge   65% of its pixels in the TOP row of a 3x3 split
#     green plaque        56% in the bottom-centre cell, 24% centre
# Shoes, the car and hands were all at the border too. 0.70 keeps the aiming area
# comfortable while cutting the frame edge where the clutter is.
ROI_RATIO = 0.70

# Absolute minimum blob size, in pixels. This used to be 0.5% of the ROI AREA, which
# scales with resolution - so raising the camera resolution moved the threshold by exactly
# the same factor and bought no extra range at all (measured: 9.0 m at 320x240 vs 9.5 m at
# 4K). An absolute floor is what actually turns resolution into detection range.
# 20x20 px = 400 px. At 1280x720 that corresponds to roughly 22 m altitude.
MIN_BLOB_AREA_PX = 400

# Upper area limit, BLACK MASK ONLY. Its only job is to reject a large shadow; a big red or
# green region simply IS the target and must never be capped.
#
# This was originally applied to all three colours at 0.25, which silently broke bench
# testing: at 1280x720 the ROI is 896x504, so 25% is a 336x336 px blob. A 1x1 m plaque
# exceeds that from closer than 2.70 m (a 0.5 m plaque from 1.35 m). Holding the drone over
# a plaque by hand is well inside that, so the plaque was rejected as "too big" and the
# result was BELIRSIZ - while a phone screen at arm's length stayed small enough to pass.
BLACK_MAX_AREA_FRAC = 0.60

# Bench/hand-held testing. The flight-altitude gates (upper area limit, border rejection)
# assume the plaque is a small object in a large frame. Up close it fills the frame and
# touches the edges, so those gates fight you. Set True while testing by hand, False to fly.
CLOSE_RANGE_TEST = False

# The black ceiling is this fraction of the frame's 95th-percentile V. P95 stands in for
# "the concrete", and it keeps doing so even when the plaque fills most of the frame, which
# a median would not. Checked against the measured footage:
#     shaded concrete alone   P95  95 -> ceiling 38, concrete median 67  -> not black OK
#     plaque in frame         P95 139 -> ceiling 56, plaque median  17  -> black     OK
#     sunlit concrete         P95 209 -> ceiling 84, concrete median 111 -> not black OK
# 0.40 was measured against a matte plaque. The glossy test plaque reflects the sky, so a
# large part of it is far brighter than its paint would suggest and 0.40 clipped most of it
# away. 0.55 recovers it; the parameter sweep over the labelled footage put the best black
# score here too.
BLACK_V_FRACTION = 0.55
BLACK_V_MIN = 25             # never go below this, or nothing is ever dark enough
BLACK_V_MAX = 110            # never go above this, whatever the frame looks like

# Minimum ABSOLUTE spread between the BGR channels (max - min) for a pixel to count as
# red or green.
#
# This is what stops the black plaque being reported as KIRMIZI. HSV saturation is a
# RATIO, S = (max-min)/max, so on a dark surface a tiny channel difference produces a huge
# S: BGR (30,35,70) - a black plaque under warm sunlight, or with the white balance a
# little off - comes out H=4 S=146 V=70, which clears both the red hue window and the S
# floor. Measured channel spreads:
#     black plaque reading as "red"   delta ~ 35-40
#     RAL 6037 green                  delta ~ 120
#     RAL 3026 red                    delta ~ 190
# A floor of 45 sits in the gap. It was 60, but the red plaque was measured down to
# delta=56 in shade - right on the edge - and shade is the worst case.
MIN_CHROMA_DELTA = 45

# --- Black plaque shape gate (see detect_color) ---
# Concrete makes red/green trivial to separate by saturation, but it makes BLACK hard:
# a shadow on concrete and the RAL 9005 plaque land at almost the same brightness, because
# concrete/plaque albedo (~0.30 / ~0.045) differ by about the same factor as sun/shade.
# No brightness threshold can split them, so the black mask gets two cheap shape checks.
# contour area / bounding-box area.
#
# 0.75 came from a synthetic clean square and it rejected EVERY real detection: measured
# 0/24 on the labelled black-plaque frames of renk_20260811_201821. Two reasons, both
# geometric rather than colour related:
#
#   * boundingRect is AXIS ALIGNED. A perfect square rotated 20-40 deg scores 0.50-0.60,
#     which is exactly the 0.53-0.62 band measured in the footage. The gate was mostly
#     measuring how the plaque happened to be turned.
#   * the test plaque is GLOSSY. It reflects the sky, so the bright part of it fails the
#     brightness test and the surviving mask is a crescent, not a square.
#
# 0.30 still throws out the drone's own shadow (~0.11-0.30 measured on the arms+props
# shape) while keeping the plaque: 21/24 on the same frames.
BLACK_MIN_EXTENT = 0.30
# Border rejection, but only for blobs BIG enough to be a ground shadow rather than a
# plaque. Rejecting everything that touched the edge was throwing the plaque away whenever
# it sat near the side of the window; rejecting nothing let a building shadow walk in from
# the edge. At altitude the plaque is a small object (a 50 cm plaque is ~0.45% of the ROI at
# 10 m), while a shadow entering the frame is an order of magnitude larger.
BLACK_REJECT_BORDER = True
BLACK_BORDER_MAX_FRAC = 0.05

# --- Video recording ---
# Same on/off idea as the boat's cfg.RECORD_VIDEO. The recorded frame carries the overlay
# (label, blob outline, live HSV), so reviewing it frame by frame shows what the detector
# actually decided rather than just what the camera saw.
RECORD_VIDEO = True
VIDEO_FPS = 15.0             # written at this rate; keep below the capture rate
VIDEO_MAX_QUEUE = 120        # drop frames rather than stall the loop if the disk lags

# --- Logging ---
LOG_ENABLED = True
LOG_CSV = "color_log.csv"
CAPTURE_DIR = "captures"
MAX_CAPTURES = 300           # disk guard
SCENE_STATS_INTERVAL_S = 2.0 # background survey cadence
# Frames were only saved on a colour CHANGE, so a run that is stuck on BELIRSIZ produced
# no evidence at all - exactly the case you most need to look at afterwards.
FAIL_CAPTURE_INTERVAL_S = 5.0
# Pixels above this saturation are treated as "actually coloured" for the readout below.
# Concrete sits at S 20-60.
CHROMA_MIN_S = 90

# --- Terminal ---
DISPLAY_INTERVAL_S = 0.2     # the old code redrew the whole screen at ~100 Hz

# ==========================================================================


# --- Serial ---
# write_timeout matters: without it a blocked write hangs this loop forever and the drone
# silently stops reporting. Same failure mode that was breaking the boat's telemetry.
master = None
try:
    master = serial.Serial(TELEMETRY_PORT, TELEMETRY_BAUD, timeout=1, write_timeout=0.5)
    print("Serial connection established successfully.")
except Exception as e:
    print(f"Serial connection failed: {e}")
    master = None


# --- Camera ---
def _median_v(cap, n=4):
    """Median V of the last few frames, or -1 if nothing could be read."""
    vals = []
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            small = cv2.resize(f, (160, 120))
            vals.append(float(np.median(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2])))
    return float(np.median(vals)) if vals else -1.0


def _set_auto_exposure(cap, auto):
    """
    Switch the driver's auto exposure on or off.

    OpenCV's encoding for CAP_PROP_AUTO_EXPOSURE is not stable across versions: older
    builds used 0.25 (manual) / 0.75 (auto), newer ones pass the raw V4L2 control value
    where 1 is Manual Mode and 3 is Aperture Priority. Writing 0.75 to a modern build
    rounds to 1 - MANUAL - so the camera silently kept whatever exposure it happened to
    have. That is what produced the blown-out V=254 frames in which no colour has any
    saturation left and every plaque reads BELIRSIZ.
    Write both encodings; the one the build does not understand is a harmless no-op.
    """
    for v in ((3, 0.75) if auto else (1, 0.25)):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)


def calibrate_exposure(cap):
    """
    Get a usable picture, whatever the driver does.

    Ask for auto exposure first and MEASURE the result. If the frame is blown out or
    crushed, take manual control and sweep EXPOSURE_CANDIDATES, keeping whichever lands
    closest to TARGET_V. Measuring instead of calculating means this does not depend on
    knowing the driver's exposure units or its control encoding.

    Returns a short description for the status line.
    """
    _set_auto_exposure(cap, True)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    t0 = time.time()
    while time.time() - t0 < AE_SETTLE_S:
        cap.read()

    v = _median_v(cap)
    if V_TOO_DARK <= v <= V_TOO_BRIGHT:
        return f"auto (V={v:.0f})"

    print(f"[CAM] Driver auto exposure gave V={v:.0f} - taking manual control and sweeping.")
    _set_auto_exposure(cap, False)

    best = None
    for e in EXPOSURE_CANDIDATES:
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        for _ in range(3):           # let the change take effect
            cap.read()
        mv = _median_v(cap, n=3)
        if mv < 0:
            continue
        print(f"[CAM]   exposure {e:>5} -> V={mv:.0f}")
        if best is None or abs(mv - TARGET_V) < abs(best[1] - TARGET_V):
            best = (e, mv)
        if mv < V_TOO_DARK:          # already too dark, going lower will not help
            break

    if best is None:
        return "manual sweep failed"

    cap.set(cv2.CAP_PROP_EXPOSURE, best[0])
    for _ in range(3):
        cap.read()
    final = _median_v(cap, n=3)
    if final > V_TOO_BRIGHT or final < V_TOO_DARK:
        return f"UNCONTROLLABLE (best V={final:.0f}) - check 'v4l2-ctl -d /dev/video0 -l'"
    return f"manual exposure={best[0]} (V={final:.0f})"


def open_camera():
    """Open the camera at the requested format and get the exposure into a usable range."""
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        return None

    # MJPG is required for 720p+ over USB; the default YUYV runs out of bandwidth.
    # Ordering matters on V4L2: FOURCC before resolution.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency down on a moving platform
    except Exception:
        pass

    how = calibrate_exposure(cap)

    print(f"[CAM] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} | exposure: {how}")
    return cap


cap = open_camera()
if cap is None:
    print("[CAM][ERROR] Camera could not be opened at startup.")


# --- Colour ranges (HSV, OpenCV convention: H 0-179, S 0-255, V 0-255) ---
# Ground is concrete, so there is no vegetation to confuse the green channel and concrete
# itself is achromatic (S ~20-60). That makes saturation a very clean discriminator for the
# two chromatic plaques, and it is the brightness thresholds that need the care.
USE_OUTDOOR = True

if USE_OUTDOOR:
    # ------------------------------------------------------------------------------
    # MEASURED from renk_20260811_194931.mp4 (983 frames, shaded concrete). These are
    # no longer derived from RAL swatch values - two earlier attempts to do that put the
    # windows in the wrong place, because what matters is what the CAMERA reports, not
    # what the paint chip says:
    #
    #     red plaque      87% of its pixels at H 168-179   S 180-200  delta 56-93
    #     green plaque    60% at H 45-54, peak 53-54       S 139-169  delta 80-101
    #     orange wood     77% at H 11-20                   S 150-205  delta 70-93
    #       (a bench edge in the test footage - the reason the old 0-15 red band and the
    #        55-95 green window between them produced "big green plaque -> KIRMIZI")
    #
    # The three targets sit far apart in hue, so the windows are deliberately generous:
    # the test was shot in SHADE and competition light may be direct sun, which shifts
    # hue a few degrees and can wash out saturation on a glossy surface.
    # ------------------------------------------------------------------------------

    # RED - the plaque lives on the wrap side. The 0-15 half is dropped entirely: the
    # plaque puts essentially nothing there, while orange/brown/skin-toned objects do.
    # A narrow 0-4 sliver is kept only so a hue that drifts past 179 is not lost.
    lower_red1, upper_red1 = (0, 100, 55), (4, 255, 255)
    lower_red2, upper_red2 = (163, 100, 55), (179, 255, 255)

    # GREEN - measured at H 45-57. 38-88 gives ~7 deg below and ~30 above, and still
    # stops well clear of the orange cluster that ends around H 24.
    lower_green, upper_green = (38, 100, 55), (88, 255, 255)

    # Saturation floor 100 and value floor 55: measured S was 139-205 and V 116-148 in
    # shade, so both have wide margin, and direct sun only raises them.
    #
    # NOTE: the S floor alone is NOT enough to keep dark surfaces out - see
    # MIN_CHROMA_DELTA below.

    # BLACK - no fixed brightness ceiling. Measured in shade:
    #     black plaque   V median 17   (P25 11, P75 29)
    #     concrete       V median 67   (P5 35, P95 95)
    # but under direct sun the SAME surfaces land at roughly V 77 and V 180. Any constant
    # ceiling that works in one lighting fails in the other - a fixed 45 misses the plaque
    # in sun, a fixed 85 swallows shaded concrete. The ceiling is therefore derived from
    # the frame itself in detect_color (BLACK_V_FRACTION); the value here is only a
    # fallback and the S ceiling stays wide open on purpose.
    #
    # The S ceiling is 255 because saturation is a RATIO and is useless on dark pixels - a
    # black plaque under warm light reads S=146 and would fail any sane S ceiling, which is
    # why it was coming out BELIRSIZ. "Achromatic" is decided by the ABSOLUTE channel spread
    # instead, using the same MIN_CHROMA_DELTA that gates red and green. The two tests are
    # exact complements: a pixel is either coloured enough for red/green or flat enough for
    # black, never both.
    lower_black, upper_black = (0, 0, 0), (179, 255, 85)
else:
    # Indoor set, kept for bench testing. Not tuned.
    lower_red1, upper_red1 = (0, 120, 80), (10, 255, 255)
    lower_red2, upper_red2 = (170, 120, 80), (179, 255, 255)
    lower_green, upper_green = (36, 80, 80), (85, 255, 255)
    lower_black, upper_black = (0, 0, 0), (179, 80, 55)

kernel = np.ones((5, 5), np.uint8)


def clean(mask):
    """Performs morphological operations (opening and closing) to clean noise."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _best_blob(mask, max_area=None, shape_gated=False):
    """
    Largest acceptable contour in `mask`. Returns (area, extent, reason).

    `reason` describes why the LARGEST contour was thrown away, so a BELIRSIZ result can
    say what actually happened instead of leaving you guessing. It is None on success.

    max_area=None means no upper limit - that is the correct setting for red and green,
    where a large blob simply is a large target.

    shape_gated=True applies the two cheap checks that separate a 50x50 cm plaque from a
    shadow. They are applied to the BLACK mask only - red and green already separate from
    concrete by saturation alone, so gating them would add risk for no benefit.

      extent = contour area / bounding-box area
        square plaque ~0.95, drone's own shadow ~0.3 (body + four arms + prop rings)
      border contact
        the plaque is a discrete object in frame; building/pole/operator shadows usually
        run off the edge
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]

    best_area = 0.0
    best_extent = 0.0
    best_cnt = None
    largest = 0.0
    reason = None

    for c in cnts:
        area = cv2.contourArea(c)
        if area > largest:
            largest = area

        if area < MIN_BLOB_AREA_PX:
            if area >= largest:
                reason = f"kucuk ({area:.0f}<{MIN_BLOB_AREA_PX})"
            continue
        if max_area is not None and area > max_area:
            if area >= largest:
                reason = f"buyuk ({area:.0f}>{max_area:.0f})"
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        extent = area / float(bw * bh) if bw * bh > 0 else 0.0

        if shape_gated:
            if extent < BLACK_MIN_EXTENT:
                if area >= largest:
                    reason = f"extent {extent:.2f}<{BLACK_MIN_EXTENT}"
                continue
            if (BLACK_REJECT_BORDER and not CLOSE_RANGE_TEST
                    and area > (w * h) * BLACK_BORDER_MAX_FRAC
                    and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)):
                if area >= largest:
                    reason = "kenara degiyor (buyuk)"
                continue

        if area > best_area:
            best_area = area
            best_extent = extent
            best_cnt = c

    if best_area > 0:
        reason = None
    elif reason is None and largest == 0.0:
        reason = "maske bos"
    return best_area, best_extent, reason, best_cnt


def detect_color(roi):
    """
    Detects the most dominant color (Black, Red, Green or Undefined) in the ROI.

    Returns (label, confidence, area_px, black_extent, diag) where diag maps each colour to
    its rejection reason (or None).
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Scene-relative black ceiling (see BLACK_V_FRACTION) so the same code works in shade
    # and in direct sun without retuning.
    v_ref = float(np.percentile(hsv[:, :, 2], 95))
    black_v = int(min(BLACK_V_MAX, max(BLACK_V_MIN, v_ref * BLACK_V_FRACTION)))
    mask_black = cv2.inRange(hsv, lower_black, (upper_black[0], upper_black[1], black_v))

    # Absolute chroma gate for the two colour classes (see MIN_CHROMA_DELTA). Without it a
    # dark surface with a slight warm cast passes the red hue+saturation test, which is why
    # the black plaque was being reported as KIRMIZI.
    mx = roi.max(axis=2)
    mn = roi.min(axis=2)
    delta = cv2.subtract(mx, mn)
    chroma_ok = cv2.inRange(delta, np.uint8([MIN_CHROMA_DELTA]), np.uint8([255]))
    achromatic = cv2.bitwise_not(chroma_ok)

    mask_red = cv2.bitwise_and(mask_red, chroma_ok)
    mask_green = cv2.bitwise_and(mask_green, chroma_ok)
    # Black is "dark AND flat", not "dark AND low saturation" - see lower_black above.
    mask_black = cv2.bitwise_and(mask_black, achromatic)

    mask_red = clean(mask_red)
    mask_green = clean(mask_green)
    mask_black = clean(mask_black)

    roi_area = roi.shape[0] * roi.shape[1]
    # No upper bound on the chromatic plaques; the black mask keeps one so a large shadow
    # cannot win, unless we are bench testing up close.
    black_max = None if CLOSE_RANGE_TEST else roi_area * BLACK_MAX_AREA_FRAC

    red_area, _, red_why, red_cnt = _best_blob(mask_red)
    green_area, _, green_why, green_cnt = _best_blob(mask_green)
    black_area, black_extent, black_why, black_cnt = _best_blob(
        mask_black, black_max, shape_gated=True)

    diag = {"KIRMIZI": red_why, "YESIL": green_why, "SIYAH": black_why}
    cnts = {"KIRMIZI": red_cnt, "YESIL": green_cnt, "SIYAH": black_cnt}

    areas = {"KIRMIZI": red_area, "YESIL": green_area, "SIYAH": black_area}
    max_area_label = max(areas, key=areas.get)
    max_area_value = areas[max_area_label]

    if max_area_value > 0:
        return (max_area_label, max_area_value / roi_area, max_area_value,
                black_extent, diag, cnts[max_area_label])
    return "BELIRSIZ", 0.0, 0.0, 0.0, diag, None


# --- Logging ---
# Without the plaques in hand the thresholds above are reasoned, not measured. This log is
# how they get corrected after the first real flight: it records what was decided, and the
# scene statistics record what the concrete and its shadows actually look like - which is
# half of the calibration and needs no plaques at all.
log_writer = None
log_file = None
capture_count = 0
if LOG_ENABLED:
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        new_file = not os.path.exists(LOG_CSV)
        log_file = open(LOG_CSV, "a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if new_file:
            log_writer.writerow(["time", "status", "color", "conf", "area_px",
                                 "black_extent", "scene_med_h", "scene_med_s", "scene_med_v"])
    except Exception as e:
        print(f"[LOG] Could not open {LOG_CSV}: {e}")
        log_writer = None


def log_row(status, color, conf, area_px, extent, scene):
    if log_writer is None:
        return
    try:
        log_writer.writerow([datetime.datetime.now().isoformat(timespec="milliseconds"),
                             status, color, f"{conf:.4f}", f"{area_px:.0f}",
                             f"{extent:.2f}", scene[0], scene[1], scene[2]])
        log_file.flush()
    except Exception:
        pass


def save_capture(frame, color):
    global capture_count
    if not LOG_ENABLED or capture_count >= MAX_CAPTURES:
        return
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"{ts}_{color}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        capture_count += 1
    except Exception:
        pass


class AsyncVideoWriter(threading.Thread):
    """
    Background MP4 writer. Encoding on the capture thread would throttle detection, so
    frames go through a bounded queue and are dropped when the disk cannot keep up - a
    dropped frame costs a gap in the review video, a stalled loop costs detections.
    """

    def __init__(self, filename, fps, size, max_queue=120):
        super().__init__(daemon=True)
        self.filename = filename
        self.fps = fps
        self.size = size
        self.q = queue.Queue(maxsize=max_queue)
        self.running = True
        self.dropped = 0
        self.written = 0

    def enqueue(self, frame):
        if not self.running:
            return
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def run(self):
        writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.fps, self.size)
        if not writer.isOpened():
            print(f"[VIDEO] Could not open {self.filename} for writing.")
            self.running = False
            return
        while self.running or not self.q.empty():
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            writer.write(frame)
            self.written += 1
        writer.release()

    def stop(self):
        self.running = False
        self.join(timeout=5.0)
        print(f"[VIDEO] {self.filename}: {self.written} frames written, {self.dropped} dropped.")


def draw_overlay(frame, label, conf, area, extent, chroma, scene, diag, contour, roi_box=None):
    """Burn the decision into the frame so the recording is reviewable on its own."""
    colours = {"KIRMIZI": (0, 0, 255), "YESIL": (0, 255, 0),
               "SIYAH": (255, 255, 255), "BELIRSIZ": (0, 200, 255)}
    col = colours.get(label, (200, 200, 200))

    # The evaluated window. Drawn so you can see what the detector is allowed to look at
    # while aiming - nothing outside this rectangle is considered.
    if roi_box is not None:
        rx0, ry0, rx1, ry1 = roi_box
        cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (255, 255, 0), 1)

    if contour is not None:
        cv2.drawContours(frame, [contour], -1, col, 3)
        x, y, w_, h_ = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w_, y + h_), col, 1)

    lines = [f"{label}  conf={conf:.3f}  area={area:.0f}px"]
    if label == "SIYAH":
        lines.append(f"extent={extent:.2f} (min {BLACK_MIN_EXTENT})")
    if chroma:
        lines.append(f"chroma HSV H={chroma[0]} S={chroma[1]} V={chroma[2]} ({chroma[3]}%)")
    else:
        lines.append("chroma: yok")
    lines.append(f"scene HSV {scene}")
    if label == "BELIRSIZ":
        for c in ("KIRMIZI", "YESIL", "SIYAH"):
            if diag.get(c):
                lines.append(f"  {c}: {diag[c]}")

    y0 = 26
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, t, (12, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    col if i == 0 else (230, 230, 230), 1, cv2.LINE_AA)
    return frame


last_sent_label = None
last_sent_time = 0.0
SEND_MIN_INTERVAL_S = 1.0   # heartbeat rate when the colour is unchanged


def send_serial_message(label):
    """
    Sends a JSON message with the detected color.

    Rate limited: this used to fire on every camera frame (~30 Hz). The GCS forwards
    each one to the boat as a set_target_color command, so a single telemetry link was
    carrying ~60 extra packets a second on top of the boat's own telemetry. Send
    immediately when the colour changes, otherwise only as a 1 Hz heartbeat.
    """
    global last_sent_label, last_sent_time

    if master is None or not master.is_open:
        return

    now = time.time()
    if label == last_sent_label and (now - last_sent_time) < SEND_MIN_INTERVAL_S:
        return

    msg = {"id": 3, "drone_color": label}
    try:
        master.write((json.dumps(msg) + "\n").encode('utf-8'))
        last_sent_label = label
        last_sent_time = now
    except Exception as e:
        print(f"Failed to send serial message: {e}")


# --- Main Loop & GPIO Setup ---
# Jetson GPIO Kurulumu
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)  # Fiziksel (header) pin numaralandırmasını kullan

# Jetson'da yazılımsal pull-down her pinde desteklenmiyor; parametre sessizce yok
# sayılabilir. Bu yüzden HARİCİ 10k pull-down direnç şart. Kablo çıkarsa pin havada
# kalır ve gürültüden rastgele HIGH okur.
try:
    GPIO.setup(TRIGGER_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
except Exception:
    GPIO.setup(TRIGGER_PIN, GPIO.IN)


def read_trigger():
    """Cube AUX1'in mevcut seviyesini okur. True = tetik aktif konumda."""
    return GPIO.input(TRIGGER_PIN) == ACTIVE_LEVEL


stable_level = read_trigger()
candidate_level = stable_level
candidate_count = 0

active_start_time = None
idle_start_time = time.time()
current_status = "Boşta"
last_status = "Boşta"

last_detected_color = "BELIRSIZ"
last_detected_conf = 0.0
last_black_extent = 0.0
last_area_px = 0.0

scene_stats = (0, 0, 0)
chroma_stats = None
diag = {}
last_scene_t = 0.0
last_display_t = 0.0
last_fail_capture = 0.0
last_video_t = 0.0
cam_fail_count = 0
bad_exposure_since = 0.0
recalibrations = 0

video_writer = None
if RECORD_VIDEO and cap is not None:
    _vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or CAM_WIDTH
    _vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or CAM_HEIGHT
    _vname = f"renk_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    video_writer = AsyncVideoWriter(_vname, VIDEO_FPS, (_vw, _vh), VIDEO_MAX_QUEUE)
    video_writer.start()
    print(f"[VIDEO] Recording to {_vname} at {VIDEO_FPS:.0f} fps ({_vw}x{_vh})")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


clear_screen()
print("-------------------------------------")
print(" Live Color Detection and Status Screen ")
print("-------------------------------------")
print(f"RC Trigger Pin: fiziksel pin {TRIGGER_PIN}")

try:
    while True:
        # Tetik seviyesini oku ve yumuşat.
        raw_level = read_trigger()
        if raw_level == candidate_level:
            candidate_count += 1
        else:
            candidate_level = raw_level
            candidate_count = 1
        if candidate_count >= DEBOUNCE_SAMPLES:
            stable_level = candidate_level

        # Check for state changes with a timer
        if stable_level:
            if active_start_time is None:
                active_start_time = time.time()
            if time.time() - active_start_time >= TIMER_THRESHOLD:
                current_status = "AKTİF"
                idle_start_time = None
        else:
            active_start_time = None
            if idle_start_time is None:
                idle_start_time = time.time()
            if time.time() - idle_start_time >= TIMER_THRESHOLD:
                current_status = "Boşta"

        if current_status != last_status:
            last_status = current_status

        # A single dropped USB frame used to `break` out of the loop and end the mission.
        # Retry, and rebuild the capture (which re-applies the exposure/WB lock) if the
        # camera really has gone away.
        ok, frame = (False, None)
        if cap is not None:
            ok, frame = cap.read()
        if not ok:
            cam_fail_count += 1
            if cam_fail_count >= 15:
                print("\n[CAM] Read failing - reopening camera...")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = open_camera()
                cam_fail_count = 0
            time.sleep(0.05)
            continue
        cam_fail_count = 0

        # No downscale: the frame is processed at capture resolution. Resizing to 320x240
        # here was throwing away every pixel the camera setting above buys us.
        h, w = frame.shape[:2]
        x0 = int((1 - ROI_RATIO) / 2 * w)
        x1 = int((1 + ROI_RATIO) / 2 * w)
        y0 = int((1 - ROI_RATIO) / 2 * h)
        y1 = int((1 + ROI_RATIO) / 2 * h)
        roi = frame[y0:y1, x0:x1]

        current_color, current_conf, current_area, current_extent, diag, win_cnt = detect_color(roi)

        # Background survey: median H/S/V of the ROI. With no plaques available to
        # calibrate against, knowing exactly what the concrete and its shadows read is the
        # half of the calibration that can still be done - it tells us what the thresholds
        # must exclude.
        now = time.time()
        if now - last_scene_t >= SCENE_STATS_INTERVAL_S:
            last_scene_t = now
            hsv_small = cv2.cvtColor(cv2.resize(roi, (160, 120)), cv2.COLOR_BGR2HSV)
            flat = hsv_small.reshape(-1, 3)
            scene_stats = tuple(int(v) for v in np.median(flat, axis=0))

            # Dominant chroma: the median HSV of everything that is meaningfully coloured.
            # Concrete is achromatic, so whatever survives this filter IS the plaque - and
            # its H/S/V is the number the thresholds have to accept. This is what turns a
            # "still says BELIRSIZ" report into an actual measurement.
            chroma = flat[flat[:, 1] > CHROMA_MIN_S]
            if chroma.shape[0] >= 20:
                m = np.median(chroma, axis=0).astype(int)
                chroma_stats = (int(m[0]), int(m[1]), int(m[2]),
                                round(100.0 * chroma.shape[0] / flat.shape[0], 1))
            else:
                chroma_stats = None

            # Recalibrate if the picture goes bad mid-run (started indoors, flown out into
            # the sun). Bounded: a camera whose exposure genuinely cannot be driven must
            # not turn this into an endless reopen loop spamming the terminal, which is
            # exactly what the first version did.
            if cap is not None and recalibrations < MAX_RECALIBRATIONS:
                v_med = scene_stats[2]
                if v_med > V_TOO_BRIGHT or v_med < V_TOO_DARK:
                    if bad_exposure_since == 0.0:
                        bad_exposure_since = now
                    elif now - bad_exposure_since >= RECALIBRATE_AFTER_S:
                        recalibrations += 1
                        print(f"\n[CAM] Scene median V={v_med} - recalibrating exposure "
                              f"({recalibrations}/{MAX_RECALIBRATIONS})...")
                        print(f"[CAM] {calibrate_exposure(cap)}")
                        bad_exposure_since = 0.0
                        if recalibrations >= MAX_RECALIBRATIONS:
                            print("[CAM] Giving up on automatic exposure - will keep running "
                                  "with whatever the camera gives.")
                else:
                    bad_exposure_since = 0.0

        colour_changed = current_color != last_detected_color
        last_detected_color = current_color
        # These used to update only when the colour CHANGED, so the confidence on screen
        # was a stale value from whenever the last transition happened.
        last_detected_conf = current_conf
        last_black_extent = current_extent
        last_area_px = current_area

        if colour_changed:
            log_row(current_status, current_color, current_conf, current_area,
                    current_extent, scene_stats)
            save_capture(frame, current_color)
        elif current_color == "BELIRSIZ" and (now - last_fail_capture) >= FAIL_CAPTURE_INTERVAL_S:
            # Keep evidence from a run that never detects anything.
            last_fail_capture = now
            log_row(current_status, current_color, current_conf, current_area,
                    current_extent, scene_stats)
            save_capture(frame, "BELIRSIZ")

        if current_status == "AKTİF":
            send_serial_message(last_detected_color)

        # Video: the overlay is drawn on a copy so the saved captures and the detector
        # itself keep seeing clean pixels.
        if video_writer is not None and (now - last_video_t) >= (1.0 / VIDEO_FPS):
            last_video_t = now
            vis = draw_overlay(frame.copy(), current_color, current_conf, current_area,
                               current_extent, chroma_stats, scene_stats, diag,
                               None if win_cnt is None else win_cnt + np.array([[x0, y0]]),
                               roi_box=(x0, y0, x1, y1))
            video_writer.enqueue(vis)

        # Terminal refresh was redrawing the whole screen every iteration (~100 Hz).
        if now - last_display_t >= DISPLAY_INTERVAL_S:
            last_display_t = now
            print("\033[H\033[J", end="")
            print("-------------------------------------")
            print(" Live Color Detection and Status Screen ")
            print("-------------------------------------")
            print(f"Last Detected Color: {last_detected_color}")
            print(f"Confidence: {last_detected_conf:.3f}   Area: {last_area_px:.0f} px")
            if last_detected_color == "SIYAH":
                print(f"Black extent: {last_black_extent:.2f} (min {BLACK_MIN_EXTENT})")
            # Say WHY nothing matched. A bare BELIRSIZ hides whether the colour was absent,
            # too small, too large, or thrown out by a shape gate.
            if last_detected_color == "BELIRSIZ":
                for _c in ("KIRMIZI", "YESIL", "SIYAH"):
                    print(f"   {_c:<8} red: {diag.get(_c) or '-'}")
                # The area cap and the border test both assume the plaque is a small object
                # in a big frame, which is true at altitude and false when you are holding
                # the drone over it. Say so rather than letting it look like a colour bug.
                _sr = diag.get("SIYAH") or ""
                if not CLOSE_RANGE_TEST and ("buyuk" in _sr or "kenara" in _sr):
                    print("   >> yakin mesafe? CLOSE_RANGE_TEST = True yapin")
            if chroma_stats:
                print(f"RENKLI BOLGE  HSV: H={chroma_stats[0]:3d} S={chroma_stats[1]:3d} "
                      f"V={chroma_stats[2]:3d}  (ROI'nin %{chroma_stats[3]}'i)")
                print(f"   kirmizi ister H 0-{upper_red1[0]} veya {lower_red2[0]}-179, "
                      f"S>={lower_red1[1]}, V>={lower_red1[2]}")
                print(f"   yesil   ister H {lower_green[0]}-{upper_green[0]}, "
                      f"S>={lower_green[1]}, V>={lower_green[2]}")
            else:
                print(f"RENKLI BOLGE  yok (S>{CHROMA_MIN_S} olan piksel < %0.1)")
            print(f"Scene median HSV: {scene_stats}")
            print(f"Frame: {w}x{h}  ROI: {x1-x0}x{y1-y0}"
                  f"{'  [YAKIN TEST MODU]' if CLOSE_RANGE_TEST else ''}")
            print(f"RC Trigger Pin: fiziksel pin {TRIGGER_PIN} <- Cube AUX1")
            print(f"Pin Level: {'HIGH' if GPIO.input(TRIGGER_PIN) else 'LOW'}"
                  f"  (aktif kabul edilen: {'HIGH' if ACTIVE_LEVEL else 'LOW'})")

            if master and master.is_open:
                print(f"Serial Connection: OK ({TELEMETRY_PORT})")
            else:
                print(f"Serial Connection: ERROR")

            print(f"\nRC Trigger Status: {current_status}")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    if video_writer is not None:
        video_writer.stop()
    if cap is not None:
        cap.release()
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    GPIO.cleanup()  # Program kapanırken pinleri güvenli hale getir
