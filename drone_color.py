# -*- coding: utf-8 -*-
# Gerekli kütüphaneleri içe aktar
import cv2
import numpy as np
import time
import datetime
import csv
import serial
import json
import os
import Jetson.GPIO as GPIO

# ==========================================================================
# TUNABLES - hepsi burada, sahada tek yerden ayarlanır
# ==========================================================================

# --- GPIO and Telemetry Settings ---
# BCM pin numaralandırması kullanılıyor. Orin Nano'da 40-pin header'a göre doğru pini seçtiğinden emin ol.
TRIGGER_PIN = 17

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
# Whole frame. The centre crop was throwing away the outer 30% for no good reason - if the
# plaque is anywhere in view we want it. Set below 1.0 only if something at the edge of the
# frame is causing false positives.
ROI_RATIO = 1.0

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

# --- Black plaque shape gate (see detect_color) ---
# Concrete makes red/green trivial to separate by saturation, but it makes BLACK hard:
# a shadow on concrete and the RAL 9005 plaque land at almost the same brightness, because
# concrete/plaque albedo (~0.30 / ~0.045) differ by about the same factor as sun/shade.
# No brightness threshold can split them, so the black mask gets two cheap shape checks.
BLACK_MIN_EXTENT = 0.75      # contour area / bounding-box area. Square ~0.95, drone shadow ~0.3
BLACK_REJECT_BORDER = True   # drop blobs touching the frame edge (large ground shadows)

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
    # RED - RAL 3026 is a luminous, slightly orange red; the swatch computes to H 0-5.
    # The window is 0-15 / 165-179 rather than 0-10 / 170-179 because everything between
    # 11 and 164 is a dead zone: a white-balance shift of a few degrees pushes the plaque
    # into that gap and it matches NO mask at all. Concrete sits at S 20-60, so widening
    # the hue costs nothing here - the saturation floor is what does the rejecting.
    lower_red1, upper_red1 = (0, 110, 60), (15, 255, 255)
    lower_red2, upper_red2 = (165, 110, 60), (179, 255, 255)

    # GREEN - RAL 6037 computes to H ~70-78.5. The old 40-85 window left only 6.5 deg of
    # headroom above the target while wasting 30 deg below it; 55-95 centres it properly.
    lower_green, upper_green = (55, 110, 60), (95, 255, 255)

    # Saturation floor was 150. Concrete is achromatic (S 20-60) so 110 still separates it
    # by a wide margin, while leaving room for a plaque that is partially washed out by
    # direct sun or a slightly wrong white balance.

    # BLACK - RAL 9005. The old V ceiling of 45 was almost certainly rejecting the plaque
    # outright: with exposure set for sunlit concrete the plaque is expected around V 54-76.
    # Raising the ceiling to 85 lets it in - and also lets shadowed concrete (V ~62-87) in,
    # which is why the shape gate in detect_color is not optional.
    lower_black, upper_black = (0, 0, 0), (179, 80, 85)
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
                    and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)):
                if area >= largest:
                    reason = "kenara degiyor"
                continue

        if area > best_area:
            best_area = area
            best_extent = extent

    if best_area > 0:
        reason = None
    elif reason is None and largest == 0.0:
        reason = "maske bos"
    return best_area, best_extent, reason


def detect_color(roi):
    """
    Detects the most dominant color (Black, Red, Green or Undefined) in the ROI.

    Returns (label, confidence, area_px, black_extent, diag) where diag maps each colour to
    its rejection reason (or None).
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_red = clean(mask_red)
    mask_green = clean(mask_green)
    mask_black = clean(mask_black)

    roi_area = roi.shape[0] * roi.shape[1]
    # No upper bound on the chromatic plaques; the black mask keeps one so a large shadow
    # cannot win, unless we are bench testing up close.
    black_max = None if CLOSE_RANGE_TEST else roi_area * BLACK_MAX_AREA_FRAC

    red_area, _, red_why = _best_blob(mask_red)
    green_area, _, green_why = _best_blob(mask_green)
    black_area, black_extent, black_why = _best_blob(mask_black, black_max, shape_gated=True)

    diag = {"KIRMIZI": red_why, "YESIL": green_why, "SIYAH": black_why}

    areas = {"KIRMIZI": red_area, "YESIL": green_area, "SIYAH": black_area}
    max_area_label = max(areas, key=areas.get)
    max_area_value = areas[max_area_label]

    if max_area_value > 0:
        return max_area_label, max_area_value / roi_area, max_area_value, black_extent, diag
    return "BELIRSIZ", 0.0, 0.0, 0.0, diag


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
global pulse_width_us, pulse_start_time, last_pulse_time
pulse_width_us = 0.0
pulse_start_time = 0.0
last_pulse_time = 0.0

# If no RC edge arrives for this long the signal is gone (transmitter off, cable out).
# pulse_width_us only ever updates on a falling edge, so without this it FREEZES at its
# last value - a link loss while active left the drone stuck transmitting forever.
RC_SIGNAL_TIMEOUT_S = 0.5

# Jetson GPIO Kurulumu
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # BCM pin numaralandırmasını kullan
GPIO.setup(TRIGGER_PIN, GPIO.IN)


# Pin olay işleyicisi (Interrupt Callback)
def pin_edge_callback(channel):
    global pulse_start_time, pulse_width_us, last_pulse_time
    now = time.time()
    last_pulse_time = now
    if GPIO.input(channel) == GPIO.HIGH:  # Yükselen kenar (Activated)
        pulse_start_time = now
    else:  # Düşen kenar (Deactivated)
        if pulse_start_time > 0:
            pulse_width_us = (now - pulse_start_time) * 1000000


# Hem yükselen hem düşen kenarda tetiklenecek şekilde ayarla
GPIO.add_event_detect(TRIGGER_PIN, GPIO.BOTH, callback=pin_edge_callback)

ACTIVATION_THRESHOLD = 1500
DEACTIVATION_THRESHOLD = 1500
TIMER_THRESHOLD = 2.0  # 2 seconds
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
cam_fail_count = 0
bad_exposure_since = 0.0
recalibrations = 0


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


clear_screen()
print("-------------------------------------")
print(" Live Color Detection and Status Screen ")
print("-------------------------------------")
print(f"RC Trigger Pin: GPIO {TRIGGER_PIN}")

try:
    while True:
        # RC failsafe: no edges for a while means there is no signal any more, so the
        # last measured pulse width is stale and must not keep the system ACTIVE.
        if last_pulse_time > 0.0 and (time.time() - last_pulse_time) > RC_SIGNAL_TIMEOUT_S:
            pulse_width_us = 0.0
            rc_link_ok = False
        else:
            rc_link_ok = last_pulse_time > 0.0

        # Check for state changes with a timer
        if pulse_width_us > ACTIVATION_THRESHOLD:
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

        current_color, current_conf, current_area, current_extent, diag = detect_color(roi)

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
            print(f"RC Trigger Pin: GPIO {TRIGGER_PIN} (BCM)")
            print(f"Pulse Width (µs): {pulse_width_us:.2f}")
            print(f"RC Signal: {'OK' if rc_link_ok else 'NO SIGNAL'}")

            if master and master.is_open:
                print(f"Serial Connection: OK ({TELEMETRY_PORT})")
            else:
                print(f"Serial Connection: ERROR")

            print(f"\nRC Trigger Status: {current_status}")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
finally:
    if cap is not None:
        cap.release()
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass
    GPIO.cleanup()  # Program kapanırken pinleri güvenli hale getir
