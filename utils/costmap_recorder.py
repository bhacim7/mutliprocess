import numpy as np
import cv2
import time
import os
import atexit
import signal
import math

import config as cfg
import utils.navigasyon as nav


# Raw YOLO/data.yaml class id -> BGR. Matches camera_process's colour_map.
CID_BGR = {
    0: (0, 0, 255),      # red
    1: (0, 255, 255),    # yellow
    2: (80, 80, 80),     # black (lifted off pure black so it is visible)
    3: (0, 165, 255),    # orange
    4: (0, 255, 0),      # green
}
CID_NAME = {0: "red", 1: "yellow", 2: "black", 3: "orange", 4: "green"}
UNKNOWN_BGR = (128, 128, 128)


class CostmapRecorder:
    """
    Post-mission map: where the boat actually went, and where it thought the buoys were.

    Rewritten because the previous version's output was dominated by artefacts of the
    drawing method rather than by the data:

      * Every tracked object was re-drawn into the image on every 1 Hz tick and nothing was
        ever erased. A 60 s run with ~15 live objects meant ~900 filled circles stamped on
        top of each other, so the whole history of the position noise was baked in
        permanently. Measured contribution to a single buoy's blob: ~10 m of the ~14 m.
      * Objects were drawn at radius 3 px which, at 0.5 m/px, is a 3.0 m diameter disc for a
        0.5 m buoy - 6x oversized.
      * The boat track was sampled at 1 Hz. At ~1.8 m/s that is 1.8 m per segment, so the
        ~10 s circle in the 2026-08-11 run rendered as a 10-sided polygon instead of a curve.
      * The canvas was a fixed 500 m square while the course is ~55 m across, so ~90% of the
        saved PNG was empty black.

    The fix is to record data (in metres) and render ONCE at save() time:

      * `self.track`        - boat positions, sampled fast enough to show real manoeuvres
      * `self.objects`      - id -> latest smoothed position (one bright dot per buoy)
      * `self.observations` - every raw sighting, drawn as a faint cloud

    Keeping both layers is deliberate. The faint cloud IS the measurement of how noisy the
    perception chain (GPS + magnetic heading + ZED depth) is - the very thing that makes A*
    flip its avoidance side between replans. A ring showing each object's RMS scatter is
    drawn on top so the uncertainty can be read straight off the map in metres.

    Rendering at the end also removes the incremental canvas-expansion logic entirely, along
    with its origin-shifting bookkeeping.
    """

    def __init__(self, output_path="final_costmap.png", res_m_per_px=None,
                 track_hz=None, object_hz=None):
        self.output_path = output_path
        self.res_m_per_px = float(
            res_m_per_px if res_m_per_px is not None
            else getattr(cfg, 'COSTMAP_REC_RES_M_PER_PX', 0.2))

        track_hz = float(track_hz if track_hz is not None
                         else getattr(cfg, 'COSTMAP_REC_TRACK_HZ', 5.0))
        object_hz = float(object_hz if object_hz is not None
                          else getattr(cfg, 'COSTMAP_REC_OBJECT_HZ', 1.0))
        self.track_interval = 1.0 / max(0.1, track_hz)
        self.object_interval = 1.0 / max(0.1, object_hz)

        # Local ENU frame anchored on the first fix (x = East, y = North, metres).
        self.start_lat = None
        self.start_lon = None

        self.track = []          # [(x_m, y_m)]
        self.objects = {}        # id -> {'x': , 'y': , 'cid': , 'n': }
        self.observations = []   # [(x_m, y_m, cid)]

        self._last_track_t = 0.0
        self._last_object_t = 0.0
        self._registered_exit = False
        self._saved = False

        # Bound memory on a long run. At 5 Hz / 1 Hz these are minutes of headroom.
        self.MAX_TRACK = 20000
        self.MAX_OBSERVATIONS = 60000

    # ------------------------------------------------------------------ capture

    def register_exit_handlers(self):
        if self._registered_exit:
            return
        self._registered_exit = True
        atexit.register(self.save)

        def signal_handler(sig, frame):
            self.save()
            # Restore default handler and re-raise to actually exit
            signal.signal(sig, signal.SIG_DFL)
            os.kill(os.getpid(), sig)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _latlon_to_meter(self, lat, lon):
        """(East, North) metres relative to the first fix seen."""
        if self.start_lat is None or self.start_lon is None:
            self.start_lat = lat
            self.start_lon = lon
            return 0.0, 0.0

        dist = nav.haversine(self.start_lat, self.start_lon, lat, lon)
        bearing_rad = math.radians(
            nav.calculate_bearing(self.start_lat, self.start_lon, lat, lon))
        return dist * math.sin(bearing_rad), dist * math.cos(bearing_rad)

    def update(self, boat_lat, boat_lon, objects_list=None):
        """Called every nav cycle; internally throttled to track_hz / object_hz."""
        if not boat_lat or not boat_lon:
            return

        now = time.time()

        if (now - self._last_track_t) >= self.track_interval:
            self._last_track_t = now
            if len(self.track) < self.MAX_TRACK:
                self.track.append(self._latlon_to_meter(boat_lat, boat_lon))

        if not objects_list or (now - self._last_object_t) < self.object_interval:
            return
        self._last_object_t = now

        for obj in objects_list:
            obj_lat = obj.get('lat')
            obj_lon = obj.get('lon')
            if not obj_lat or not obj_lon:
                continue

            x_m, y_m = self._latlon_to_meter(obj_lat, obj_lon)
            cid = obj.get('cid')

            if len(self.observations) < self.MAX_OBSERVATIONS:
                self.observations.append((x_m, y_m, cid))

            # One registry entry per tracked object; the position is simply the most recent
            # one, which is already alpha-smoothed inside ObjectMemoryManager.
            key = obj.get('id')
            if key is None:
                continue
            entry = self.objects.get(key)
            if entry is None:
                self.objects[key] = {'x': x_m, 'y': y_m, 'cid': cid, 'n': 1}
            else:
                entry['x'] = x_m
                entry['y'] = y_m
                entry['cid'] = cid
                entry['n'] += 1

    # ------------------------------------------------------------------- render

    def _bounds(self, margin_m=5.0):
        xs = [p[0] for p in self.track] + [o['x'] for o in self.objects.values()] \
             + [p[0] for p in self.observations]
        ys = [p[1] for p in self.track] + [o['y'] for o in self.objects.values()] \
             + [p[1] for p in self.observations]
        if not xs or not ys:
            return None
        return (min(xs) - margin_m, min(ys) - margin_m,
                max(xs) + margin_m, max(ys) + margin_m)

    def _scatter_radii(self):
        """RMS distance of each object's observations from its final position, in metres."""
        if not self.objects:
            return {}

        # Group by class id once: a long run can hold tens of thousands of observations and
        # a naive all-pairs scan would stall the shutdown path.
        by_cid = {}
        for k, o in self.objects.items():
            by_cid.setdefault(o['cid'], []).append((k, o['x'], o['y']))

        acc = {k: [0.0, 0] for k in self.objects}
        for x_m, y_m, cid in self.observations:
            # Attribute each observation to the nearest registry entry of the same class.
            best_k, best_d2 = None, None
            for k, ox, oy in by_cid.get(cid, ()):
                d2 = (ox - x_m) ** 2 + (oy - y_m) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2, best_k = d2, k
            if best_k is not None:
                acc[best_k][0] += best_d2
                acc[best_k][1] += 1
        return {k: math.sqrt(s / n) for k, (s, n) in acc.items() if n >= 3}

    def _draw_scale_bar(self, img, res):
        h, w = img.shape[:2]
        for candidate in (50.0, 20.0, 10.0, 5.0, 2.0, 1.0):
            length_px = int(candidate / res)
            if length_px <= w * 0.35:
                break
        else:
            return
        x0, y0 = 12, h - 16
        cv2.line(img, (x0, y0), (x0 + length_px, y0), (200, 200, 200), 2)
        cv2.line(img, (x0, y0 - 4), (x0, y0 + 4), (200, 200, 200), 2)
        cv2.line(img, (x0 + length_px, y0 - 4), (x0 + length_px, y0 + 4), (200, 200, 200), 2)
        cv2.putText(img, f"{candidate:.0f} m", (x0, y0 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def render(self):
        """Build the final image. Returns None if nothing was recorded."""
        bounds = self._bounds()
        if bounds is None:
            return None

        min_x, min_y, max_x, max_y = bounds
        res = self.res_m_per_px
        w = max(64, int((max_x - min_x) / res) + 1)
        h = max(64, int((max_y - min_y) / res) + 1)

        # Guard against a runaway canvas if the GPS ever produced a wild outlier.
        MAX_PX = 4000
        if w > MAX_PX or h > MAX_PX:
            res = max((max_x - min_x) / MAX_PX, (max_y - min_y) / MAX_PX)
            w = max(64, int((max_x - min_x) / res) + 1)
            h = max(64, int((max_y - min_y) / res) + 1)

        img = np.zeros((h, w, 3), dtype=np.uint8)

        def to_px(x_m, y_m):
            # East -> +x, North -> up (-y in image space)
            return (int((x_m - min_x) / res), int((max_y - y_m) / res))

        # 1. Faint cloud of every raw sighting - this is the perception noise, and its
        #    diameter is the number that matters when tuning heading/depth.
        for x_m, y_m, cid in self.observations:
            b, g, r = CID_BGR.get(cid, UNKNOWN_BGR)
            cv2.circle(img, to_px(x_m, y_m), 1, (b // 4, g // 4, r // 4), -1)

        # 2. Boat track.
        if len(self.track) >= 2:
            pts = np.array([to_px(x, y) for x, y in self.track], dtype=np.int32)
            cv2.polylines(img, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        if self.track:
            cv2.circle(img, to_px(*self.track[0]), 4, (0, 255, 0), 1, cv2.LINE_AA)   # start
            cv2.circle(img, to_px(*self.track[-1]), 4, (0, 0, 255), 1, cv2.LINE_AA)  # end

        # 3. One bright dot per buoy at its final position, plus its RMS scatter ring.
        buoy_px = max(1, int(round(getattr(cfg, 'BUOY_RADIUS_M', 0.25) / res)))
        scatter = self._scatter_radii()
        for key, o in self.objects.items():
            colour = CID_BGR.get(o['cid'], UNKNOWN_BGR)
            centre = to_px(o['x'], o['y'])
            rms = scatter.get(key)
            if rms and rms > res:
                cv2.circle(img, centre, int(rms / res),
                           tuple(c // 3 for c in colour), 1, cv2.LINE_AA)
            cv2.circle(img, centre, buoy_px, colour, -1)

        self._draw_scale_bar(img, res)
        return img, res, scatter

    def save(self):
        # atexit and the signal handler can both fire; only write once.
        if self._saved:
            return
        try:
            rendered = self.render()
            if rendered is None:
                print("[COSTMAP] Nothing recorded - no map written.")
                self._saved = True
                return

            img, res, scatter = rendered
            cv2.imwrite(self.output_path, img)
            self._saved = True

            h, w = img.shape[:2]
            print(f"[COSTMAP] Saved {self.output_path}  {w}x{h} px @ {res:.2f} m/px  "
                  f"({w * res:.0f} x {h * res:.0f} m)")
            print(f"[COSTMAP]   track points: {len(self.track)}   "
                  f"objects: {len(self.objects)}   raw sightings: {len(self.observations)}")

            # Per-colour position scatter. This is the diagnostic number: while it stays
            # above the A* clearance (BUOY_RADIUS_M + ROBOT_RADIUS_M + INFLATION_MARGIN_M)
            # the planner's choice of which side to pass a buoy on is essentially noise.
            if scatter:
                by_colour = {}
                for key, rms in scatter.items():
                    name = CID_NAME.get(self.objects[key]['cid'], "unknown")
                    by_colour.setdefault(name, []).append(rms)
                clearance = (getattr(cfg, 'BUOY_RADIUS_M', 0.25)
                             + getattr(cfg, 'ROBOT_RADIUS_M', 0.4)
                             + getattr(cfg, 'INFLATION_MARGIN_M', 0.55))
                print(f"[COSTMAP]   position scatter (RMS), A* clearance = {clearance:.2f} m:")
                for name, vals in sorted(by_colour.items()):
                    mean = sum(vals) / len(vals)
                    flag = "  <-- exceeds clearance" if mean > clearance else ""
                    print(f"[COSTMAP]     {name:<7} n={len(vals):<3} mean {mean:.2f} m  "
                          f"max {max(vals):.2f} m{flag}")
        except Exception as e:
            print(f"[COSTMAP] Error saving costmap: {e}")
