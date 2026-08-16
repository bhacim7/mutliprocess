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
        self.objects = {}        # id -> {'x': , 'y': , 'cid': , 'n': , 't0': }
        # Sightings carry the time they were made, in seconds since the first record. That
        # single extra number is what makes the run replayable: without it the list is an
        # unordered pile - the 1 Hz batches are appended back to back with no separator, and
        # batch size varies with how many buoys were in view, so a position in the list says
        # nothing about when. The final PNG never needed it; a video does.
        self.observations = []   # [(x_m, y_m, cid, t_s)]
        self._t0 = None

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
        if self._t0 is None:
            self._t0 = now

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
                self.observations.append((x_m, y_m, cid, now - self._t0))

            # One registry entry per tracked object; the position is simply the most recent
            # one, which is already alpha-smoothed inside ObjectMemoryManager.
            key = obj.get('id')
            if key is None:
                continue
            entry = self.objects.get(key)
            if entry is None:
                # 't0' = when this track was first seen. Used to decide the moment a buoy
                # enters the video; it cannot be recovered afterwards.
                self.objects[key] = {'x': x_m, 'y': y_m, 'cid': cid, 'n': 1,
                                     't0': now - self._t0}
            else:
                entry['x'] = x_m
                entry['y'] = y_m
                entry['cid'] = cid
                entry['n'] += 1

    # ------------------------------------------------------------------- render

    def _clustered_objects(self, merge_m=None):
        """
        Merge tracks that are really the same physical buoy.

        ObjectMemoryManager drops a track after 5 s without a sighting, so a buoy that goes
        in and out of view collects a new id every time. The 2026-08-12 run recorded 292
        object ids for a course with roughly 25-30 buoys - about a dozen tracks each - and
        the map showed every one of them.

        This also fixes what the scatter figure means. Per-track scatter (0.11 m on that
        run) is PRECISION: how steady one track's own readings were. The spread BETWEEN the
        tracks of one buoy is ACCURACY, and that is the number that decides whether a 1.5 m
        gap can be threaded. Clustering exposes it.

        Returns (clusters, stats) where each cluster is
        {'x','y','cid','n_tracks','spread'} and spread is the RMS distance of its member
        tracks from the cluster centre.
        """
        if merge_m is None:
            merge_m = getattr(cfg, 'COSTMAP_REC_MERGE_M', 1.5)

        by_cid = {}
        for o in self.objects.values():
            by_cid.setdefault(o['cid'], []).append(o)

        clusters = []
        for cid, objs in by_cid.items():
            remaining = list(objs)
            while remaining:
                seed = remaining.pop()
                members = [seed]
                changed = True
                while changed:
                    changed = False
                    cx = sum(m['x'] for m in members) / len(members)
                    cy = sum(m['y'] for m in members) / len(members)
                    keep = []
                    for r in remaining:
                        if math.hypot(r['x'] - cx, r['y'] - cy) <= merge_m:
                            members.append(r)
                            changed = True
                        else:
                            keep.append(r)
                    remaining = keep
                cx = sum(m['x'] for m in members) / len(members)
                cy = sum(m['y'] for m in members) / len(members)
                if len(members) > 1:
                    spread = math.sqrt(sum((m['x'] - cx) ** 2 + (m['y'] - cy) ** 2
                                           for m in members) / len(members))
                else:
                    spread = 0.0
                # When this buoy enters the video. Tracks with fewer than 3 sightings are
                # ignored for this: one spurious detection close enough to be merged into a
                # real buoy would otherwise drag its appearance minutes early.
                solid = [m['t0'] for m in members if m.get('n', 0) >= 3 and 't0' in m]
                all_t0 = [m['t0'] for m in members if 't0' in m]
                t_first = min(solid) if solid else (min(all_t0) if all_t0 else 0.0)
                clusters.append({'x': cx, 'y': cy, 'cid': cid, 't_first': t_first,
                                 'n_tracks': len(members), 'spread': spread})
        return clusters

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
        for x_m, y_m, cid, _t in self.observations:
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

    def _canvas(self, res_m_per_px=None):
        """
        Canvas geometry for the whole run: (w, h, res, min_x, max_y) or None.

        Shared by the PNG and the video so both land on exactly the same frame - a buoy
        sits at the same spot in each, which matters when they are shown side by side.
        """
        bounds = self._bounds()
        if bounds is None:
            return None

        min_x, min_y, max_x, max_y = bounds
        res = float(res_m_per_px or self.res_m_per_px)
        w = max(64, int((max_x - min_x) / res) + 1)
        h = max(64, int((max_y - min_y) / res) + 1)

        # Guard against a runaway canvas if the GPS ever produced a wild outlier.
        MAX_PX = 4000
        if w > MAX_PX or h > MAX_PX:
            res = max((max_x - min_x) / MAX_PX, (max_y - min_y) / MAX_PX)
            w = max(64, int((max_x - min_x) / res) + 1)
            h = max(64, int((max_y - min_y) / res) + 1)

        return w, h, res, min_x, max_y

    def render(self):
        """Build the final image. Returns None if nothing was recorded."""
        canvas = self._canvas()
        if canvas is None:
            return None
        w, h, res, min_x, max_y = canvas

        img = np.zeros((h, w, 3), dtype=np.uint8)

        def to_px(x_m, y_m):
            # East -> +x, North -> up (-y in image space)
            return (int((x_m - min_x) / res), int((max_y - y_m) / res))

        # 1. Faint cloud of every raw sighting - this is the perception noise, and its
        #    diameter is the number that matters when tuning heading/depth.
        for x_m, y_m, cid, _t in self.observations:
            b, g, r = CID_BGR.get(cid, UNKNOWN_BGR)
            cv2.circle(img, to_px(x_m, y_m), 1, (b // 4, g // 4, r // 4), -1)

        # 2. Boat track.
        if len(self.track) >= 2:
            pts = np.array([to_px(x, y) for x, y in self.track], dtype=np.int32)
            cv2.polylines(img, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
        if self.track:
            cv2.circle(img, to_px(*self.track[0]), 4, (0, 255, 0), 1, cv2.LINE_AA)   # start
            cv2.circle(img, to_px(*self.track[-1]), 4, (0, 0, 255), 1, cv2.LINE_AA)  # end

        # 3. One bright dot per PHYSICAL buoy (tracks clustered), with a ring showing how
        #    far its individual tracks disagreed - that spread is the position accuracy.
        buoy_px = max(1, int(round(getattr(cfg, 'BUOY_RADIUS_M', 0.25) / res)))
        scatter = self._scatter_radii()
        clusters = self._clustered_objects()
        for c in clusters:
            colour = CID_BGR.get(c['cid'], UNKNOWN_BGR)
            centre = to_px(c['x'], c['y'])
            if c['spread'] > res:
                cv2.circle(img, centre, int(c['spread'] / res),
                           tuple(v // 3 for v in colour), 1, cv2.LINE_AA)
            cv2.circle(img, centre, buoy_px, colour, -1)

        self._draw_scale_bar(img, res)
        return img, res, scatter, clusters

    # -------------------------------------------------------------- replay data

    def data_path(self):
        base = self.output_path.rsplit('.', 1)[0]
        return base + ".npz"

    def video_path(self):
        base = self.output_path.rsplit('.', 1)[0]
        return base + ".mp4"

    def save_data(self):
        """
        Write the raw recording next to the PNG. Milliseconds, a few hundred KB.

        The PNG is one frame - the state at the end - so a video cannot be made from it;
        the chronology is simply not in there. This file is what costmap_video.py replays,
        and writing it ALWAYS (even when the video is not generated here) means a run can
        be turned into a video later, on a laptop, with different speed or scale, without
        going back on the water.
        """
        try:
            objs = list(self.objects.values())
            np.savez_compressed(
                self.data_path(),
                # float64 throughout. float32 has ample precision for metres, but the
                # canvas origin comes from min()/max() over these values, so a rounding
                # difference of 1e-6 m can shift the whole grid and move dots by a pixel:
                # measured 0.47% of pixels differing on a re-render. The offline render is
                # supposed to reproduce the boat's PNG exactly, and the extra bytes are
                # not worth arguing about.
                track=np.array(self.track, dtype=np.float64).reshape(-1, 2),
                obs=np.array(self.observations, dtype=np.float64).reshape(-1, 4),
                obj_x=np.array([o['x'] for o in objs], dtype=np.float64),
                obj_y=np.array([o['y'] for o in objs], dtype=np.float64),
                obj_cid=np.array([o['cid'] if o['cid'] is not None else -1 for o in objs],
                                 dtype=np.int16),
                obj_n=np.array([o['n'] for o in objs], dtype=np.int32),
                obj_t0=np.array([o.get('t0', 0.0) for o in objs], dtype=np.float64),
                meta=np.array([self.res_m_per_px, self.track_interval,
                               self.start_lat or 0.0, self.start_lon or 0.0],
                              dtype=np.float64),
            )
            return self.data_path()
        except Exception as e:
            print(f"[COSTMAP] Could not write replay data: {e}")
            return None

    @classmethod
    def from_npz(cls, path):
        """Rebuild a recorder from a saved run, for offline rendering."""
        z = np.load(path)
        meta = z['meta']
        rec = cls(output_path=str(path).rsplit('.', 1)[0] + ".png",
                  res_m_per_px=float(meta[0]))
        rec.track_interval = float(meta[1])
        rec.start_lat, rec.start_lon = float(meta[2]), float(meta[3])
        rec.track = [(float(a), float(b)) for a, b in z['track']]
        rec.observations = [(float(a), float(b), int(c), float(t)) for a, b, c, t in z['obs']]
        rec.objects = {
            i: {'x': float(x), 'y': float(y), 'cid': (int(c) if int(c) >= 0 else None),
                'n': int(n), 't0': float(t0)}
            for i, (x, y, c, n, t0) in enumerate(zip(
                z['obj_x'], z['obj_y'], z['obj_cid'], z['obj_n'], z['obj_t0']))
        }
        return rec

    def render_video(self, path=None, sample_hz=None, fps=None, scale=None, progress=True):
        """
        Replay the run as an mp4: the map filling in, in the order it was discovered.

        Two deliberate choices, both about keeping this fast enough to be usable:

        * Buoys are drawn at their FINAL clustered position from the moment they are first
          seen, rather than at whatever the estimate was at that instant. Re-clustering
          every frame would be honest but it makes dots jitter, and two tracks merging
          would make a dot vanish mid-video - which reads as a bug to anyone watching.
          The convergence of the estimate is not lost: the PNG's faint sighting cloud and
          scatter rings show exactly that. The PNG carries uncertainty, the video carries
          chronology.
        * The scatter rings and _scatter_radii() are skipped. That routine compares every
          sighting against every same-class track - roughly half a million distance
          computations on a ten minute run, about a second - and doing it per frame would
          turn a 30 s render into ten minutes for a number nobody reads off a video.

        Sampling rate and playback rate are different things: sampling at 1 Hz and playing
        at 1 fps would make a ten minute run a ten minute video. Default is 1 Hz sampled,
        15 fps played, i.e. 15x, with the elapsed time burned into each frame so the real
        timing stays readable.
        """
        path = path or self.video_path()
        sample_hz = float(sample_hz or getattr(cfg, 'COSTMAP_REC_VIDEO_SAMPLE_HZ', 1.0))
        fps = float(fps or getattr(cfg, 'COSTMAP_REC_VIDEO_FPS', 15.0))
        scale = float(scale or getattr(cfg, 'COSTMAP_REC_VIDEO_SCALE', 2.0))

        canvas = self._canvas()
        if canvas is None:
            return None
        _, _, res_png, _, _ = canvas

        # Same geometry as the PNG, finer pixels: a 55 m course at 0.2 m/px is only ~275 px
        # across, which looks soft on a projector.
        canvas = self._canvas(res_m_per_px=res_png / scale)
        w, h, res, min_x, max_y = canvas
        # mp4v silently produces an unreadable file on odd dimensions.
        w += w & 1
        h += h & 1

        # Run length: the track is sampled on a fixed interval, so its index IS its time.
        t_end = max([o[3] for o in self.observations] or [0.0])
        t_end = max(t_end, (len(self.track) - 1) * self.track_interval)
        if t_end <= 0:
            return None

        # A very long run must not turn into thousands of frames.
        max_frames = int(getattr(cfg, 'COSTMAP_REC_VIDEO_MAX_FRAMES', 900))
        n_frames = int(t_end * sample_hz) + 1
        if n_frames > max_frames:
            sample_hz = max_frames / t_end
            n_frames = max_frames

        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not writer.isOpened():
            print(f"[COSTMAP] VideoWriter refused to open {path} - no mp4 support in this "
                  f"OpenCV build. The .npz is saved; render it elsewhere.")
            return None

        def to_px(x_m, y_m):
            return (int((x_m - min_x) / res), int((max_y - y_m) / res))

        clusters = self._clustered_objects()          # ONCE, not per frame
        clusters.sort(key=lambda c: c['t_first'])
        buoy_px = max(1, int(round(getattr(cfg, 'BUOY_RADIUS_M', 0.25) / res)))
        obs_sorted = sorted(self.observations, key=lambda o: o[3])

        # The cloud and the track only ever grow, so they are painted into a base canvas
        # once each and never redrawn. Only the buoys and the labels are per frame.
        base = np.zeros((h, w, 3), dtype=np.uint8)
        obs_i = 0
        track_i = 0
        n_clusters = 0

        try:
            for f in range(n_frames):
                t = (f / sample_hz) if sample_hz > 0 else t_end

                while obs_i < len(obs_sorted) and obs_sorted[obs_i][3] <= t:
                    x_m, y_m, cid, _ = obs_sorted[obs_i]
                    b, g, r = CID_BGR.get(cid, UNKNOWN_BGR)
                    cv2.circle(base, to_px(x_m, y_m), 1, (b // 4, g // 4, r // 4), -1)
                    obs_i += 1

                want_track = min(len(self.track), int(t / self.track_interval) + 1)
                while track_i < want_track - 1:
                    cv2.line(base, to_px(*self.track[track_i]),
                             to_px(*self.track[track_i + 1]), (255, 255, 255), 1, cv2.LINE_AA)
                    track_i += 1

                img = base.copy()

                while n_clusters < len(clusters) and clusters[n_clusters]['t_first'] <= t:
                    n_clusters += 1
                for c in clusters[:n_clusters]:
                    cv2.circle(img, to_px(c['x'], c['y']), buoy_px,
                               CID_BGR.get(c['cid'], UNKNOWN_BGR), -1)

                if self.track:
                    cv2.circle(img, to_px(*self.track[0]), 4, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(img, to_px(*self.track[min(track_i, len(self.track) - 1)]),
                               5, (0, 0, 255), 2, cv2.LINE_AA)

                self._draw_scale_bar(img, res)
                cv2.putText(img, f"t={int(t) // 60:02d}:{int(t) % 60:02d}   "
                                 f"samandira {n_clusters}/{len(clusters)}",
                            (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (220, 220, 220), 1, cv2.LINE_AA)
                writer.write(img)

                if progress and n_frames > 60 and f % 100 == 0 and f:
                    print(f"[COSTMAP]   video {f}/{n_frames}...", flush=True)
        finally:
            writer.release()   # writes the mp4 index; without it the file will not open

        print(f"[COSTMAP] Saved {path}  {w}x{h}  {n_frames} frames @ {fps:.0f} fps "
              f"({n_frames / fps:.0f} s, {t_end:.0f} s of run)")
        return path

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

            img, res, scatter, clusters = rendered
            cv2.imwrite(self.output_path, img)

            # Both of these are milliseconds. The flag is set straight after them and
            # BEFORE the video, and the ordering is not cosmetic: save() runs inside the
            # SIGINT handler, which does not restore the default handler until it returns
            # (see register_exit_handlers). A second Ctrl+C during a 30 s video render
            # would therefore re-enter save() and, with the flag still unset, start the
            # whole render again - making the program effectively unkillable by Ctrl+C.
            # With the flag set here, that second Ctrl+C returns immediately: the video is
            # abandoned, the PNG and the replay data are already on disk, and the process
            # exits cleanly.
            data_file = self.save_data()
            self._saved = True

            h, w = img.shape[:2]
            print(f"[COSTMAP] Saved {self.output_path}  {w}x{h} px @ {res:.2f} m/px  "
                  f"({w * res:.0f} x {h * res:.0f} m)")
            print(f"[COSTMAP]   track points: {len(self.track)}   "
                  f"tracks: {len(self.objects)}   buoys (clustered): {len(clusters)}   "
                  f"raw sightings: {len(self.observations)}")

            # Between-track spread per physical buoy. This is the ACCURACY figure - the one
            # that decides whether a narrow gap can be threaded. The per-track scatter below
            # is only precision and will always look better than this.
            multi = [c for c in clusters if c['n_tracks'] > 1]
            if multi:
                by_name = {}
                for c in multi:
                    by_name.setdefault(CID_NAME.get(c['cid'], "unknown"), []).append(c['spread'])
                print("[COSTMAP]   between-track spread per buoy (ACCURACY):")
                for name, vals in sorted(by_name.items()):
                    print(f"[COSTMAP]     {name:<7} n={len(vals):<3} "
                          f"mean {sum(vals)/len(vals):.2f} m  max {max(vals):.2f} m")

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

            if data_file:
                print(f"[COSTMAP]   replay data: {data_file}   "
                      f"(video: python utils/costmap_video.py {data_file})")

            # Last, and only on request. Everything above is already safely on disk.
            if getattr(cfg, 'COSTMAP_REC_VIDEO', False):
                print("[COSTMAP] Rendering video, this takes a while - Ctrl+C skips it...")
                self.render_video()
        except Exception as e:
            print(f"[COSTMAP] Error saving costmap: {e}")
