import numpy as np
import cv2
import time
import os
import atexit
import signal
import math
import utils.navigasyon as nav

class CostmapRecorder:
    def __init__(self, output_path="final_costmap.png", initial_size_m=500.0, res_m_per_px=0.5):
        self.output_path = output_path
        self.res_m_per_px = res_m_per_px

        # Initial size in pixels
        self.size_px = int(initial_size_m / self.res_m_per_px)

        # Internal image: BGR format, initialized to black
        self.img = np.zeros((self.size_px, self.size_px, 3), dtype=np.uint8)

        # Origin offset in pixels (where (0,0) meter is on the image)
        # Initially center of the image
        self.origin_x_px = self.size_px // 2
        self.origin_y_px = self.size_px // 2

        self.start_lat = None
        self.start_lon = None

        self.last_update_time = 0.0
        self.update_interval = 1.0 # 1 Hz

        self.prev_boat_x_px = None
        self.prev_boat_y_px = None

        self._registered_exit = False

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

    def _expand_image_if_needed(self, x_px, y_px, margin=50):
        h, w = self.img.shape[:2]

        need_expand = False
        new_w, new_h = w, h
        dx, dy = 0, 0

        if x_px < margin:
            dx = margin - x_px
            new_w += dx
            need_expand = True
        elif x_px > w - margin:
            new_w += (x_px - (w - margin))
            need_expand = True

        if y_px < margin:
            dy = margin - y_px
            new_h += dy
            need_expand = True
        elif y_px > h - margin:
            new_h += (y_px - (h - margin))
            need_expand = True

        if need_expand:
            new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            new_img[dy:dy+h, dx:dx+w] = self.img
            self.img = new_img

            self.origin_x_px += dx
            self.origin_y_px += dy

            if self.prev_boat_x_px is not None:
                self.prev_boat_x_px += dx
                self.prev_boat_y_px += dy

            return dx, dy
        return 0, 0

    def _meter_to_px(self, x_m, y_m):
        # x_m is East, y_m is North. We map North to -y in image (up) and East to +x (right)
        x_px = self.origin_x_px + int(x_m / self.res_m_per_px)
        y_px = self.origin_y_px - int(y_m / self.res_m_per_px)
        return x_px, y_px

    def _latlon_to_meter(self, lat, lon):
        if self.start_lat is None or self.start_lon is None:
            self.start_lat = lat
            self.start_lon = lon

        # Using haversine to get distance and bearing
        dist = nav.haversine(self.start_lat, self.start_lon, lat, lon)
        bearing = nav.calculate_bearing(self.start_lat, self.start_lon, lat, lon)

        bearing_rad = math.radians(bearing)

        # North is bearing 0, East is bearing 90.
        y_m = dist * math.cos(bearing_rad)
        x_m = dist * math.sin(bearing_rad)

        return x_m, y_m

    def update(self, boat_lat, boat_lon, objects_list=None):
        if boat_lat is None or boat_lon is None:
            return

        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time

        # Get boat position in meters relative to start
        x_m, y_m = self._latlon_to_meter(boat_lat, boat_lon)

        boat_x_px, boat_y_px = self._meter_to_px(x_m, y_m)

        self._expand_image_if_needed(boat_x_px, boat_y_px)

        # Recalculate in case of expansion
        boat_x_px, boat_y_px = self._meter_to_px(x_m, y_m)

        # Draw path
        if self.prev_boat_x_px is not None and self.prev_boat_y_px is not None:
            cv2.line(self.img, (self.prev_boat_x_px, self.prev_boat_y_px), (boat_x_px, boat_y_px), (255, 255, 255), 2)

        self.prev_boat_x_px = boat_x_px
        self.prev_boat_y_px = boat_y_px

        # Draw objects
        if objects_list:
            for obj in objects_list:
                # obj is usually a dict with lat, lon, and label
                obj_lat = obj.get('lat')
                obj_lon = obj.get('lon')

                if obj_lat and obj_lon:
                    obj_x_m, obj_y_m = self._latlon_to_meter(obj_lat, obj_lon)
                    obj_x_px, obj_y_px = self._meter_to_px(obj_x_m, obj_y_m)

                    self._expand_image_if_needed(obj_x_px, obj_y_px)
                    obj_x_px, obj_y_px = self._meter_to_px(obj_x_m, obj_y_m)

                    # NOTE: objects coming from camera_process.py's shared_state payload never
                    # had a 'label' field - only 'cid' (raw YOLO/data.yaml class id:
                    # 0=red,1=yellow,2=black,3=orange,4=green). The old label-string lookup
                    # below always fell through to "unknown"/gray for every single object.
                    cid_val = obj.get('cid')
                    label = obj.get('label', '')
                    color = (128, 128, 128)  # Gray for unknown
                    if cid_val == 0 or 'red' in label.lower() or 'kirmizi' in label.lower():
                        color = (0, 0, 255)  # BGR
                    elif cid_val == 1 or 'yellow' in label.lower() or 'sari' in label.lower():
                        color = (0, 255, 255)
                    elif cid_val == 2 or 'black' in label.lower() or 'siyah' in label.lower():
                        color = (50, 50, 50)
                    elif cid_val == 3 or 'orange' in label.lower() or 'turuncu' in label.lower():
                        color = (0, 165, 255)  # BGR
                    elif cid_val == 4 or 'green' in label.lower() or 'yesil' in label.lower():
                        color = (0, 255, 0)

                    cv2.circle(self.img, (obj_x_px, obj_y_px), 3, color, -1)

    def save(self):
        try:
            cv2.imwrite(self.output_path, self.img)
            print(f"[COSTMAP] Costmap saved to {self.output_path}")
        except Exception as e:
            print(f"[COSTMAP] Error saving costmap: {e}")
