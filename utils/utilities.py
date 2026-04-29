import math
import threading
import queue
import os
import time
import cv2
import numpy as np

def nfloat(x):
    try:
        if x is None:
            return None
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None

def nint(x):
    try:
        if x is None:
            return None
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return int(round(xf))
    except Exception:
        return None

def wrap_to_pi(angle):
    """Açıyı radyan cinsinden [-pi, pi] aralığına sarar."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def angle_diff(a, b):
    """a ve b arasındaki en küçük işaretli farkı (radyan) döndürür."""
    return wrap_to_pi(a - b)

class EmergencyShutdown(Exception):
    """Yer kontrolden veya sistemden acil kapatma isteği geldiğinde fırlatılır."""
    pass

class KalmanFilter:
    """
    Manyetik pusula (heading) verisini filtrelemek için kullanılır.
    Kullanımı: filter = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)
    """
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.x_estimate = None
        self.y_estimate = None
        self.error_covariance = 1.0

    def update(self, angle):
        x_meas = math.cos(math.radians(angle))
        y_meas = math.sin(math.radians(angle))

        if self.x_estimate is None or self.y_estimate is None:
            self.x_estimate, self.y_estimate = x_meas, y_meas
            return angle

        predicted_x = self.x_estimate
        predicted_y = self.y_estimate
        predicted_error_covariance = self.error_covariance + self.process_variance

        kalman_gain = predicted_error_covariance / (predicted_error_covariance + self.measurement_variance)

        self.x_estimate += kalman_gain * (x_meas - predicted_x)
        self.y_estimate += kalman_gain * (y_meas - predicted_y)
        self.error_covariance = (1 - kalman_gain) * predicted_error_covariance

        norm = math.sqrt(self.x_estimate ** 2 + self.y_estimate ** 2)
        self.x_estimate /= norm
        self.y_estimate /= norm

        filtered_angle = math.degrees(math.atan2(self.y_estimate, self.x_estimate))
        return filtered_angle % 360

class AlphaFilter:
    """Basit Low-Pass (Düşük Geçiren) Filtre alternatifi."""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.filtered_x = None
        self.filtered_y = None

    def update(self, new_heading):
        new_x = math.cos(math.radians(new_heading))
        new_y = math.sin(math.radians(new_heading))

        if self.filtered_x is None or self.filtered_y is None:
            self.filtered_x = new_x
            self.filtered_y = new_y
        else:
            self.filtered_x = self.alpha * new_x + (1 - self.alpha) * self.filtered_x
            self.filtered_y = self.alpha * new_y + (1 - self.alpha) * self.filtered_y

        filtered_heading = math.degrees(math.atan2(self.filtered_y, self.filtered_x))
        if filtered_heading < 0:
            filtered_heading += 360
        return filtered_heading