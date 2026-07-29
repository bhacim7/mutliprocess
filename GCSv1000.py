# -*- coding: utf-8 -*-
"""
Gereklilikler:
  pip install PySide6 pyserial
(Projeye internet gerekmez. pyproj yok; WebMercator dönüşümü için formüller içeride.)

PATCH: Arayüzde bütünlüğü sağlamak için bu çok yerinde bir karar. Tüm kutuların (Telemetri, Görev, Kontrol, Log) tıpatıp aynı "Modern HUD" stiline sahip olması için önce küçük bir Yardımcı Fonksiyon yazacağız.

"""
from __future__ import annotations
import sys, os, json, csv, time, math
from typing import Optional, Dict, Any

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt, Signal, QThread

import serial
import serial.tools.list_ports
from PySide6.QtGui import QImageReader
from collections import OrderedDict
from datetime import datetime
import numpy as np
import socket
import struct
import config as cfg

def res_path(name: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(__file__))  # onefile için _MEIPASS
    return os.path.join(base, name)

# ---- Harita resmi ve köşe koordinatları (WGS84) ----
MAP_IMAGE_PATH = res_path("roboboat.jpg")
# Sol-Üst (NW) ve Sağ-Alt (SE) köşeler (WGS84 derece)
NORTH = 27.381
SOUTH = 27.356
WEST  = -82.455
EAST  = -82.445
# ---- Marker boyutları ----
WP_RADIUS_PX = 1.5        # Waypoint dairesi yarıçapı (px)
WP_FONT_PT   = 1.4       # Waypoint etiket yazı boyutu (pt)
CURR_RADIUS_PX = 1.0      # Mevcut konum noktası yarıçapı (px)
PATH_WIDTH_PX  = 3      # İz çizgisi kalınlığı (px)
KEEP_PIXEL_SIZE = False   # Zoom yapsan da ikonlar ekranda aynı piksel boyutunda kalsın mı?

KIRMIZI_CIZGI_KALINLIK = 0.7
GPS_UPDATE_INTERVAL_MS = 5000  # Yalnızca GPS1..GPS5 5 sn'de bir uygulanır


def haversine_distance(lat1, lon1, lat2, lon2):
    """İki koordinat arasındaki mesafeyi (metre) döndürür."""
    R = 6371000  # Dünya yarıçapı (metre)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """İki koordinat arasındaki açıyı (derece, 0=Kuzey) döndürür."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.cos(math.radians(lon2 - lon1))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def is_point_in_polygon(lat, lon, polygon_points):
    """
    Ray-Casting algoritması ile nokta çokgenin içinde mi kontrol eder.
    polygon_points: [(lat1, lon1), (lat2, lon2), ...] listesi
    """
    inside = False
    j = len(polygon_points) - 1
    for i in range(len(polygon_points)):
        xi, yi = polygon_points[i]
        xj, yj = polygon_points[j]

        intersect = ((yi > lon) != (yj > lon)) and \
                    (lat < (xj - xi) * (lon - yi) / (yj - yi + 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

# --- Halat kenar şeridi (döşeme çizer) ---
class RopeStripe(QtWidgets.QWidget):
    def __init__(self, side: str, pix_path: str, thickness: int, parent=None):
        """
        side: "top" | "bottom" | "left" | "right"
        pix_path: halat.png yolu (2560x51)
        thickness: şerit kalınlığı (px)
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)  # tıklamaları engellemesin
        self._side = side
        self._thk = int(thickness)

        pm = QtGui.QPixmap(pix_path)
        if pm.isNull():
            # fallback: gri
            pm = QtGui.QPixmap(512, 51)
            pm.fill(QtGui.QColor("#777"))
        # Dikey kenarlar için 90° döndür
        if side in ("left", "right"):
            pm = pm.transformed(QtGui.QTransform().rotate(90), QtCore.Qt.SmoothTransformation)
        self._pm = pm

        # Boyutu sabitle (geometriyi yine de dışarıdan set edeceğiz)
        if side in ("top", "bottom"):
            self.setFixedHeight(self._thk)
        else:
            self.setFixedWidth(self._thk)

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        # Tüm rect’i halat dokusuyla döşe
        p.drawTiledPixmap(self.rect(), self._pm)

class HugeImageItem(QtWidgets.QGraphicsItem):
    """
    Tam çözünürlükte dev JPEG/PNG'leri karo (tile) halinde tembel yükleyip çizer.
    QImageReader.setClipRect ile sadece gereken bölgeyi diskten decode eder.
    Basit LRU cache ile ekranda görünen karolar tutulur.
    """
    def __init__(self, image_path: str, tile_size: int = 2048, cache_limit: int = 128, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.tile_size = int(tile_size)
        self.cache_limit = int(cache_limit)  # aynı anda tutulacak en fazla karo sayısı
        self._cache: OrderedDict[tuple[int,int], QtGui.QPixmap] = OrderedDict()

        r = QImageReader(self.image_path)
        r.setAutoTransform(True)
        sz = r.size()
        if not sz.isValid():
            # Boyut metadata’sı yoksa bir kere okuyup boyut al
            img = r.read()
            if img.isNull():
                raise RuntimeError(f"Image cannot be read: {self.image_path}")
            self._W, self._H = img.width(), img.height()
        else:
            self._W, self._H = sz.width(), sz.height()

        self.setFlag(QtWidgets.QGraphicsItem.ItemUsesExtendedStyleOption, True)

    # genel boyut
    @property
    def width(self):  return self._W
    @property
    def height(self): return self._H

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(0, 0, self._W, self._H)

    def _load_tile(self, tx: int, ty: int) -> QtGui.QPixmap:
        key = (tx, ty)
        if key in self._cache:
            # LRU güncelle
            pm = self._cache.pop(key)
            self._cache[key] = pm
            return pm

        x = tx * self.tile_size
        y = ty * self.tile_size
        w = min(self.tile_size, self._W - x)
        h = min(self.tile_size, self._H - y)

        r = QImageReader(self.image_path)
        r.setAutoTransform(True)
        r.setClipRect(QtCore.QRect(x, y, w, h))
        img = r.read()
        if img.isNull():
            # başarısızsa gri placeholder
            pm = QtGui.QPixmap(w, h)
            pm.fill(QtGui.QColor("#777"))
        else:
            pm = QtGui.QPixmap.fromImage(img)

        # LRU: kapasiteyi koru
        self._cache[key] = pm
        if len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)  # en eskiyi at
        return pm

    def paint(self, painter: QtGui.QPainter, option: QtWidgets.QStyleOptionGraphicsItem, widget=None):
        # Görünen dikdörtgen (exposed rect) → hangi karolar gerekli?
        rect = option.exposedRect.intersected(self.boundingRect())
        if rect.isEmpty():
            return

        ts = self.tile_size
        x0 = int(rect.left() // ts)
        y0 = int(rect.top()  // ts)
        x1 = int((rect.right()) // ts)
        y1 = int((rect.bottom()) // ts)

        # bir miktar tampon karoyu da getir (daha akıcı pan/zoom)
        pad = 1
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min((self._W - 1) // ts, x1 + pad)
        y1 = min((self._H - 1) // ts, y1 + pad)

        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                pm = self._load_tile(tx, ty)
                painter.drawPixmap(tx * ts, ty * ts, pm)

def normalize_gps_dict(gps_raw):
    """
    Kabul edilen örnek biçimler:
      {"GPS1":{"lat":..,"lon":..}, ...}
      {"GPS1_enlem":..,"GPS1_boylam":.., ...}
      {"GPS1":[lat,lon], ...}  veya {"GPS1":"lat,lon"}
      {"points":[{"name":"GPS1","lat":..,"lon":..}, ...]}
    Dönüş: {"GPS1":{"lat":float,"lon":float}, ...}
    """
    if not isinstance(gps_raw, dict):
        return None

    out = {}
    # 1) Zaten nested dict ise
    ok_nested = True
    for i in range(1, 6):
        k = f"GPS{i}"
        v = gps_raw.get(k)
        if not (isinstance(v, dict) and "lat" in v and "lon" in v):
            ok_nested = False
            break
    if ok_nested:
        # hepsini float’a çevir
        for i in range(1, 6):
            k = f"GPS{i}"
            v = gps_raw.get(k)
            if v is None:
                continue
            try:
                out[k] = {"lat": float(v["lat"]), "lon": float(v["lon"])}
            except Exception:
                pass
        return out if out else None

    # 2) GPS1_enlem / GPS1_boylam
    any_found = False
    for i in range(1, 6):
        ke = f"GPS{i}_enlem"
        kb = f"GPS{i}_boylam"
        if ke in gps_raw and kb in gps_raw:
            try:
                out[f"GPS{i}"] = {"lat": float(gps_raw[ke]), "lon": float(gps_raw[kb])}
                any_found = True
            except Exception:
                pass
    if any_found:
        return out

    # 3) GPS1: [lat,lon]  ya da "lat,lon"
    any_found = False
    for i in range(1, 6):
        k = f"GPS{i}"
        if k in gps_raw:
            v = gps_raw[k]
            try:
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    lat, lon = float(v[0]), float(v[1])
                elif isinstance(v, str) and "," in v:
                    a, b = v.split(",", 1)
                    lat, lon = float(a), float(b)
                else:
                    continue
                out[k] = {"lat": lat, "lon": lon}
                any_found = True
            except Exception:
                pass
    if any_found:
        return out

    # 4) {"points":[{"name":"GPS1","lat":..,"lon":..}, ...]}
    pts = gps_raw.get("points")
    if isinstance(pts, list):
        for p in pts:
            try:
                name = str(p.get("name"))
                if name.upper().startswith("GPS"):
                    out[name.upper()] = {"lat": float(p["lat"]), "lon": float(p["lon"])}
            except Exception:
                pass
        if out:
            return out

    return None


# -------------------- Serial Worker (Düzeltilmiş & Hızlandırılmış) --------------------
class SerialWorker(QThread):
    packet = Signal(dict)
    status = Signal(str)
    link = Signal(bool)

    def __init__(self, port: str, baud: int, csv_path: str | None = None):
        super().__init__()
        self.port = port
        self.baud = baud
        self.csv_path = csv_path
        self._stop = False
        self.ser: Optional[serial.Serial] = None
        self._tx_queue: list[str] = []
        self._csv_file = None
        self._csv_writer = None

        self._rx_buffer = ""  # <--- YENİ: Yarım kalan verileri tutacak havuz

    def configure(self, port: str, baud: int, csv_path: str | None):
        self.port = port
        self.baud = baud
        self.csv_path = csv_path

    def queue_send(self, obj: Dict[str, Any]):
        try:
            line = json.dumps(obj, ensure_ascii=False) + "\r\n"
            self._tx_queue.append(line)
        except Exception as e:
            self.status.emit(f"Serialize error: {e}")

    def run(self):
        # CSV open (optional)
        if self.csv_path:
            try:
                self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv_file)
                if self._csv_file.tell() == 0:
                    self._csv_writer.writerow([
                        "t_ms", "FPS", "SOL_PWM", "SAG_PWM", "HIZ_mps", "HDG_deg", "HDG_HEDEF_deg",
                        "HDG_SAGLIGI", "SONRAKI_NOKTA", "LAT", "LON", "KALAN_m", "MANUEL"
                    ])
            except Exception as e:
                self.status.emit(f"CSV open failed: {e}")
                self._csv_file = None
                self._csv_writer = None

        while not self._stop:
            # 1. Port Açık mı?
            if self.ser is None or not self.ser.is_open:
                self._open_serial()
                if self._stop: break

            try:
                # 2. Gönderilecek Paket Var mı? (TX)
                if self._tx_queue and self.ser and self.ser.is_open:
                    while self._tx_queue:
                        line = self._tx_queue.pop(0)
                        self.ser.write(line.encode("utf-8"))
                    self.ser.flush()  # Tamponu zorla boşalt (Anında gitmesi için)

                # 3. Okunacak Veri Var mı? (RX) - %100 BLOKLAMADAN OKU
                if self.ser and self.ser.is_open:
                    if self.ser.in_waiting > 0:
                        try:
                            # Tamponda NE KADAR veri varsa sadece onu oku, \n bekleme!
                            raw_data = self.ser.read(self.ser.in_waiting).decode("utf-8", errors="ignore")

                            # Gelen veriyi havuza ekle
                            self._rx_buffer += raw_data

                            # Havuzda tam bir satır (\n) oluştu mu diye bak
                            if "\n" in self._rx_buffer:
                                # Satırları \n'e göre böl
                                lines = self._rx_buffer.split("\n")

                                # En son eleman yarım kalmış olabilir (örn: {"id":1, "sp),
                                # onu havuzda bırak, tam satırları al
                                self._rx_buffer = lines.pop()

                                # Tamamlanmış satırları döngüye sok
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        try:
                                            obj = json.loads(line)
                                            if isinstance(obj, dict):
                                                self.packet.emit(obj)  # UI'a gönder

                                                # CSV Kayıt (Opsiyonel)
                                                if self._csv_writer:
                                                    pass
                                        except json.JSONDecodeError:
                                            pass  # Paketin içi bozuksa atla

                        except Exception as e:
                            print(f"Read Error: {e}")

                # CPU'yu rahatlat (Döngü çok hızlı dönmesin, 10ms bekle)
                time.sleep(0.01)

            except (serial.SerialException, OSError):
                if self._stop: break
                self.status.emit(f"Bağlantı koptu, tekrar deneniyor...")
                self.link.emit(False)
                self._close_serial()
                time.sleep(1.0)
            except Exception as e:
                self.status.emit(f"Thread Error: {e}")
                time.sleep(0.1)

        # Temizlik
        self._close_serial()
        if self._csv_file:
            try:
                self._csv_file.close()
            except:
                pass

    def stop(self):
        self._stop = True
        if self.ser and self.ser.is_open:
            try:
                self.ser.cancel_read()
            except:
                pass
            try:
                self.ser.close()
            except:
                pass

    def _open_serial(self):
        while not self._stop:
            try:
                avail = [p.device for p in serial.tools.list_ports.comports()]
                if self.port in avail:
                    # Timeout değerini 0.1 yaptık (Daha hızlı pes etsin)
                    self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
                    self.status.emit(f"Connected {self.port} @ {self.baud}")
                    self.link.emit(True)
                    return
                else:
                    self.status.emit(f"Waiting for {self.port}...")
                    time.sleep(1.0)
            except Exception:
                time.sleep(1.0)

    def _close_serial(self):
        try:
            if self.ser: self.ser.close()
        except:
            pass
        self.ser = None


import socket
import struct
import ntplib
import time
from PySide6.QtCore import QThread, Signal

# ------------- Fare ile Zoom ve Manuel Sürükleme ---------------
class GraphicsView(QtWidgets.QGraphicsView):
    zoomChanged = QtCore.Signal(float)
    clicked = Signal(QtCore.QPointF, QtCore.QPoint)
    moved = Signal(QtCore.QPointF)

    def __init__(self, scene, parent=None, min_zoom=0.25, max_zoom=16.0):
        super().__init__(scene, parent)
        self._min_zoom = float(min_zoom)
        self._max_zoom = float(max_zoom)

        # --- GÖRÜNÜM AYARLARI ---
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)

        # Kaydırma çubuklarını gizle (Daha temiz görüntü)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # --- İMLEÇ AYARI ---
        # Fareyi GİZLE (Sadece senin kırmızı artın görünsün)
        self.setCursor(Qt.BlankCursor)
        # Eğer gizlemek yerine ince artı istersen üsttekini sil, alttakini aç:
        # self.setCursor(Qt.CrossCursor)

        # Sürükleme değişkenleri
        self._is_panning = False
        self._pan_start = QtCore.QPoint()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.pixelDelta().y() if not event.angleDelta().y() else event.angleDelta().y()
        if delta == 0:
            event.ignore();
            return
        step_factor = 1.15 if delta > 0 else (1.0 / 1.15)
        cur = self.transform().m11()
        new_zoom = max(self._min_zoom, min(self._max_zoom, cur * step_factor))
        if new_zoom == cur: return
        factor = new_zoom / cur
        self.scale(factor, factor)
        smooth = new_zoom >= 1.0
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, smooth)
        self.zoomChanged.emit(new_zoom)
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        # Sol tık basılınca sürüklemeyi başlat
        if event.button() == Qt.LeftButton:
            self._is_panning = True
            self._pan_start = event.position().toPoint()

            # Tıklama sinyali (Shift vs. kontrolü dışarıda yapılıyor)
            view_pt = event.position().toPoint()
            self.clicked.emit(self.mapToScene(view_pt), event.globalPosition().toPoint())

            # İmleci değiştirmek istersen (Örn: yumruk yap)
            # self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._is_panning = False
            # Bırakınca tekrar gizle (veya ince artı yap)
            self.setCursor(Qt.BlankCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        # 1. Sürükleme Mantığı
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()

            # Scrollbarları manuel kaydır
            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            hs.setValue(hs.value() - delta.x())
            vs.setValue(vs.value() - delta.y())
            event.accept()

        # 2. Koordinat Sinyali (Crosshair için)
        self.moved.emit(self.mapToScene(event.position().toPoint()))
        super().mouseMoveEvent(event)


# -------------------- Map Panel --------------------
class MapPanel(QtWidgets.QWidget):
    """QGraphicsView tabanlı basit görüntü haritası.
    - Drag: sol tuş ile sürükle
    - Zoom: butonlar +/-, ayrıca tekerlek destekli
    - Lat/Lon -> piksel dönüşümü WebMercator formülleri ile
    """
    # İndeks (1-5), Lat, Lon gönderir
    gps_selected = Signal(int, float, float, int)

    def __init__(self, image_path: str, north: float, south: float, west: float, east: float, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(420)
        self.setMinimumHeight(400)
        self.image_path = image_path
        self.north, self.south, self.west, self.east = north, south, west, east
        self._R = 6378137.0  # WebMercator yarıçap
        self._max_zoom = 16.0
        self._min_zoom = 0.25

        # Scene & View
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = GraphicsView(self.scene, self, min_zoom=self._min_zoom, max_zoom=self._max_zoom)

        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._on_context_menu)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        # Scrollbarları gizle (HUD şıklığı için)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)

        # Pixmap (büyük görselleri otomatik küçült)
        MAX_DIM = 32000  # QPixmap/texture güvenli sınır

        # --- MESAFE HALKALARI (RANGE RINGS) ---
        self.ring_group = QtWidgets.QGraphicsItemGroup()
        self.ring_group.setZValue(13)
        self.scene.addItem(self.ring_group)
        self.rings_enabled = False
        self.ring_step_m = 5.0
        self.ring_count = 3

        # --- SANAL ÇİT (GEOFENCE) ---
        self._fence_mode = False
        self._fence_points_latlon = []
        self.fence_polygon = QtWidgets.QGraphicsPolygonItem()
        pen = QtGui.QPen(QtGui.QColor("#ff1744"))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        brush = QtGui.QBrush(QtGui.QColor(255, 23, 68, 50))
        self.fence_polygon.setPen(pen)
        self.fence_polygon.setBrush(brush)
        self.fence_polygon.setZValue(14)
        self.scene.addItem(self.fence_polygon)

        # --- HAYALET İZ (GHOST TRAIL) ---
        self.ghost_pen = QtGui.QPen(QtGui.QColor(200, 200, 200, 120))
        self.ghost_pen.setWidthF(1.5)
        self.ghost_pen.setStyle(Qt.DashLine)
        self.ghost_path = QtGui.QPainterPath()
        self.ghost_item = self.scene.addPath(self.ghost_path, self.ghost_pen)
        self.ghost_item.setZValue(11)

        # --- AKILLI CETVEL (RULER) ---
        self._ruler_active = False
        self._ruler_start_lat = None
        self._ruler_start_lon = None
        self._ruler_start_scene = None
        self.ruler_line = QtWidgets.QGraphicsLineItem()
        pen = QtGui.QPen(QtGui.QColor("yellow"))
        pen.setWidth(2);
        pen.setStyle(Qt.DashLine)
        self.ruler_line.setPen(pen);
        self.ruler_line.setZValue(900);
        self.ruler_line.setVisible(False)
        self.scene.addItem(self.ruler_line)

        self.ruler_text = QtWidgets.QGraphicsTextItem("")
        self.ruler_text.setDefaultTextColor(QtGui.QColor("yellow"))
        f = self.ruler_text.font();
        f.setBold(True);
        f.setPointSize(10);
        self.ruler_text.setFont(f)
        self.ruler_text.setZValue(901);
        self.ruler_text.setVisible(False)

        self.ruler_bg = QtWidgets.QGraphicsRectItem()
        self.ruler_bg.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 150)))
        self.ruler_bg.setZValue(900);
        self.ruler_bg.setVisible(False)
        self.scene.addItem(self.ruler_bg);
        self.scene.addItem(self.ruler_text)

        # --- PARKUR (COURSE) YÖNETİMİ ---
        self._course_mode = None
        self._temp_corners = []
        self._temp_corner_items = []
        self.courses = {
            "A": {"corners": [], "rows": 4, "cols": 4, "visible": True, "color": QtGui.QColor(0, 255, 255)},
            "B": {"corners": [], "rows": 4, "cols": 4, "visible": True, "color": QtGui.QColor(255, 0, 255)},
            "C": {"corners": [], "rows": 4, "cols": 4, "visible": True, "color": QtGui.QColor(255, 255, 0)}
        }
        self.course_groups = {
            "A": self.scene.createItemGroup([]),
            "B": self.scene.createItemGroup([]),
            "C": self.scene.createItemGroup([])
        }
        for k, grp in self.course_groups.items():
            self.scene.addItem(grp);
            grp.setZValue(5)

        # --- GÖRÜNTÜ YÜKLEME ---
        if not os.path.exists(self.image_path):
            pm = QtGui.QPixmap(800, 600);
            pm.fill(QtGui.QColor("#555"))
            self._img_W, self._img_H = pm.width(), pm.height()
            self.image_item = self.scene.addPixmap(pm)
        else:
            self.image_item = HugeImageItem(self.image_path, tile_size=2048, cache_limit=128)
            self.scene.addItem(self.image_item)
            self._img_W, self._img_H = self.image_item.width, self.image_item.height

        # Overlay grubu
        self.overlay_group = QtWidgets.QGraphicsItemGroup();
        self.overlay_group.setZValue(10)
        self.scene.addItem(self.overlay_group)

        # Mevcut konum işaretçisi (kırmızı)
        r = CURR_RADIUS_PX
        self.curr_marker = QtWidgets.QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        self.curr_marker.setBrush(QtGui.QBrush(QtGui.QColor("#ff1744")))
        self.curr_marker.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.0))
        self.curr_marker.setZValue(20);
        self.curr_marker.setVisible(False)
        if KEEP_PIXEL_SIZE: self.curr_marker.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.scene.addItem(self.curr_marker)

        self.path_pen = QtGui.QPen(QtGui.QColor("#ff5252"));
        self.path_pen.setWidthF(PATH_WIDTH_PX)
        self.path_pen.setCosmetic(True)  # <--- KRİTİK: Zoom yapınca incelmesin!
        self.path_path = QtGui.QPainterPath()
        self.path_item = self.scene.addPath(self.path_path, self.path_pen);
        self.path_item.setZValue(12)

        # İz geçmişini tutacak listeler (Maks 500m için)
        self.trail_points_1 = []
        self.trail_points_2 = []
        # --- YUMUŞATMA (SMOOTHING) VE İNTERPOLASYON DEĞİŞKENLERİ ---
        self._target_lat_1 = None
        self._target_lon_1 = None
        self._visual_lat_1 = None
        self._visual_lon_1 = None

        self._target_lat_2 = None
        self._target_lon_2 = None
        self._visual_lat_2 = None
        self._visual_lon_2 = None

        # Saniyede ~30 kare güncelleyecek animasyon zamanlayıcısı
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._animate_markers)
        self.anim_timer.start(33)  # ~30 FPS (1000ms / 30)


        # --- Heading overlays ---
        self._last_lat = None;
        self._last_lon = None;
        self._real_hdg_deg = None;
        self._targ_hdg_deg = None
        self._real_line = QtWidgets.QGraphicsLineItem();
        real_pen = QtGui.QPen(QtGui.QColor("#ff1744"));
        real_pen.setWidthF(2.2)
        self._real_line.setPen(real_pen);
        self._real_line.setZValue(18);
        self.scene.addItem(self._real_line);
        self._real_line.setVisible(False)
        self._real_arrow = QtWidgets.QGraphicsPathItem();
        self._real_arrow.setPen(real_pen);
        self._real_arrow.setZValue(19);
        self.scene.addItem(self._real_arrow);
        self._real_arrow.setVisible(False)
        self._targ_line = QtWidgets.QGraphicsLineItem();
        targ_pen = QtGui.QPen(QtGui.QColor("#00e676"));
        targ_pen.setWidthF(2.0)
        self._targ_line.setPen(targ_pen);
        self._targ_line.setZValue(17);
        self.scene.addItem(self._targ_line);
        self._targ_line.setVisible(False)

        # Waypoint katmanı
        self.wp_group = QtWidgets.QGraphicsItemGroup();
        self.wp_group.setZValue(15);
        self.scene.addItem(self.wp_group)

        # Yerleşim
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)

        # Sinyaller
        self.view.clicked.connect(self._on_map_click)
        self.view.moved.connect(self._on_mouse_move)

        # Etiketler
        self.click_labels_root = QtWidgets.QGraphicsItemGroup();
        self.click_labels_root.setZValue(500);
        self.scene.addItem(self.click_labels_root)

        # Crosshair
        self.crosshair = QtWidgets.QGraphicsItemGroup()
        pen = QtGui.QPen(QtGui.QColor("#ffff00"));
        pen.setWidthF(1.6)
        l1 = QtWidgets.QGraphicsLineItem(-5, 0, 5, 0);
        l2 = QtWidgets.QGraphicsLineItem(0, -5, 0, 5)
        for ln in (l1, l2):
            ln.setPen(pen);
            ln.setZValue(1000);
            ln.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True);
            ln.setAcceptedMouseButtons(Qt.NoButton)
            self.crosshair.addToGroup(ln)
        self.crosshair.setZValue(1000);
        self.crosshair.setAcceptedMouseButtons(Qt.NoButton)
        self.scene.addItem(self.crosshair);
        self.crosshair.setVisible(True)

        # --- YENİ EKLENTİ BURADA BAŞLIYOR (ESKİ _btn_box SİLİNDİ) ---
        # Butonları "Overlay" olarak ekleyen fonksiyonu çağırıyoruz
        self._setup_overlay_controls()
        self._label_offset = QtCore.QPointF(8, -8)

        # Başlangıç
        self.view.setSceneRect(QtCore.QRectF(0, 0, self._img_W, self._img_H))
        self.center_image()
        self.set_zoom(0.2)

        self._click_labels = []
        self.view.zoomChanged.connect(self._relayout_click_labels)
        self._debug_corners()
        self.cached_pts_1 = []
        self.cached_pts_2 = []

    def _setup_overlay_controls(self):
        """Askeri tema HUD butonlarını oluşturur (Sol: Zoom+Büyüteç, Sağ: Pan+Pusula)."""

        # Ortak Stil
        hud_style = """
            QPushButton {
                background-color: rgba(20, 20, 20, 200);
                color: #00e5ff;
                border: 1px solid #546e7a;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: rgba(0, 229, 255, 40);
                border: 1px solid #00e5ff;
                color: white;
            }
            QPushButton:pressed {
                background-color: rgba(0, 229, 255, 100);
                border: 1px solid #00e5ff;
                padding-top: 2px;
                padding-left: 2px;
            }
            QLabel {
                color: #00e5ff;
                font-size: 22px;
                background: transparent;
                border: none;
            }
        """

        # --- 1. ZOOM PANELİ (SOL ÜST İÇİN) ---
        self.zoom_widget = QtWidgets.QWidget(self)
        self.zoom_widget.setStyleSheet(hud_style)

        v_zoom = QtWidgets.QVBoxLayout(self.zoom_widget)
        v_zoom.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        v_zoom.setContentsMargins(4, 4, 4, 4)
        v_zoom.setSpacing(4)
        v_zoom.setAlignment(Qt.AlignCenter)

        z_size = 38

        self.btn_in = QtWidgets.QPushButton("+")
        self.btn_in.setFixedSize(z_size, z_size)
        self.btn_in.clicked.connect(lambda: self.zoom(1.25))

        lbl_magnifier = QtWidgets.QLabel("🔍")
        lbl_magnifier.setAlignment(Qt.AlignCenter)

        self.btn_out = QtWidgets.QPushButton("−")
        self.btn_out.setFixedSize(z_size, z_size)
        self.btn_out.clicked.connect(lambda: self.zoom(0.8))

        v_zoom.addWidget(self.btn_in)
        v_zoom.addWidget(lbl_magnifier)
        v_zoom.addWidget(self.btn_out)

        # --- 2. PAN PANELİ (SAĞ ÜST İÇİN) ---
        self.pan_widget = QtWidgets.QWidget(self)
        self.pan_widget.setStyleSheet(hud_style)

        g_pan = QtWidgets.QGridLayout(self.pan_widget)
        g_pan.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        g_pan.setContentsMargins(4, 4, 4, 4)
        g_pan.setSpacing(4)

        p_size = 34

        btn_u = QtWidgets.QPushButton("▲");
        btn_u.setFixedSize(p_size, p_size)
        btn_u.clicked.connect(lambda: self._pan_map(0, -150))

        btn_d = QtWidgets.QPushButton("▼");
        btn_d.setFixedSize(p_size, p_size)
        btn_d.clicked.connect(lambda: self._pan_map(0, 150))

        btn_l = QtWidgets.QPushButton("◀");
        btn_l.setFixedSize(p_size, p_size)
        btn_l.clicked.connect(lambda: self._pan_map(-150, 0))

        btn_r = QtWidgets.QPushButton("▶");
        btn_r.setFixedSize(p_size, p_size)
        btn_r.clicked.connect(lambda: self._pan_map(150, 0))

        lbl_nav = QtWidgets.QLabel("🧭")
        lbl_nav.setAlignment(Qt.AlignCenter)

        g_pan.addWidget(btn_u, 0, 1)
        g_pan.addWidget(btn_l, 1, 0)
        g_pan.addWidget(lbl_nav, 1, 1)
        g_pan.addWidget(btn_r, 1, 2)
        g_pan.addWidget(btn_d, 2, 1)

    def _pan_map(self, dx, dy):
        """Haritayı butonlarla kaydırma fonksiyonu."""
        h_bar = self.view.horizontalScrollBar()
        v_bar = self.view.verticalScrollBar()
        h_bar.setValue(h_bar.value() + dx)
        v_bar.setValue(v_bar.value() + dy)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        """Pencere boyutu değişince butonları köşelere yapıştır."""
        super().resizeEvent(e)

        margin = 15  # Kenar boşluğu

        # 1. ZOOM PANELİ -> SOL ÜST KÖŞE
        # Önce boyutunu güncelle
        self.zoom_widget.adjustSize()
        # Sola yasla (X = margin)
        self.zoom_widget.move(margin, margin)

        # 2. PAN PANELİ -> SAĞ ÜST KÖŞE
        # Önce boyutunu güncelle
        self.pan_widget.adjustSize()

        # SAĞA YASLAMA MATEMATİĞİ:
        # X = (Pencere Genişliği) - (Panel Genişliği) - (Boşluk)
        pan_width = self.pan_widget.width()
        pan_x = self.width() - pan_width - margin

        self.pan_widget.move(pan_x, margin)
    def snapshot_ghost_trail(self):
        """Mevcut kırmızı izi dondurup gri hayalet ize çevirir ve canlı yolu temizler."""
        self.ghost_path.addPath(self.path_path)
        self.ghost_item.setPath(self.ghost_path)

        # Çizgiyi sıfırla
        self.path_path = QtGui.QPainterPath()
        self.path_item.setPath(self.path_path)

        # Canlı iz hafızasını temizle, sadece son konumu bırak ki yeni iz oradan başlasın
        if self.trail_points_1:
            self.trail_points_1 = [self.trail_points_1[-1]]

    def change_map(self, image_path: str, north: float, south: float, west: float, east: float):
        """Harita görselini ve koordinat sınırlarını dinamik olarak değiştirir."""
        self.image_path = image_path
        self.north, self.south, self.west, self.east = north, south, west, east

        # 1. Eski harita görselini sahneden kaldır
        if hasattr(self, 'image_item') and self.image_item:
            self.scene.removeItem(self.image_item)
            self.image_item = None

        # 2. Yeni görseli yükle (HugeImageItem veya QPixmap)
        if not os.path.exists(self.image_path):
            # Dosya yoksa gri placeholder
            pm = QtGui.QPixmap(800, 600)
            pm.fill(QtGui.QColor("#333"))
            painter = QtGui.QPainter(pm)
            painter.setPen(QtGui.QPen(Qt.white))
            painter.drawText(pm.rect(), Qt.AlignCenter, f"Harita Yüklenemedi:\n{os.path.basename(self.image_path)}")
            painter.end()
            self._img_W, self._img_H = pm.width(), pm.height()
            self.image_item = self.scene.addPixmap(pm)
        else:
            # Varsa HugeImageItem kullan
            self.image_item = HugeImageItem(self.image_path, tile_size=2048, cache_limit=128)
            self.scene.addItem(self.image_item)
            self._img_W, self._img_H = self.image_item.width, self.image_item.height

        # 3. Görseli en arkaya (Z=0) gönder
        self.image_item.setZValue(0)

        # 4. Sahne boyutunu güncelle ve ortala
        self.view.setSceneRect(QtCore.QRectF(0, 0, self._img_W, self._img_H))
        self.center_image()

        # 5. Varsa eski çizimleri (Waypoint, Rota) temizle veya yeni koordinata göre güncelle
        # (Genelde harita değişince eski waypointler anlamsız kalır, o yüzden waypointleri siliyoruz)
        self.path_path = QtGui.QPainterPath()
        self.path_item.setPath(self.path_path)
        # Waypointleri de temizlemek istersen:
        for item in self.wp_group.childItems(): self.scene.removeItem(item)

    def update_range_rings(self):
        """Mevcut konum etrafına mesafe halkalarını çizer."""
        # Önce eskileri temizle
        for item in self.ring_group.childItems():
            self.scene.removeItem(item)

        if not self.rings_enabled:
            return

        # Teknenin konumu yoksa çizemeyiz
        if self._last_lat is None or self._last_lon is None:
            return

        # --- MATEMATİK: 1 Metre kaç piksel eder? ---
        # WebMercator projeksiyonunda ölçek enleme göre değişir.
        # Basitçe: Teknenin olduğu yerden 'step' metre doğuya gidince piksel ne kadar değişiyor?

        lat = self._last_lat
        lon = self._last_lon

        # Merkez (Tekne) pikseli
        center_pt = self._scene_pos_from_latlon(lat, lon)
        if center_pt is None: return

        # Referans nokta: 'step' metre doğusu
        # (offset_latlon fonksiyonunu daha önce eklemiştik, onu kullanıyoruz)
        _, lon_east = self._offset_latlon(lat, lon, 0, self.ring_step_m)
        east_pt = self._scene_pos_from_latlon(lat, lon_east)

        if east_pt is None: return

        # Yarıçap (piksel cinsinden 'step' metre)
        # Sadece X farkına bakmak yeterli (daire olduğu için)
        radius_px_base = abs(east_pt.x() - center_pt.x())

        # --- ÇİZİM ---
        pen = QtGui.QPen(QtGui.QColor("#00e676"))  # Radar Yeşili
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)  # Kesikli çizgi radar hissi verir

        font = QtGui.QFont("Arial", 8)
        font.setBold(True)

        for i in range(1, self.ring_count + 1):
            r_px = radius_px_base * i
            dist_label = int(self.ring_step_m * i)

            # Halka (Ellipse)
            # x, y, w, h -> Sol üst köşe ve boyut ister
            ellipse = QtWidgets.QGraphicsEllipseItem(
                center_pt.x() - r_px,
                center_pt.y() - r_px,
                r_px * 2,
                r_px * 2
            )
            ellipse.setPen(pen)
            self.ring_group.addToGroup(ellipse)

            # Etiket (Örn: "10m") - Halkanın tam üstüne koyalım
            txt = QtWidgets.QGraphicsTextItem(f"{dist_label}m")
            txt.setFont(font)
            txt.setDefaultTextColor(QtGui.QColor("#00e676"))
            # Yazıyı ortala
            br = txt.boundingRect()
            txt.setPos(center_pt.x() - br.width() / 2, center_pt.y() - r_px - br.height())
            self.ring_group.addToGroup(txt)

    def clear_ghost_trail(self):
        """Hayalet izi tamamen siler."""
        self.ghost_path = QtGui.QPainterPath()
        self.ghost_item.setPath(self.ghost_path)

    def start_defining_course(self, course_name):
        """Köşe belirleme modunu açar."""
        self._course_mode = course_name
        self._temp_corners = []
        # Varsa eskileri temizle
        self.clear_course_visuals(course_name)
        # Mouse imlecini değiştir
        self.view.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.view.setCursor(Qt.CrossCursor)

    def _latlon_interp(self, p1, p2, ratio):
        """İki lat/lon arasında orantısal nokta bulur (Lineer interpolasyon)."""
        lat = p1[0] + (p2[0] - p1[0]) * ratio
        lon = p1[1] + (p2[1] - p1[1]) * ratio
        return (lat, lon)

    def render_course_grid(self, name):
        """Verilen isimdeki kursun ızgarasını çizer (Daha Saydam Versiyon)."""
        data = self.courses[name]
        corners = data["corners"]
        rows = data["rows"]
        cols = data["cols"]
        base_color = data["color"]  # Bu saf renk (Örn: Cyan)
        grp = self.course_groups[name]

        # Önce grubu temizle
        for item in grp.childItems():
            self.scene.removeItem(item)

        if len(corners) != 4:
            return

        # --- RENK AYARLARI (SAYDAMLAŞTIRMA) ---
        # 1. Çizgi Rengi (Yarı Saydam)
        line_color = QtGui.QColor(base_color)
        line_color.setAlpha(40)  # 0-255 arası (120 = %50 Görünürlük) ____120

        # 2. Dolgu Rengi (Çok Saydam)
        fill_color = QtGui.QColor(base_color)
        fill_color.setAlpha(5)  # Neredeyse yok gibi ____ 25

        # 3. Köşe Noktası Rengi (Biraz daha belirgin kalsın)
        corner_color = QtGui.QColor(base_color)
        corner_color.setAlpha(40) #_______180

        # -------------------------------------

        # 1. Köşeleri çiz (Daha kibar noktalar)
        for i, (lat, lon) in enumerate(corners):
            p = self._scene_pos_from_latlon(lat, lon)
            if p:
                r = 3  # Yarıçapı biraz küçülttük (4->3)
                el = QtWidgets.QGraphicsEllipseItem(p.x() - r, p.y() - r, 2 * r, 2 * r)
                el.setBrush(QtGui.QBrush(corner_color))
                el.setPen(Qt.NoPen)  # Kenar çizgisi yok
                grp.addToGroup(el)

                # Etiket (1,2,3,4) - Hafif saydam
                txt = QtWidgets.QGraphicsTextItem(f"{name}{i + 1}")
                txt.setDefaultTextColor(line_color)
                f = txt.font();
                f.setPixelSize(9);
                txt.setFont(f)
                txt.setPos(p + QtCore.QPointF(4, -12))
                grp.addToGroup(txt)

        # 2. Dış Çerçeveyi Çiz (Polygon)
        pts_scene = [self._scene_pos_from_latlon(c[0], c[1]) for c in corners]
        if all(pts_scene):
            poly = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF(pts_scene))

            # Çerçeve Çizgisi
            pen = QtGui.QPen(line_color)
            pen.setWidth(2)
            poly.setPen(pen)

            # İç Dolgu
            poly.setBrush(QtGui.QBrush(fill_color))
            grp.addToGroup(poly)

        # 3. Izgara (Grid) Çizimi (Bilinear)
        c0, c1, c2, c3 = corners[0], corners[1], corners[2], corners[3]

        grid_pen = QtGui.QPen(line_color)
        grid_pen.setStyle(Qt.DotLine)  # Noktalı çizgi
        grid_pen.setWidth(1)

        # Dikey Çizgiler (Sütunlar)
        if cols > 0:
            for i in range(1, cols + 1):
                ratio = i / (cols + 1)
                top_pt_ll = self._latlon_interp(c0, c1, ratio)
                bot_pt_ll = self._latlon_interp(c3, c2, ratio)

                p_top = self._scene_pos_from_latlon(*top_pt_ll)
                p_bot = self._scene_pos_from_latlon(*bot_pt_ll)

                if p_top and p_bot:
                    l = QtWidgets.QGraphicsLineItem(QtCore.QLineF(p_top, p_bot))
                    l.setPen(grid_pen)
                    grp.addToGroup(l)

        # Yatay Çizgiler (Satırlar)
        if rows > 0:
            for i in range(1, rows + 1):
                ratio = i / (rows + 1)
                left_pt_ll = self._latlon_interp(c0, c3, ratio)
                right_pt_ll = self._latlon_interp(c1, c2, ratio)

                p_left = self._scene_pos_from_latlon(*left_pt_ll)
                p_right = self._scene_pos_from_latlon(*right_pt_ll)

                if p_left and p_right:
                    l = QtWidgets.QGraphicsLineItem(QtCore.QLineF(p_left, p_right))
                    l.setPen(grid_pen)
                    grp.addToGroup(l)

    def clear_course_visuals(self, name):
        # Önce kalıcı ızgarayı sil
        grp = self.course_groups.get(name)
        if grp:
            for item in grp.childItems():
                self.scene.removeItem(item)

        # --- DEĞİŞİKLİK: Eğer çizim yarım kaldıysa o beyaz noktaları da sil ---
        if self._course_mode == name:
            self._clear_temp_visuals()
            self._temp_corners = []  # Mantıksal listeyi de sıfırla

    def _clear_temp_visuals(self):
        """Geçici beyaz noktaları sahneden siler."""
        for item in self._temp_corner_items:
            try:
                self.scene.removeItem(item)
            except:
                pass
        self._temp_corner_items = []

    def start_fence_drawing(self):
        """Çizim modunu başlatır ve eski çiti temizler."""
        self._fence_mode = True
        self._fence_points_latlon = []
        self.fence_polygon.setPolygon(QtGui.QPolygonF())  # Görseli temizle
        self.view.setDragMode(QtWidgets.QGraphicsView.NoDrag)  # Harita kaymasın, rahat çizilsin

    def finish_fence_drawing(self):
        """Çizimi bitirir."""
        self._fence_mode = False
        self.view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)  # Tekrar kaydırma modu

    def clear_fence(self):
        """Çiti tamamen siler."""
        self._fence_points_latlon = []
        self.fence_polygon.setPolygon(QtGui.QPolygonF())

    def get_fence_points(self):
        return self._fence_points_latlon

    def _on_context_menu(self, pos: QtCore.QPoint):
        """Haritaya sağ tıklandığında menü açar."""
        scene_pos = self.view.mapToScene(pos)
        lat, lon = self._latlon_from_scene(scene_pos.x(), scene_pos.y())

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #333; color: white; border: 1px solid #555; }
            QMenu::item:selected { background-color: #555; }
        """)

        header = menu.addAction(f"K: {lat:.5f}, {lon:.5f}")
        header.setEnabled(False)
        menu.addSeparator()

        # --- IDA 1 MENÜSÜ ---
        menu_ida1 = menu.addMenu("🔴 IDA 1 İçin...")

        # IDA 1 Yeni Ekle
        action_add1 = menu_ida1.addAction("➕ Listenin Sonuna Ekle")
        action_add1.triggered.connect(lambda: self.gps_selected.emit(1, lat, lon, -1))

        menu_ida1.addSeparator()

        # IDA 1 Mevcut Noktaları Değiştir (Ekranda görünen noktalar kadar seçenek çıkar)
        for i in range(len(self.cached_pts_1)):
            action_update = menu_ida1.addAction(f"✏️ {i + 1}. Noktayı Değiştir")
            # i değerini lambda içine hapsetmek için m=i kullanıyoruz (Python closure trick)
            action_update.triggered.connect(lambda checked=False, m=i: self.gps_selected.emit(1, lat, lon, m))

        menu.exec(self.view.mapToGlobal(pos))

    def _latlon_from_scene(self, px: float, py: float) -> tuple[float, float]:
        x_left = self._lon_to_x(self.west)
        x_right = self._lon_to_x(self.east)
        y_top = self._lat_to_y(self.north)
        y_bot = self._lat_to_y(self.south)

        W = float(self._img_W)
        H = float(self._img_H)

        x_merc = x_left + (px / W) * (x_right - x_left)
        y_merc = y_top - (py / H) * (y_top - y_bot)

        lon = math.degrees(x_merc / self._R)
        lat = math.degrees(2.0 * math.atan(math.exp(y_merc / self._R)) - math.pi / 2.0)
        return lat, lon

    # MapPanel sınıfının içine ekle:
    def start_smart_ruler(self):
        """Shift'e basıldığı an fare konumunu başlangıç noktası yapar."""
        if self._ruler_active: return  # Zaten açıksa tekrar başlatma

        # Farenin ekran üzerindeki (Global) konumunu al
        global_pos = QtGui.QCursor.pos()
        # Bunu harita üzerindeki (View) konuma çevir
        view_pos = self.view.mapFromGlobal(global_pos)

        # Eğer fare harita sınırları içinde değilse başlatma
        if not self.view.rect().contains(view_pos):
            return

        # Scene (Sahne) koordinatına çevir
        scene_pos = self.view.mapToScene(view_pos)

        # Değişkenleri ayarla
        self._ruler_active = True
        self._ruler_start_scene = scene_pos
        self._ruler_start_lat, self._ruler_start_lon = self._latlon_from_scene(scene_pos.x(), scene_pos.y())

        # Görselleri görünür yap
        self.ruler_line.setVisible(True)
        self.ruler_text.setVisible(True)
        self.ruler_bg.setVisible(True)

        # İlk çizgi güncellemesi (Nokta şeklinde başlasın)
        self.ruler_line.setLine(scene_pos.x(), scene_pos.y(), scene_pos.x(), scene_pos.y())

    def stop_smart_ruler(self):
        """Cetveli kapatır ve gizler."""
        self._ruler_active = False
        self.ruler_line.setVisible(False)
        self.ruler_text.setVisible(False)
        self.ruler_bg.setVisible(False)

    def toggle_smart_ruler(self):
        """Cetvel açıksa kapatır, kapalıysa başlatır."""
        if self._ruler_active:
            self.stop_smart_ruler()
        else:
            self.start_smart_ruler()

    @QtCore.Slot(QtCore.QPointF)
    def _on_mouse_move(self, scene_pt: QtCore.QPointF):
        # kırmızı + imleci takip etsin
        self.crosshair.setPos(scene_pt)

        # --- CETVEL GÜNCELLEMESİ ---
        if self._ruler_active and self._ruler_start_lat is not None:
            # 1. Çizgiyi güncelle
            line = QtCore.QLineF(self._ruler_start_scene, scene_pt)
            self.ruler_line.setLine(line)

            # 2. Mesafeyi hesapla
            curr_lat, curr_lon = self._latlon_from_scene(scene_pt.x(), scene_pt.y())
            dist_m = haversine_distance(self._ruler_start_lat, self._ruler_start_lon, curr_lat, curr_lon)
            bearing_deg = calculate_bearing(self._ruler_start_lat, self._ruler_start_lon, curr_lat, curr_lon)

            # 3. Metni güncelle
            txt = f"{dist_m:.2f} m\n{bearing_deg:.1f}°"
            self.ruler_text.setPlainText(txt)

            # 4. Metni imlecin biraz yanına koy
            self.ruler_text.setPos(scene_pt + QtCore.QPointF(15, 15))

            # 5. Arka plan kutusunu metne göre ayarla
            br = self.ruler_text.boundingRect()
            self.ruler_bg.setRect(br)
            self.ruler_bg.setPos(self.ruler_text.pos())

    def _ascend_to_label_group(self, item: QtWidgets.QGraphicsItem | None):
        it = item
        while it is not None:
            if isinstance(it, QtWidgets.QGraphicsItemGroup) and it.data(0) == "click_label":
                return it
            it = it.parentItem()
        return None

    def _relayout_click_labels(self, *_):
        scale = max(1e-9, self.view.transform().m11())
        ofs_scene = QtCore.QPointF(self._label_offset.x() / scale, self._label_offset.y() / scale)
        for it in list(self._click_labels):
            grp = it.get("grp")
            line = it.get("line")
            scene_pt = it.get("scene_pt")
            bg_rect = it.get("bg_rect")
            if not grp or not line:
                continue
            # etiketi yeniden konumlandır
            grp.setPos(scene_pt + ofs_scene)
            # çizgi başlangıcını sahnede yeniden hesapla
            start_scene = grp.pos() + QtCore.QPointF(bg_rect.left() / scale, bg_rect.top() / scale)
            line.setLine(QtCore.QLineF(start_scene, scene_pt))

    def _add_click_label(self, scene_pt: QtCore.QPointF, lat: float, lon: float):
        # --- METİN ---
        txt = QtWidgets.QGraphicsTextItem(f"{lat:.6f}, {lon:.6f}")
        f = txt.font()
        f.setPointSizeF(8.5)
        f.setBold(True)
        txt.setFont(f)
        txt.setDefaultTextColor(QtGui.QColor("#ffffff"))
        txt.setAcceptedMouseButtons(Qt.NoButton)

        # --- ARKA PLAN ---
        br = txt.boundingRect()
        bg_rect = QtCore.QRectF(-4, -2, br.width() + 8, br.height() + 4)

        class _BgRect(QtWidgets.QGraphicsRectItem):
            def __init__(self, rect, closer):
                super().__init__(rect)
                self._closer = closer

            def mousePressEvent(self, e):
                if e.button() == Qt.LeftButton:
                    self._closer()
                else:
                    super().mousePressEvent(e)

        # --- GRUP (sabit piksel boyutlu etiket) ---
        grp = QtWidgets.QGraphicsItemGroup()
        grp.setOpacity(0.30)
        grp.setZValue(600)
        grp.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)

        bg = _BgRect(bg_rect, closer=lambda: None)
        bg.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        bg.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))

        grp.addToGroup(bg)
        grp.addToGroup(txt)

        # --- POZİSYON: ofseti ölçekle normalize et ---
        scale = max(1e-9, self.view.transform().m11())
        ofs_scene = QtCore.QPointF(self._label_offset.x() / scale, self._label_offset.y() / scale)
        grp.setPos(scene_pt + ofs_scene)
        self.scene.addItem(grp)

        # --- KIRMIZI ÇİZGİ: sol ÜST köşeden piksele ---
        start_scene = grp.pos() + QtCore.QPointF(bg_rect.left() / scale, bg_rect.top() / scale)
        line = self.scene.addLine(QtCore.QLineF(start_scene, scene_pt),
                                  QtGui.QPen(QtGui.QColor("#ff1744"), KIRMIZI_CIZGI_KALINLIK))
        line.setZValue(599)

        # --- KAPATMA ---
        def _close():
            # sahneden sil
            try:
                self.scene.removeItem(line)
            except Exception:
                pass
            try:
                self.scene.removeItem(grp)
            except Exception:
                pass
            # listeden düş
            self._click_labels[:] = [it for it in self._click_labels if it.get("grp") is not grp]

        bg._closer = _close  # etikete tıklayınca hemen kapanır

        # --- 10 sn sonra otomatik kaldır ---
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(_close)
        timer.start(10_000)  # ms

        # listede sakla (zoom’da relayout için)
        self._click_labels.append({
            "grp": grp,
            "bg_rect": QtCore.QRectF(bg_rect),
            "scene_pt": QtCore.QPointF(scene_pt),
            "line": line,
            "timer": timer,
        })

    @QtCore.Slot(QtCore.QPointF, QtCore.QPoint)
    def _on_map_click(self, scene_pt: QtCore.QPointF, global_pos: QtCore.QPoint):

        # 1. CETVELİ KAPATMA (Sol tıklandığında)
        if self._ruler_active:
            self.stop_smart_ruler()  # <-- Kodu kısalttık
            return

        # 2. --- GEOFENCE ÇİZİMİ ---
        if self._fence_mode:
            # Koordinatı al
            lat, lon = self._latlon_from_scene(scene_pt.x(), scene_pt.y())
            self._fence_points_latlon.append((lat, lon))

            # Görseli güncelle (LatLon listesini ScenePoint listesine çevirip çiz)
            poly_points = []
            for (flat, flon) in self._fence_points_latlon:
                p = self._scene_pos_from_latlon(flat, flon)
                if p: poly_points.append(p)

            self.fence_polygon.setPolygon(QtGui.QPolygonF(poly_points))
            return  # Başka işlem yapma

        # 3. --- PARKUR KÖŞE BELİRLEME ---
        if self._course_mode:
            lat, lon = self._latlon_from_scene(scene_pt.x(), scene_pt.y())
            self._temp_corners.append((lat, lon))

            # Geçici nokta koy (Görsel geri bildirim)
            r = 3
            dot = QtWidgets.QGraphicsEllipseItem(scene_pt.x() - r, scene_pt.y() - r, 2 * r, 2 * r)
            dot.setBrush(Qt.white)
            dot.setPen(QtGui.QPen(Qt.black))  # Siyah kenarlık ekledim daha net görünsün
            self.scene.addItem(dot)

            # --- DEĞİŞİKLİK BURADA: Noktayı listeye kaydet ---
            self._temp_corner_items.append(dot)

            # 4 Köşe tamamlandı mı?
            if len(self._temp_corners) == 4:
                # Kaydet
                nm = self._course_mode
                self.courses[nm]["corners"] = list(self._temp_corners)

                # Moddan çık
                self._course_mode = None
                self.view.setDragMode(
                    QtWidgets.QGraphicsView.NoDrag)  # Qt'nin kendi sürüklemesini kapat (Bizimki çalışsın)
                self.view.setCursor(Qt.BlankCursor)  # İmleci tekrar gizle (Sadece Crosshair kalsın)

                # --- DEĞİŞİKLİK BURADA: Geçici beyaz noktaları temizle ---
                self._clear_temp_visuals()

                # Izgarayı Çiz
                self.render_course_grid(nm)

            return


    def _debug_corners(self):
        """Harita BBOX köşelerine yeşil çarpı koyarak hizalamayı test eder."""
        # Eski debug objelerini temizle
        try:
            for it in getattr(self, "_debug_items", []):
                self.scene.removeItem(it)
        except Exception:
            pass
        self._debug_items = []

        def cross(p, size=8, color="#00c853"):
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidthF(1.5)
            l1 = QtWidgets.QGraphicsLineItem(p.x() - size, p.y(), p.x() + size, p.y())
            l2 = QtWidgets.QGraphicsLineItem(p.x(), p.y() - size, p.x(), p.y() + size)
            l1.setPen(pen)
            l2.setPen(pen)
            l1.setZValue(30)
            l2.setZValue(30)
            # Zoom’dan bağımsız olsun istiyorsan:
            if globals().get("KEEP_PIXEL_SIZE", True):
                l1.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
                l2.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
            self.scene.addItem(l1)
            self.scene.addItem(l2)
            self._debug_items.extend([l1, l2])

        # NW, NE, SW, SE köşelerine çarpı
        for (lat, lon) in [
            (self.north, self.west), (self.north, self.east),
            (self.south, self.west), (self.south, self.east)
        ]:
            p = self._scene_pos_from_latlon(lat, lon)
            if p is not None:
                cross(p)

    # ---- Zoom / Pan ----
    def current_zoom(self) -> float:
        m = self.view.transform()
        # scaleX
        return m.m11()

    def set_zoom(self, z: float):
        z = max(self._min_zoom, min(self._max_zoom, z))
        cur = self.current_zoom()
        if cur == 0:
            return
        self.view.scale(z / cur, z / cur)

    def zoom(self, factor: float):
        self.set_zoom(self.current_zoom() * factor)

    def center_image(self):
        self.view.setSceneRect(QtCore.QRectF(0, 0, self._img_W, self._img_H))
        self.view.centerOn(self._img_W / 2.0, self._img_H / 2.0)

    # ---- Lat/Lon -> Scene koordinat dönüşümü ----
    def _lon_to_x(self, lon_deg: float) -> float:
        return self._R * math.radians(lon_deg)

    def _lat_to_y(self, lat_deg: float) -> float:
        # WebMercator (EPSG:3857)
        lat_rad = math.radians(lat_deg)
        return self._R * math.log(math.tan(math.pi/4.0 + lat_rad/2.0))

    def _scene_pos_from_latlon(self, lat: float, lon: float) -> QtCore.QPointF | None:
        # Dışarıda kalanı çizme
        if not (self.south <= lat <= self.north and self.west <= lon <= self.east):
            # Yine de harita dışında olsa bile piksele dönüştürmek isteyebilirsin.
            # Burada clamp'leyelim ki kenarda görünsün.
            lat = min(max(lat, self.south), self.north)
            lon = min(max(lon, self.west), self.east)

        x_left  = self._lon_to_x(self.west)
        x_right = self._lon_to_x(self.east)
        y_top   = self._lat_to_y(self.north)
        y_bot   = self._lat_to_y(self.south)

        x = self._lon_to_x(lon)
        y = self._lat_to_y(lat)

        # Pixmap boyutu
        W = float(self._img_W)
        H = float(self._img_H)

        # WebMercator yukarı pozitif; QGraphics'te y aşağı pozitif.
        # Bu nedenle y'yi ters çeviriyoruz.
        px = (x - x_left) / (x_right - x_left) * W
        py = (y_top - y)   / (y_top - y_bot)  * H

        return QtCore.QPointF(px, py)

    # ---- Geo helpers ----
    def _offset_latlon(self, lat_deg: float, lon_deg: float, north_m: float, east_m: float) -> tuple[float, float]:
        """Küçük ofset yaklaşıklığı. north_m: +Kuzey, east_m: +Doğu (metre)."""
        R = 6378137.0
        d_lat = (north_m / R) * (180.0 / math.pi)
        d_lon = (east_m / (R * math.cos(math.radians(lat_deg)))) * (180.0 / math.pi)
        return lat_deg + d_lat, lon_deg + d_lon

    def _end_latlon_from_heading(self, lat: float, lon: float, heading_deg: float, length_m: float) -> tuple[
        float, float]:
        """0° = True North, saat yönü (+). length_m kadar ilerleyince uç lat/lon."""
        theta = math.radians(heading_deg if heading_deg is not None else 0.0)
        north_m = length_m * math.cos(theta)
        east_m = length_m * math.sin(theta)
        return self._offset_latlon(lat, lon, north_m, east_m)

    def _update_heading_items(self):
        lat = self._last_lat;
        lon = self._last_lon
        if lat is None or lon is None:
            self._real_line.setVisible(False)
            self._real_arrow.setVisible(False)
            self._targ_line.setVisible(False)
            return

        # --- Real heading (3 m + ok) ---
        if self._real_hdg_deg is not None:
            lat2, lon2 = self._end_latlon_from_heading(lat, lon, self._real_hdg_deg, 3.0)
            p1 = self._scene_pos_from_latlon(lat, lon)
            p2 = self._scene_pos_from_latlon(lat2, lon2)
            self._real_line.setLine(p1.x(), p1.y(), p2.x(), p2.y())
            self._real_line.setVisible(True)

            # Ok başı: uca yakın V şekli (±25°)
            arrow_len = 0.8  # m
            left_dir = (self._real_hdg_deg + 180 - 25) % 360
            right_dir = (self._real_hdg_deg + 180 + 25) % 360
            latL, lonL = self._end_latlon_from_heading(lat2, lon2, left_dir, arrow_len)
            latR, lonR = self._end_latlon_from_heading(lat2, lon2, right_dir, arrow_len)
            pL = self._scene_pos_from_latlon(latL, lonL)
            pR = self._scene_pos_from_latlon(latR, lonR)

            path = QtGui.QPainterPath()
            path.moveTo(pL)
            path.lineTo(p2)
            path.lineTo(pR)
            self._real_arrow.setPath(path)
            self._real_arrow.setVisible(True)
        else:
            self._real_line.setVisible(False)
            self._real_arrow.setVisible(False)

        # --- Target heading (5 m, ok yok) ---
        if self._targ_hdg_deg is not None:
            lat3, lon3 = self._end_latlon_from_heading(lat, lon, self._targ_hdg_deg, 5.0)
            p1 = self._scene_pos_from_latlon(lat, lon)
            p3 = self._scene_pos_from_latlon(lat3, lon3)
            self._targ_line.setLine(p1.x(), p1.y(), p3.x(), p3.y())
            self._targ_line.setVisible(True)
        else:
            self._targ_line.setVisible(False)

    # ---- Public setters ----
    def set_real_heading(self, hdg_deg: float | None):
        try:
            self._real_hdg_deg = float(hdg_deg) if hdg_deg is not None else None
        except Exception:
            self._real_hdg_deg = None
        self._update_heading_items()

    def set_target_heading(self, hdg_deg: float | None):
        try:
            self._targ_hdg_deg = float(hdg_deg) if hdg_deg is not None else None
        except Exception:
            self._targ_hdg_deg = None
        self._update_heading_items()

    def _prune_and_redraw_trail(self, boat_id):
        """İzi 500 metre ile sınırlar ve haritayı yeniden çizer."""
        trail = self.trail_points_1 if boat_id == 1 else self.trail_points_2
        if len(trail) < 2:
            return

        max_dist_m = 400.0  # MAKSİMUM İZ UZUNLUĞU (METRE)
        total_dist = 0.0
        new_trail = [trail[-1]]

        # Sondan başa doğru (en yeni noktadan en eskiye) mesafeleri topla
        for i in range(len(trail) - 2, -1, -1):
            d = haversine_distance(trail[i][0], trail[i][1], trail[i + 1][0], trail[i + 1][1])
            total_dist += d
            new_trail.append(trail[i])
            # Eğer toplam uzunluk 500 metreyi geçerse daha eski noktaları alma!
            if total_dist >= max_dist_m:
                break

        new_trail.reverse()  # Listeyi tekrar kronolojik sıraya çevir

        if boat_id == 1:
            self.trail_points_1 = new_trail
        else:
            self.trail_points_2 = new_trail

        # QPainterPath'i yeni ve kırpılmış noktalarla baştan çiz
        painter_path = QtGui.QPainterPath()
        first = True
        for lat, lon in new_trail:
            p = self._scene_pos_from_latlon(lat, lon)
            if p:
                if first:
                    painter_path.moveTo(p)
                    first = False
                else:
                    painter_path.lineTo(p)

        # Çizgiyi haritaya uygula
        if boat_id == 1:
            self.path_path = painter_path
            self.path_item.setPath(self.path_path)
        else:
            self.path_path_2 = painter_path
            self.path_item_2.setPath(self.path_path_2)

    def update_current_position(self, lat: float, lon: float, boat_id=1, draw_tail=True):
        # 1. EMA Filtresi uygula (Low-Pass Filter) - Zikzakları engeller
        alpha = 0.3  # Yumuşatma faktörü (0 ile 1 arası, 1 = filtresiz, düşük değer = yumuşak ama gecikmeli)
        dist_threshold_m = 50.0  # 50 metreden büyük atlamalarda filtreyi sıfırla (Işınlanma/ilk açılış)

        if boat_id == 1:
            if getattr(self, '_target_lat_1', None) is None:
                # İlk konum geldi
                self._target_lat_1, self._target_lon_1 = lat, lon
                self._visual_lat_1, self._visual_lon_1 = lat, lon
            else:
                dist = haversine_distance(self._target_lat_1, self._target_lon_1, lat, lon)
                if dist > dist_threshold_m:
                    # Büyük atlama, direkt zıpla
                    self._target_lat_1, self._target_lon_1 = lat, lon
                    self._visual_lat_1, self._visual_lon_1 = lat, lon
                else:
                    # EMA yumuşatma
                    self._target_lat_1 = (alpha * lat) + ((1.0 - alpha) * self._target_lat_1)
                    self._target_lon_1 = (alpha * lon) + ((1.0 - alpha) * self._target_lon_1)

            self._last_lat, self._last_lon = self._target_lat_1, self._target_lon_1
            self._update_heading_items()
            self.update_range_rings()

        elif boat_id == 2:
            if getattr(self, '_target_lat_2', None) is None:
                self._target_lat_2, self._target_lon_2 = lat, lon
                self._visual_lat_2, self._visual_lon_2 = lat, lon
            else:
                dist = haversine_distance(self._target_lat_2, self._target_lon_2, lat, lon)
                if dist > dist_threshold_m:
                    self._target_lat_2, self._target_lon_2 = lat, lon
                    self._visual_lat_2, self._visual_lon_2 = lat, lon
                else:
                    self._target_lat_2 = (alpha * lat) + ((1.0 - alpha) * self._target_lat_2)
                    self._target_lon_2 = (alpha * lon) + ((1.0 - alpha) * self._target_lon_2)

    def _animate_markers(self):
        """Timer ile saniyede 30 kez çağrılır. Görsel marker'ı hedef EMA koordinatına yumuşakça kaydırır (Tweening)."""
        interp_factor = 0.2  # Her animasyon karesinde hedefe %20 yaklaş

        # Performans için sayaçları tanımla (eğer yoksa)
        if not hasattr(self, 'prune_counter_1'): self.prune_counter_1 = 0
        if not hasattr(self, 'prune_counter_2'): self.prune_counter_2 = 0

        # --- IDA 1 ---
        if getattr(self, '_visual_lat_1', None) is not None and getattr(self, '_target_lat_1', None) is not None:
            # İnterpolasyon
            self._visual_lat_1 += (self._target_lat_1 - self._visual_lat_1) * interp_factor
            self._visual_lon_1 += (self._target_lon_1 - self._visual_lon_1) * interp_factor

            p = self._scene_pos_from_latlon(self._visual_lat_1, self._visual_lon_1)
            if p is not None:
                self.curr_marker.setPos(p)
                self.curr_marker.setVisible(True)

                # İz (Trail) Çizimi - Artık yumuşatılmış görsel koordinatlarla çiziliyor
                draw_tail = True  # İstenirse dışarıdan parametre alınabilir
                if draw_tail:
                    if not self.trail_points_1 or self.trail_points_1[-1] != (self._visual_lat_1, self._visual_lon_1):
                        self.trail_points_1.append((self._visual_lat_1, self._visual_lon_1))

                        if self.path_path.isEmpty():
                            self.path_path.moveTo(p)
                        else:
                            self.path_path.lineTo(p)
                        self.path_item.setPath(self.path_path)

                        self.prune_counter_1 += 1
                        if self.prune_counter_1 > 50:
                            self._prune_and_redraw_trail(1)
                            self.prune_counter_1 = 0

        # --- IDA 2 ---
        if getattr(self, '_visual_lat_2', None) is not None and getattr(self, '_target_lat_2', None) is not None and hasattr(self, 'curr_marker_2'):
            self._visual_lat_2 += (self._target_lat_2 - self._visual_lat_2) * interp_factor
            self._visual_lon_2 += (self._target_lon_2 - self._visual_lon_2) * interp_factor

            p2 = self._scene_pos_from_latlon(self._visual_lat_2, self._visual_lon_2)
            if p2 is not None:
                self.curr_marker_2.setPos(p2)
                self.curr_marker_2.setVisible(True)

                draw_tail = True
                if draw_tail and hasattr(self, 'path_path_2'):
                    if not self.trail_points_2 or self.trail_points_2[-1] != (self._visual_lat_2, self._visual_lon_2):
                        self.trail_points_2.append((self._visual_lat_2, self._visual_lon_2))

                        if self.path_path_2.isEmpty():
                            self.path_path_2.moveTo(p2)
                        else:
                            self.path_path_2.lineTo(p2)
                        self.path_item_2.setPath(self.path_path_2)

                        self.prune_counter_2 += 1
                        if self.prune_counter_2 > 50:
                            self._prune_and_redraw_trail(2)
                            self.prune_counter_2 = 0

    def clear_ghost_trail_by_id(self, boat_id):
        """İlgili tekneye ait canlı izi (ve varsa hayalet izi) tamamen sıfırlar."""
        if boat_id == 1:
            # Hayalet izi sil
            self.ghost_path = QtGui.QPainterPath()
            self.ghost_item.setPath(self.ghost_path)

            # Canlı İzi (Kırmızı) Sil
            self.trail_points_1 = []
            self.path_path = QtGui.QPainterPath()
            self.path_item.setPath(self.path_path)
        elif boat_id == 2:
            # 2. IDA'nın Canlı İzini (Mavi) Sil
            self.trail_points_2 = []
            self.path_path_2 = QtGui.QPainterPath()
            self.path_item_2.setPath(self.path_path_2)

    def update_mission_waypoints(self, boat_id, local_points, remote_points=[], tags=None):
        # 1. Gelen veriyi hafızaya al
        if boat_id == 1:
            self.cached_pts_1 = local_points
        else:
            self.cached_pts_2 = local_points

        other_list = self.cached_pts_2 if boat_id == 1 else self.cached_pts_1

        # Grupları hazırla
        if not hasattr(self, 'mission_grp_1'):
            self.mission_grp_1 = QtWidgets.QGraphicsItemGroup()
            self.scene.addItem(self.mission_grp_1)
            self.mission_grp_1.setZValue(25)
        if not hasattr(self, 'mission_grp_2'):
            self.mission_grp_2 = QtWidgets.QGraphicsItemGroup()
            self.scene.addItem(self.mission_grp_2)
            self.mission_grp_2.setZValue(25)

        grp = self.mission_grp_1 if boat_id == 1 else self.mission_grp_2

        # Grubu temizle (Eski çizimleri sil)
        for child in grp.childItems():
            self.scene.removeItem(child)

        # Renkler
        if boat_id == 1:
            def_color = QtGui.QColor("#ff1744")  # Kırmızı
            text_color = QtGui.QColor("#ff8a80")  # Açık Kırmızı (Yazı)
            y_offset_factor = -1  # YUKARIYA YAZ
        else:
            def_color = QtGui.QColor("#2979ff")  # Mavi
            text_color = QtGui.QColor("#80d8ff")  # Açık Mavi (Yazı)
            y_offset_factor = 1  # AŞAĞIYA YAZ

        for i, (lat, lon) in enumerate(local_points):
            p = self._scene_pos_from_latlon(lat, lon)
            if not p: continue

            # --- Durum Kontrolü (Teyitli/Ortak) ---
            is_verified = False
            for (r_lat, r_lon) in remote_points:
                if abs(lat - r_lat) < 0.000001 and abs(lon - r_lon) < 0.000001:
                    is_verified = True
                    break

            is_shared = False
            if not is_verified:
                for (o_lat, o_lon) in other_list:
                    if abs(lat - o_lat) < 0.000001 and abs(lon - o_lon) < 0.000001:
                        is_shared = True
                        break

            # Nokta Rengi
            if is_verified:
                color = QtGui.QColor("#00e676")  # Yeşil (Teyitli)
            elif is_shared:
                color = QtGui.QColor("#ff9100")  # Turuncu (Çakışma)
            else:
                color = def_color

            # --- 1. Nokta Çizimi (Dot) ---
            r = 1.5
            dot = QtWidgets.QGraphicsEllipseItem(p.x() - r, p.y() - r, 2 * r, 2 * r)
            dot.setBrush(QtGui.QBrush(color))
            dot.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
            grp.addToGroup(dot)

            # --- 2. Etiket (Text) Hazırlığı ---
            label_text = str(i + 1)
            # Görev ismini kısalt (TASK1_START -> T1_START) yer kazanmak için
            if tags and i < len(tags):
                short_tag = tags[i].replace("TASK", "T").replace("_", " ")
                label_text += f" {short_tag}"

            txt = QtWidgets.QGraphicsSimpleTextItem(label_text)
            f = txt.font()
            f.setPixelSize(4)  # Okunaklı boyut
            f.setBold(True)
            txt.setFont(f)
            txt.setBrush(QtGui.QBrush(text_color))

            # --- 3. Pozisyon Ayarlama (ÇAKIŞMA ÖNLEME) ---
            br = txt.boundingRect()

            # X Ekseni: Yazıyı noktanın biraz sağına al
            txt_x = p.x() + 6

            # Y Ekseni: ID'ye göre yukarı veya aşağı al
            if boat_id == 1:
                # IDA 1: Noktanın YUKARISINA (Height kadar yukarı + boşluk)
                txt_y = p.y() - br.height() - 4
            else:
                # IDA 2: Noktanın AŞAĞISINA (Direkt aşağı + boşluk)
                txt_y = p.y() + 4

            txt.setPos(txt_x, txt_y)

            # --- 4. Arka Plan Kutucuğu (Okunabilirlik İçin) ---
            # Yazının arkasına yarı saydam siyah kutu koyuyoruz
            bg_rect = QtWidgets.QGraphicsRectItem(
                txt_x - 2, txt_y - 1,
                br.width() + 4, br.height() + 2
            )
            bg_rect.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 30)))  # %60 Siyah
            bg_rect.setPen(Qt.NoPen)

            grp.addToGroup(bg_rect)  # Önce kutuyu ekle (altta kalsın)
            grp.addToGroup(txt)  # Sonra yazıyı ekle (üstte olsun)

            # İsteğe bağlı: Kutudan noktaya ince bir çizgi (Leader Line)
            line = QtWidgets.QGraphicsLineItem(p.x(), p.y(), txt_x, txt_y + br.height() / 2)
            line.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 100), 1))
            grp.addToGroup(line)

    def set_waypoints(self, gps_dict: dict):
        """gps_dict beklenen biçim: {"GPS1":{"lat":..,"lon":..}, ...}"""
        if not isinstance(gps_dict, dict):
            return

        # Eski öğeleri temizle
        try:
            for item in list(self.wp_group.childItems()):
                self.scene.removeItem(item)
            # Grubu yeniden yarat (bazı Qt sürümlerinde childItems temizlenince grup 'tuhaf' davranabiliyor)
            self.scene.removeItem(self.wp_group)
        except Exception:
            pass

        self.wp_group = QtWidgets.QGraphicsItemGroup()
        self.wp_group.setZValue(15)
        self.scene.addItem(self.wp_group)

        # Boyut/sabitler (yoksa varsayılanları kullan)
        r_px = globals().get("WP_RADIUS_PX", 4.0)
        font_pt = globals().get("WP_FONT_PT", 8.5)
        keep_px = globals().get("KEEP_PIXEL_SIZE", True)

        # GPS1..GPS5 sırayla çiz
        for i in range(1, 6):
            key = f"GPS{i}"
            entry = gps_dict.get(key)
            if not isinstance(entry, dict):
                continue
            try:
                lat = float(entry.get("lat"))
                lon = float(entry.get("lon"))
            except Exception:
                continue
            if lat is None or lon is None:
                continue

            p = self._scene_pos_from_latlon(lat, lon)
            if p is None:
                continue

            # --- Daire marker ---
            circ = QtWidgets.QGraphicsEllipseItem(-r_px, -r_px, 2 * r_px, 2 * r_px)
            circ.setBrush(QtGui.QBrush(QtGui.QColor("#1e88e5")))
            circ.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.0))
            circ.setPos(p)
            circ.setZValue(16)
            if keep_px:
                circ.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
            self.wp_group.addToGroup(circ)

            # --- Etiket ---
            label = QtWidgets.QGraphicsTextItem(key)
            # Yazı tipi ve stil
            f = label.font()
            f.setPointSizeF(font_pt)
            f.setBold(True)
            label.setFont(f)

            # Renk: turkuaz
            label.setDefaultTextColor(QtGui.QColor("#90EE90")) #000000 #30d5c8

            # Metni DAİRENİN MERKEZİNE yerleştir
            br = label.boundingRect()  # font set edildikten sonra ölç
            label.setPos(p - QtCore.QPointF(br.width() / 2.0, br.height() / 2.0))

            label.setZValue(17)
            if keep_px:
                label.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)

            self.wp_group.addToGroup(label)


# -------------------- Main Window --------------------
# -------------------- Main Window --------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DHO GÖKMAVİ - YER KONTROL")
        self.showMaximized()

        self.MISSION_TAGS_1 = [
            "TASK1_START",
            "TASK1_MID",
            "TASK1_END",
            "TASK2_START",
            "TASK2_MID",
            "TASK2_END",
            "TASK3_START",
            "TASK3_MID",
            "TASK3_RIGHT",
            "TASK3_END",
            "TASK3_LEFT",
            "DOCKING"
        ]


        # Stil Dosyası (GÜÇLENDİRİLMİŞ & PARLAK BEYAZ)
        self.setStyleSheet("""
                    QMainWindow {
                        background-color: #2b2b2b;
                        color: #e0e0e0;
                    }

                    /* --- GÜÇLENDİRİLMİŞ BUTONLAR --- */
                    /* Daha spesifik seçici kullanarak stilin uygulanmasını garantiye alıyoruz */
                    QWidget QPushButton {
                        background-color: #37474f;       /* Koyu Zemin */
                        color: #eceff1;                  /* Açık Gri/Beyaz Yazı */
                        border: 2px solid #546e7a;       /* DAHA KALIN (2px) BELİRGİN ÇERÇEVE */
                        border-radius: 4px;
                        padding: 4px;
                        font-weight: bold;
                        font-size: 11px;
                    }
                    /* Hover Efekti - PARLASIN */
                    QWidget QPushButton:hover {
                        background-color: #455a64;
                        border: 2px solid #81d4fa;       /* Çerçeve Parlak Açık Mavi */
                        color: #ffffff;                  /* Yazı Tam Beyaz */
                    }
                    /* Pressed Efekti - GÖMÜLSÜN */
                    QWidget QPushButton:pressed {
                        background-color: #263238;
                        border: 2px solid #263238;
                        padding-top: 6px;                /* Tıklama hissi için kaydır */
                        padding-left: 6px;
                    }
                    /* Seçili (Checked) Efekti */
                    QWidget QPushButton:checked {
                        background-color: #00897b;
                        border: 2px solid #00e676;
                        color: white;
                    }

                    /* --- GÖREV TABLOSU (PARLAK BEYAZ) --- */
                    QTableWidget {
                        background-color: #323639;       /* Koyu Zemin */
                        color: #ffffff;                  /* PARLAK BEYAZ ve KALIN YAZI */
                        font-weight: bold;
                        gridline-color: #546e7a;         /* Izgara Çizgileri */
                        border: none;                    /* Dış çerçeve yok (Modern çerçeveye uyum) */
                    }
                    /* Tablo Başlıkları */
                    QHeaderView::section {
                        background-color: #37474f;
                        color: #eceff1;
                        padding: 4px;
                        border: 1px solid #546e7a;
                        font-weight: bold;
                    }
                    /* Tablo Seçili Satır */
                    QTableWidget::item:selected {
                        background-color: #00e676;       /* Seçili satır parlak yeşil olsun */
                        color: #000000;                  /* Üzerindeki yazı siyah olsun ki okunsun */
                    }

                    /* --- DİĞER BİLEŞENLER --- */
                    QGroupBox {
                        border: 2px solid #444;          /* Çerçeveyi belirginleştir */
                        border-radius: 6px;
                        margin-top: 12px;
                        background-color: #323639;
                        font-weight: bold;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 5px;
                        color: #4fc3f7;
                        background-color: #323639;       /* Başlık arka planı */
                    }
                    QCheckBox {
                        spacing: 8px;
                        color: #b0bec5;
                        font-weight: bold;
                    }
                    QCheckBox::indicator {
                        width: 16px;                     /* Kutucuğu biraz büyüt */
                        height: 16px;
                        border: 2px solid #546e7a;
                        background: #263238;
                        border-radius: 3px;
                    }
                    QCheckBox::indicator:checked {
                        background-color: #00e676;
                        border-color: #00e676;
                    }
                    QLineEdit, QSpinBox, QComboBox {
                        background-color: #263238;
                        color: #ffffff;                  /* Girdiler de Parlak Beyaz olsun */
                        border: 2px solid #546e7a;
                        padding: 4px;
                        border-radius: 3px;
                        font-weight: bold;
                    }
                    QSplitter::handle {
                        background-color: #1c1c1c;
                    }
                    QLabel {
                        color: #cfd8dc;
                    }
                    /* Tooltip */
                    QToolTip {
                        background-color: #263238;
                        color: #00e676;
                        border: 2px solid #546e7a;
                        padding: 4px;
                        opacity: 230;
                        font-weight: bold;
                    }
                """)
        # Arka plan resmi yolu (background.jpg veya background.png dosyanın adı neyse onu yaz)
        bg_path = res_path("background.png").replace("\\", "/")

        # --- ANA YAPI KURULUMU ---
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        # Ana Yerleşim (Layout)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)  # Kenar boşluğu yok, tam ekran
        root.setSpacing(0)

        # Halat Kalınlığı (Burada tanımlıyoruz ki aşağıda hata vermesin)
        ROPE_SIZE = 12
        self._rope_thk = ROPE_SIZE

        # İçerik Çerçevesi (Tüm arayüz bunun içinde olacak)
        self.content = QtWidgets.QFrame(central)
        root.addWidget(self.content)

        # İçerik Layout'u (3 Sütunlu Splitter buraya gelecek)
        content_lay = QtWidgets.QHBoxLayout(self.content)
        # İçerik halatın altında kalmasın diye kenarlardan halat kalınlığı kadar boşluk bırakıyoruz
        content_lay.setContentsMargins(ROPE_SIZE, ROPE_SIZE, ROPE_SIZE, ROPE_SIZE)
        content_lay.setSpacing(0)

        # Halat Süsleri (Overlay olarak content üzerine ekleniyor)
        # Resim yolunu kontrol et, yoksa gri çizer
        rope_path = res_path("halat.png").replace("\\", "/")

        self.rope_top = RopeStripe("top", rope_path, ROPE_SIZE, parent=self.content)
        self.rope_bottom = RopeStripe("bottom", rope_path, ROPE_SIZE, parent=self.content)
        self.rope_left = RopeStripe("left", rope_path, ROPE_SIZE, parent=self.content)
        self.rope_right = RopeStripe("right", rope_path, ROPE_SIZE, parent=self.content)

        # Pencere boyutu değişince halatları yeniden hizalamak için filtre
        self.content.installEventFilter(self)

        # --- SPLITTER (SOL | ORTA | SAĞ) ---
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        content_lay.addWidget(splitter)

        # =========================================================
        # SOL PANEL: IDA 1
        # =========================================================
        left_widget = QtWidgets.QWidget()
        left_widget.setObjectName("panelIDA1")  # Özel isim veriyoruz
        left_widget.setAttribute(QtCore.Qt.WA_StyledBackground, True)  # Arka plan boyamayı etkinleştir
        # Hem arka planı hem de kırmızı ayracı aynı anda uyguluyoruz:
        left_widget.setStyleSheet(f"""
                    #panelIDA1 {{
                        border-image: url("{bg_path}") 0 0 0 0 stretch stretch;
                        border-right: 2px solid #b71c1c; /* Kırmızı ayraç */
                    }}
                """)
        # -------------------------------------

        v1 = QtWidgets.QVBoxLayout(left_widget)
        v1.setContentsMargins(4, 4, 4, 4)  # Panelin genel kenar boşluğunu kıstık
        v1.setSpacing(5)  # Elemanlar (Başlık, Telemetri, Tablo) birbirine yaklaşsın

        # Başlık ve LED
        h1 = QtWidgets.QHBoxLayout()
        h1.setContentsMargins(0, 0, 0, 0)  # Başlığın dış boşluğunu sıfırla!

        lbl_id1 = QtWidgets.QLabel("IDA 1")
        # Arka plan: Koyu Gri (%60 Görünür), Yazı: Kırmızı
        lbl_id1.setStyleSheet("""
                    color: #ff1744;
                    font-weight: bold;
                    font-size: 12pt;
                    background-color: rgba(40, 40, 40, 160);
                    padding: 2px 6px; /* Üst-Alt: 2px, Sağ-Sol: 6px */
                    border-radius: 4px;
                    border: 1px solid #ff1744; /* İstersen ince bir çerçeve de ekler */
                """)
        # Maksimum yükseklik vererek uzamasını engelle
        lbl_id1.setFixedHeight(30)
        h1.addWidget(lbl_id1)

        self.usv1_led = QtWidgets.QLabel()
        self.usv1_led.setFixedSize(16,16)  # LED'i de biraz kibarlaştırdık (20->16)
        self.usv1_led.setStyleSheet("background-color:#37474f; border-radius:8px;")
        h1.addWidget(self.usv1_led, alignment=Qt.AlignRight)
        v1.addLayout(h1)

        # Telemetri 1
        self.usv1_ui = self._create_telemetry_grid(v1)

        # Görev Planı 1
        self.usv1_table = self._create_mission_box(v1, 1)

        # --- MESAFE HALKALARI (MODERN) ---
        l_rng = self._add_modern_frame(v1, "RADAR HALKALARI")
        r1 = QtWidgets.QHBoxLayout()
        self.chk_rings = QtWidgets.QCheckBox("Aç")
        self.chk_rings.setStyleSheet("border:none;")  # Kenarlığı kaldır
        self.chk_rings.toggled.connect(self._toggle_rings)

        self.sp_ring_step = QtWidgets.QSpinBox();
        self.sp_ring_step.setValue(5)
        self.sp_ring_count = QtWidgets.QSpinBox();
        self.sp_ring_count.setValue(3)
        self.sp_ring_step.valueChanged.connect(self._update_ring_settings)
        self.sp_ring_count.valueChanged.connect(self._update_ring_settings)

        r1.addWidget(self.chk_rings)
        r1.addWidget(QtWidgets.QLabel("Adım(m):"));
        r1.addWidget(self.sp_ring_step)
        r1.addWidget(QtWidgets.QLabel("Adet:"));
        r1.addWidget(self.sp_ring_count)
        l_rng.addLayout(r1)

        # Kontroller 1
        self._create_controls(v1, 1)

        # --- LOG 1 (MODERN) ---
        l_log = self._add_modern_frame(v1, "SİSTEM GÜNLÜĞÜ")
        self.usv1_log = QtWidgets.QPlainTextEdit()
        self.usv1_log.setReadOnly(True)
        # Log için özel stil (Monospace font, siyah zemin)
        self.usv1_log.setStyleSheet("""
                    QPlainTextEdit {
                        background-color: black;
                        color: #00e676;
                        font-family: Consolas, monospace;
                        font-size: 9pt;
                        border: none;
                    }
                """)
        l_log.addWidget(self.usv1_log)

        # Scroll 1
        scroll1 = QtWidgets.QScrollArea();
        scroll1.setWidget(left_widget);
        scroll1.setWidgetResizable(True)
        scroll1.setFixedWidth(500)
        splitter.addWidget(scroll1)

        # =========================================================
        # ORTA PANEL: HARİTA & ORTAK
        # =========================================================
        center_widget = QtWidgets.QWidget()
        vc = QtWidgets.QVBoxLayout(center_widget)

        # 1. Ortak Üst (Harita, Com, Connect)
        top_grp = QtWidgets.QGroupBox("Sistem Ayarları")
        vc.addWidget(top_grp);
        top_lay = QtWidgets.QHBoxLayout(top_grp)

        self.MAP_PRESETS = {
            "RoboBoat (Florida)": {"file": "roboboat.jpg", "n": 27.381, "s": 27.356, "w": -82.455, "e": -82.445},
            "Okul (Tuzla)": {"file": "okul.jpg", "n": 40.810072, "s": 40.808099, "w": 29.261345, "e": 29.265630}
        }
        self.cmb_maps = QtWidgets.QComboBox();
        self.cmb_maps.addItems(self.MAP_PRESETS.keys())
        self.cmb_maps.currentIndexChanged.connect(self._on_map_changed)
        top_lay.addWidget(QtWidgets.QLabel("Harita:"));
        top_lay.addWidget(self.cmb_maps)

        self.cmb_port_1 = QtWidgets.QComboBox()
        self._refresh_ports()
        top_lay.addWidget(QtWidgets.QLabel("IDA Port:"))
        top_lay.addWidget(self.cmb_port_1)

        self.cmb_baud = QtWidgets.QComboBox();
        self.cmb_baud.addItems(["57600", "115200"]);
        self.cmb_baud.setCurrentText("57600")
        top_lay.addWidget(self.cmb_baud)
        self.chk_csv = QtWidgets.QCheckBox("CSV");
        top_lay.addWidget(self.chk_csv)

        btn_ref = QtWidgets.QPushButton("R");
        btn_ref.setFixedWidth(30);
        btn_ref.clicked.connect(self._refresh_ports)
        top_lay.addWidget(btn_ref)
        self.btn_connect = QtWidgets.QPushButton("BAĞLAN");
        self.btn_connect.clicked.connect(self._toggle)
        top_lay.addWidget(self.btn_connect)

        # 2. Harita
        # Varsayılan harita ayarı (Combobox'tan al)
        current_map_key = self.cmb_maps.currentText()
        if current_map_key in self.MAP_PRESETS:
            first_map = self.MAP_PRESETS[current_map_key]
            self.map = MapPanel(res_path(first_map["file"]), first_map["n"], first_map["s"], first_map["w"],
                                first_map["e"])
        else:
            # Fallback
            self.map = MapPanel("roboboat.jpg", 27.381, 27.356, -82.455, -82.445)

        self.map.gps_selected.connect(self._on_map_gps_selected)
        vc.addWidget(self.map)

        # 3. Ortak Alt (Geofence, Jüri, Parkur)
        bot_lay = QtWidgets.QHBoxLayout();
        vc.addLayout(bot_lay)

        # Geofence
        geo_box = QtWidgets.QGroupBox("Sanal Çit");
        gl = QtWidgets.QGridLayout(geo_box)
        self.btn_fence_draw = QtWidgets.QPushButton("Çiz");
        self.btn_fence_draw.setCheckable(True);
        self.btn_fence_draw.clicked.connect(self._toggle_fence_draw)
        self.btn_fence_clear = QtWidgets.QPushButton("Sil");
        self.btn_fence_clear.clicked.connect(lambda: self.map.clear_fence())
        self.chk_fence_active = QtWidgets.QCheckBox("AKTİF");
        self.chk_fence_active.setStyleSheet("color:red; font-weight:bold;")
        gl.addWidget(self.btn_fence_draw, 0, 0);
        gl.addWidget(self.btn_fence_clear, 0, 1);
        gl.addWidget(self.chk_fence_active, 1, 0, 1, 2)
        bot_lay.addWidget(geo_box)

        # Drone (İHA) Alanı
        iha_box = QtWidgets.QGroupBox("İHA (Drone)")
        iha_l = QtWidgets.QVBoxLayout(iha_box)

        # İHA Bağlantı Durumu ve LED
        h_iha_conn = QtWidgets.QHBoxLayout()
        lbl_iha_title = QtWidgets.QLabel("Durum:")
        lbl_iha_title.setStyleSheet("color: #cfd8dc; font-weight: bold; font-size: 11px;")
        self.iha_led = QtWidgets.QLabel()
        self.iha_led.setFixedSize(16, 16)
        self.iha_led.setStyleSheet("background-color:#37474f; border-radius:8px;")
        h_iha_conn.addWidget(lbl_iha_title)
        h_iha_conn.addWidget(self.iha_led)
        h_iha_conn.addStretch()

        # Drone Aktif Checkbox
        self.chk_drone_active = QtWidgets.QCheckBox("Drone Dinle")
        self.chk_drone_active.setStyleSheet("color:#00e5ff; font-weight:bold;")
        self.chk_drone_active.setChecked(getattr(cfg, 'DRONE_ACTIVE', False))
        self.chk_drone_active.stateChanged.connect(self._on_drone_active_changed)

        # Hedef Renk Gösterimi
        h_iha_color = QtWidgets.QHBoxLayout()
        lbl_color_title = QtWidgets.QLabel("Hedef Renk:")
        lbl_color_title.setStyleSheet("color: #cfd8dc; font-weight: bold; font-size: 11px;")

        default_color = getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red')
        tr_default = default_color.upper()
        if default_color == 'red': tr_default = "KIRMIZI"
        elif default_color == 'green': tr_default = "YESIL"
        elif default_color == 'black': tr_default = "SIYAH"

        self.lbl_iha_color = QtWidgets.QLabel(f"Varsayılan: {tr_default}")
        self.lbl_iha_color.setStyleSheet("""
            color: #00e676;
            font-weight: bold;
            font-size: 12px;
            background-color: rgba(0, 0, 0, 40);
            border-radius: 3px;
            padding: 2px;
        """)
        h_iha_color.addWidget(lbl_color_title)
        h_iha_color.addWidget(self.lbl_iha_color)
        h_iha_color.addStretch()

        iha_l.addLayout(h_iha_conn)
        iha_l.addWidget(self.chk_drone_active)
        iha_l.addLayout(h_iha_color)
        iha_l.addStretch()
        bot_lay.addWidget(iha_box)

        # Parkur Izgaraları (A-B-C)
        self.course_ui = {}
        course_grp = QtWidgets.QGroupBox("Parkur");
        cl = QtWidgets.QVBoxLayout(course_grp)
        self.tabs_course = QtWidgets.QTabWidget();
        self.tabs_course.setFixedHeight(120)
        course_grp.setMaximumWidth(400)
        for name in ["A", "B", "C"]:
            w = QtWidgets.QWidget()
            v_tab = QtWidgets.QVBoxLayout(w)
            v_tab.setContentsMargins(4, 4, 4, 4)

            # Üst Satır
            h_top = QtWidgets.QHBoxLayout()
            b_sel = QtWidgets.QPushButton("Seç (4 Köşe)")
            b_sel.clicked.connect(lambda _, n=name: self._start_course_marking(n))

            sr = QtWidgets.QSpinBox();
            sr.setValue(4)
            sc = QtWidgets.QSpinBox();
            sc.setValue(4)

            b_upd = QtWidgets.QPushButton("Güncelle")
            b_upd.clicked.connect(lambda _, n=name: self._update_grid_params(n))

            h_top.addWidget(b_sel)

            # --- YENİ ETİKETLER (Siyah ve Okunaklı) ---
            l_r = QtWidgets.QLabel("Satır:")
            l_r.setStyleSheet(
                "color: black; font-weight: bold; background-color: #b0bec5; padding: 2px; border-radius: 3px;")
            h_top.addWidget(l_r)
            h_top.addWidget(sr)

            l_c = QtWidgets.QLabel("Sütun:")
            l_c.setStyleSheet(
                "color: black; font-weight: bold; background-color: #b0bec5; padding: 2px; border-radius: 3px;")
            h_top.addWidget(l_c)
            h_top.addWidget(sc)

            h_top.addWidget(b_upd)
            v_tab.addLayout(h_top)

            # Alt Satır (Kaydet/Yükle/Sil - Aynı kalıyor)
            h_bot = QtWidgets.QHBoxLayout()
            b_save = QtWidgets.QPushButton(f"💾 {name} Kaydet");
            b_save.clicked.connect(lambda _, n=name: self._save_single_course(n))
            b_load = QtWidgets.QPushButton(f"📂 {name} Yükle");
            b_load.clicked.connect(lambda _, n=name: self._load_single_course(n))
            b_del = QtWidgets.QPushButton(f"🗑️ {name} Sil");
            b_del.setStyleSheet("color: #ff5252; font-weight: bold;");
            b_del.clicked.connect(lambda _, n=name: self._clear_single_course(n))
            h_bot.addWidget(b_save);
            h_bot.addWidget(b_load);
            h_bot.addWidget(b_del)
            v_tab.addLayout(h_bot)

            self.tabs_course.addTab(w, f"Parkur {name}")

            # --- ÖNEMLİ: Butonu da sözlüğe ekliyoruz ki rengini değiştirebilelim ---
            self.course_ui[name] = {"sp_row": sr, "sp_col": sc, "btn": b_sel}
        cl.addWidget(self.tabs_course)
        bot_lay.addWidget(course_grp)

        splitter.addWidget(center_widget)

        # Splitter Oranları
        splitter.setStretchFactor(0, 1)  # Sol
        splitter.setStretchFactor(1, 4)  # Orta (Geniş)

        # Workerlar
        self.worker_1 = None
        self._connected = False
        self.heartbeat_timer = QtCore.QTimer(self);
        self.heartbeat_timer.timeout.connect(self._on_hb_timeout);
        self.heartbeat_timer.start(2000)

        self.drone_heartbeat_timer = QtCore.QTimer(self)
        self.drone_heartbeat_timer.timeout.connect(self._on_drone_hb_timeout)
        self.drone_heartbeat_timer.start(2000)
        # Araçlardan gelen "teyit edilmiş" noktaları tutan hafıza
        self.verified_mission_1 = []
        self.verified_mission_2 = []

        # En son halatları çiz (değişkenler artık tanımlı, hata vermez)
        self._layout_rope_stripes()

    # --- UI OLUŞTURUCU FONKSİYONLAR (Tekrarı Önlemek İçin) ---
    def _create_telemetry_grid(self, layout):
        # 1. Ana Çerçeve (Konteyner)
        frame = QtWidgets.QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 180);
                border: 1px solid #546e7a;
                border-radius: 6px;
            }
        """)
        layout.addWidget(frame)

        # Çerçevenin iç yerleşimi
        l_main = QtWidgets.QVBoxLayout(frame)
        l_main.setContentsMargins(0, 0, 0, 0)
        l_main.setSpacing(0)

        # 2. Modern Başlık
        lbl_header = QtWidgets.QLabel("TELEMETRİ VERİLERİ")
        lbl_header.setAlignment(QtCore.Qt.AlignCenter)
        lbl_header.setStyleSheet("""
            background-color: #37474f;
            color: #eceff1;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 2px;
            padding: 4px;
            border: none;
            border-bottom: 1px solid #546e7a;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        """)
        l_main.addWidget(lbl_header)

        # 3. Veri Izgarası
        grid_widget = QtWidgets.QWidget()
        grid_widget.setStyleSheet("border: none; background: transparent;")
        l_main.addWidget(grid_widget)

        g = QtWidgets.QGridLayout(grid_widget)
        g.setContentsMargins(4, 6, 4, 6)
        g.setSpacing(4)

        ui = {}
        keys = [
            "ARKA_SOL_PWM", "ARKA_SAĞ_PWM", "ÖN_SOL_PWM", "ÖN_SAĞ_PWM",
            "DÜMEN_PWM", "HIZ", "HDG", "HEDEF_HDG",
            "ACI_FARKI", "KONTROL_HATA", "HEDEFE_MESAFE", "MEVCUT_GOREV", "MANUEL",
            "ZAMAN", "FPS", "SENSOR_SAGLIK", "IDA_KONUM", "HEDEF_KONUM"
        ]

        for i, k in enumerate(keys):
            row = i // 2
            col_base = (i % 2) * 2

            # --- DEĞİŞİKLİK BURADA: BAŞLIKLAR ARTIK KALIN ---
            lbl_title = QtWidgets.QLabel(k)
            lbl_title.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            # font-weight: bold ekledik ve rengi açtık (#cfd8dc)
            lbl_title.setStyleSheet("""
                color: #cfd8dc;
                font-size: 14px;
                font-weight: bold;
                margin-right: 2px;
                border: none;
            """)
            g.addWidget(lbl_title, row, col_base)

            # --- DEĞİŞİKLİK BURADA: DEĞERLER DAHA BELİRGİN ---
            lbl_val = QtWidgets.QLabel("-")
            lbl_val.setAlignment(QtCore.Qt.AlignCenter)  # Yazıyı ortala

            lbl_val.setStyleSheet("""
                            color: #00e5ff;
                            font-weight: bold;
                            font-size: 14px;
                            border: none;
                            background-color: rgba(0, 0, 0, 40);
                            border-radius: 3px;
                        """)
            g.addWidget(lbl_val, row, col_base + 1)

            ui[k] = lbl_val

        return ui

    def _create_mission_box(self, layout, bid):
        l = self._add_modern_frame(layout, f"GÖREV PLANI (IDA {bid})")

        # Tablo
        table = QtWidgets.QTableWidget(0, 3)
        # Sütun Genişliği
        table.setColumnWidth(0, 45)
        table.setHorizontalHeaderLabels(["No", "Lat", "Lon"])
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        # --- KOMPAKT AYARLAR (GÜNCELLENDİ) ---
        # Satır Yüksekliği: 16 -> 18 (8 punto için nefes payı)
        table.verticalHeader().setDefaultSectionSize(18)

        # Başlık Yüksekliği: 18 -> 20
        table.horizontalHeader().setFixedHeight(20)

        # Font: 7 -> 8 Punto
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        table.setFont(font)

        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Yüksekliği de satır artışına göre 110 -> 115 yaptık
        table.setFixedHeight(115)

        # --- STİL ---
        table.setStyleSheet("""
            QTableWidget {
                border: none;
                background-color: #263238;
                color: white;
                gridline-color: #455a64;
            }
            QHeaderView::section {
                background-color: #37474f;
                color: #cfd8dc;
                border: 1px solid #546e7a;
                padding: 0px;
                font-size: 10px;
                height: 20px;
            }
            QTableWidget::item {
                padding-top: -1px;      /* Yazıyı ortalamak için ince ayar */
                padding-bottom: 0px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #00e676;
                color: black;
            }
        """)
        l.addWidget(table)

        # --- 1. SATIR (Butonlar) ---
        row1 = QtWidgets.QHBoxLayout()
        btn_h = 22

        b_add = QtWidgets.QPushButton("➕")
        b_add.setToolTip("Yeni Konum Ekle");
        b_add.setFixedWidth(40);
        b_add.setFixedHeight(btn_h)
        b_add.setStyleSheet(self._get_btn_style("#2e7d32", "#69f0ae", "#00e676"))
        b_add.clicked.connect(lambda: self._add_wp_row(bid))

        b_del_sel = QtWidgets.QPushButton("➖")
        b_del_sel.setToolTip("Seçili Satırı Sil");
        b_del_sel.setFixedWidth(40);
        b_del_sel.setFixedHeight(btn_h)
        b_del_sel.setStyleSheet(self._get_btn_style("#ef6c00", "#ff9800", "#ffb74d"))
        b_del_sel.clicked.connect(lambda: self._delete_selected_row(bid))

        # --- YENİ EKLENEN ÇOĞALTMA BUTONU ---
        b_dup = QtWidgets.QPushButton("📋")
        b_dup.setToolTip("Seçili Noktayı Çoğalt (Aşağıya Kopyala)")
        b_dup.setFixedWidth(40)
        b_dup.setFixedHeight(btn_h)
        b_dup.setStyleSheet(self._get_btn_style("#fbc02d", "#fff59d", "#ffee58"))  # Sarı Stil
        b_dup.clicked.connect(lambda: self._duplicate_selected_row(bid))
        # ------------------------------------

        b_clr = QtWidgets.QPushButton("🗑️")
        b_clr.setToolTip("TÜM LİSTEYİ Temizle");
        b_clr.setFixedWidth(40);
        b_clr.setFixedHeight(btn_h)
        b_clr.setStyleSheet(self._get_btn_style("#c62828", "#ff5252", "#ff1744"))
        b_clr.clicked.connect(lambda: table.setRowCount(0))

        b_snd = QtWidgets.QPushButton("🚀 GÖNDER")
        b_snd.setToolTip("Görevi Araca Yükle");
        b_snd.setFixedHeight(btn_h)
        b_snd.setStyleSheet(self._get_btn_style("#00c853", "#00e676", "#69f0ae"))
        b_snd.clicked.connect(lambda: self._send_mission(bid))

        row1.addWidget(b_add);
        row1.addWidget(b_del_sel);
        row1.addWidget(b_dup)  # <--- BUNU EKLE
        row1.addWidget(b_clr);
        row1.addWidget(b_snd)
        l.addLayout(row1)

        # --- 2. SATIR ---
        row2 = QtWidgets.QHBoxLayout()
        style_grey = self._get_btn_style("#455a64", "#90a4ae", "#78909c")
        style_blue = self._get_btn_style("#1565c0", "#448aff", "#2979ff")
        # Morumsu stil (Görev butonu için)
        style_purple = self._get_btn_style("#7b1fa2", "#e1bee7", "#ea80fc")

        btn_up = QtWidgets.QPushButton("▲");
        btn_up.setFixedWidth(30);
        btn_up.setFixedHeight(btn_h)
        btn_up.setStyleSheet(style_grey);
        btn_up.clicked.connect(lambda: self._move_row(bid, -1))

        btn_down = QtWidgets.QPushButton("▼");
        btn_down.setFixedWidth(30);
        btn_down.setFixedHeight(btn_h)
        btn_down.setStyleSheet(style_grey);
        btn_down.clicked.connect(lambda: self._move_row(bid, 1))

        # --- YENİ EKLENEN BUTON: GÖREV DEĞİŞTİR ---
        btn_task = QtWidgets.QPushButton("📝 GÖREV")
        btn_task.setToolTip("Mevcut Görevi Manuel Değiştir")
        btn_task.setFixedHeight(btn_h)
        btn_task.setStyleSheet(style_purple)  # Farklı renk olsun dikkat çeksin
        btn_task.clicked.connect(lambda: self._set_manual_task(bid))
        # ------------------------------------------

        b_s = QtWidgets.QPushButton("💾");
        b_s.setFixedHeight(btn_h);
        b_s.setStyleSheet(style_blue)
        b_s.clicked.connect(lambda: self._save_mission(bid))

        b_l = QtWidgets.QPushButton("📂");
        b_l.setFixedHeight(btn_h);
        b_l.setStyleSheet(style_blue)
        b_l.clicked.connect(lambda: self._load_mission(bid))

        row2.addWidget(btn_up);
        row2.addWidget(btn_down)
        row2.addWidget(btn_task)

        row2.addStretch()

        row2.addWidget(b_s);
        row2.addWidget(b_l)
        l.addLayout(row2)

        table.itemChanged.connect(lambda item: self._refresh_map_mission(bid))
        return table

    def _set_manual_task(self, bid):
        """Kullanıcıdan görev adı ister ve araca gönderir."""
        # Input Dialog aç
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            f"IDA {bid} Görev Atama",
            "Yeni Görev Adı (Örn: TASK2_START, FINISHED):",
            QtWidgets.QLineEdit.Normal,
            ""
        )

        if ok and text:
            task_name = text.strip()
            if not task_name: return

            # Komutu hazırla ve gönder
            w = self.worker_1 if bid == 1 else self.worker_2
            if w:
                w.queue_send({
                    "target_id": bid,
                    "cmd": "set_task",
                    "task_name": task_name
                })

    # MainWindow içindeki bu fonksiyonu değiştir:
    def _refresh_map_mission(self, bid, sync_other=True):
        """
        bid: Hangi botun tablosu değişti?
        sync_other: Diğer botu da güncelle (Sonsuz döngüyü engellemek için parametre).
        """
        t = self.usv1_table if bid == 1 else self.usv2_table
        local_pts = []
        for i in range(t.rowCount()):
            item_lat = t.item(i, 1)
            item_lon = t.item(i, 2)
            if item_lat is None or item_lon is None: continue
            try:
                lt, ln = float(item_lat.text()), float(item_lon.text())
                local_pts.append((lt, ln))
            except:
                pass

        remote_pts = self.verified_mission_1 if bid == 1 else self.verified_mission_2

        # Etiket seçimi
        if bid == 1:
            tags_to_send = self.MISSION_TAGS_1
        else:
            tags_to_send = self.MISSION_TAGS_2

        # Haritayı güncelle
        self.map.update_mission_waypoints(bid, local_pts, remote_pts, tags=tags_to_send)

        # --- SENKRONİZASYON ---
        # Eğer bu güncelleme ana tetikleyiciyse (sync_other=True),
        # diğer botun haritasını da güncelle ki "Turuncu" renkler eşleşsin.
        if sync_other:
            other_id = 2 if bid == 1 else 1
            self._refresh_map_mission(other_id, sync_other=False)

    def keyPressEvent(self, e):
        # Shift'e basıldığında Toggle yap
        if e.key() == Qt.Key_Shift:
            if not e.isAutoRepeat():  # Basılı tutmayı engelle
                self.map.toggle_smart_ruler()  # <-- start yerine toggle çağırdık

        # Hangi radyo butonu seçiliyse hedef o ID'dir
        target_id = 1
        if hasattr(self, 'radio_focus_2') and self.radio_focus_2.isChecked():
            target_id = 2

        if e.key() == Qt.Key_Up:
            self._send_manual_pwm(target_id, 1600, 1600)
        elif e.key() == Qt.Key_Down:
            self._send_manual_pwm(target_id, 1300, 1300)
        elif e.key() == Qt.Key_Left:
            self._send_manual_pwm(target_id, 1300, 1600)  # Sol için: Sol motor geri, sağ ileri
        elif e.key() == Qt.Key_Right:
            self._send_manual_pwm(target_id, 1600, 1300)  # Sağ için: Sol ileri, sağ geri
        elif e.key() == Qt.Key_Space:
            self._send_manual_pwm(target_id, 1500, 1500)  # STOP
        else:
            super().keyPressEvent(e)

    def _confirm_estop(self, bid):
        """Acil kapatma için onay kutusu çıkarır."""
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle("ACİL DURUM")
        msg.setText(f"DİKKAT: IDA {bid} motorları acil olarak durdurulacak!")
        msg.setInformativeText("Devam etmek istiyor musunuz?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.setDefaultButton(QtWidgets.QMessageBox.No)

        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self._send_cmd(bid, "emergency_stop", True)
            self.on_status(f"⚠️ USV{bid} İÇİN ACİL DURDURMA GÖNDERİLDİ!")

    def _add_modern_frame(self, parent_layout, title):
        """
        Modern çerçeve ve İÇİNDEKİ BUTONLARIN stilini tanımlar.
        """
        # 1. Ana Çerçeve
        frame = QtWidgets.QFrame()
        # DİKKAT: QPushButton stilini buraya ekledik!
        frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 180);
                border: 1px solid #546e7a;
                border-radius: 6px;
                margin-top: 4px;
            }
            /* --- ÇERÇEVE İÇİNDEKİ BUTONLAR --- */
            QPushButton {
                background-color: #455a64;       /* Koyu Gri Zemin */
                color: white;                    /* Beyaz Yazı */
                border: 2px solid #90a4ae;       /* BELİRGİN ÇERÇEVE (Metalik) */
                border-radius: 4px;
                padding: 4px;
                min-width: 24px;
                min-height: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #546e7a;
                border: 2px solid #00e5ff;       /* Üstüne gelince NEON MAVİ */
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #263238;
                border: 2px solid #546e7a;
                padding-top: 6px;
                padding-left: 6px;
            }
            QPushButton:checked {
                background-color: #00897b;
                border: 2px solid #00e676;
            }
            /* Etiketler */
            QLabel {
                border: none;
                background: transparent;
                color: #cfd8dc;
            }
        """)
        parent_layout.addWidget(frame)

        # Ana Düzen
        l_main = QtWidgets.QVBoxLayout(frame)
        l_main.setContentsMargins(0, 0, 0, 0)
        l_main.setSpacing(0)

        # 2. Modern Başlık
        lbl_header = QtWidgets.QLabel(title)
        lbl_header.setAlignment(QtCore.Qt.AlignCenter)
        # Başlık stili (Override edilmemesi için spesifik yazdık)
        lbl_header.setStyleSheet("""
            background-color: #37474f;
            color: #eceff1;
            font-size: 10px;
            font-weight: bold;
            letter-spacing: 1px;
            padding: 4px;
            border: none;
            border-bottom: 1px solid #546e7a;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
        """)
        l_main.addWidget(lbl_header)

        # 3. İçerik Alanı
        content_widget = QtWidgets.QWidget()
        # İçerik alanı şeffaf olsun ki butonlar frame'in rengini almasın
        content_widget.setStyleSheet("background: transparent; border: none;")
        l_main.addWidget(content_widget)

        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(4)  # Butonlar arası boşluğu biraz açtık

        return content_layout

    def _create_controls(self, layout, bid):
        container_layout = self._add_modern_frame(layout, f"KONTROL (IDA {bid})")

        g = QtWidgets.QGridLayout()
        g.setSpacing(6)
        container_layout.addLayout(g)

        # 1. Klavye Odağı
        rb_focus = QtWidgets.QRadioButton("⌨️ Klavye Odağı")
        rb_focus.setToolTip("Klavye (Ok Tuşları) ile bu aracı kontrol et")  # <--- EKLENDİ
        rb_focus.setStyleSheet(
            "QRadioButton { color: #00e676; font-weight: bold; background: transparent; border: none; }")
        if bid == 1:
            self.radio_focus_1 = rb_focus;
            self.radio_focus_1.setChecked(True)
        else:
            self.radio_focus_2 = rb_focus
        g.addWidget(rb_focus, 0, 0, 1, 2)

        # 2. Mod Butonları
        # OTONOM (Turkuaz)
        b_auto = QtWidgets.QPushButton("🤖 OTO")
        b_auto.setToolTip("Otonom Moda Geç (Görev Takibi Başlar)")  # <--- EKLENDİ
        b_auto.setStyleSheet(self._get_btn_style("#006064", "#00bcd4", "#18ffff"))
        b_auto.clicked.connect(lambda: self._send_cmd(bid, "set_manual", False))

        # MANUEL (Mor)
        b_man = QtWidgets.QPushButton("🎮 MAN")
        b_man.setToolTip("Manuel Moda Geç (Uzaktan Kumanda)")  # <--- EKLENDİ
        b_man.setStyleSheet(self._get_btn_style("#4a148c", "#ab47bc", "#e040fb"))
        b_man.clicked.connect(lambda: self._send_cmd(bid, "set_manual", True))

        g.addWidget(b_auto, 1, 0);
        g.addWidget(b_man, 1, 1)

        # 3. İz Kontrolleri (Buz Mavisi)
        style_ice = self._get_btn_style("#37474f", "#4fc3f7", "#81d4fa")

        b_fz = QtWidgets.QPushButton("❄️");
        b_fz.setToolTip("Hayalet İzi Dondur (Analiz için çizgiyi sabitler)")  # <--- EKLENDİ
        b_fz.setStyleSheet(style_ice)
        b_fz.clicked.connect(lambda: self.map.snapshot_ghost_trail() if bid == 1 else None)

        b_cl = QtWidgets.QPushButton("🧼");
        b_cl.setToolTip("Ekrandaki İzi Temizle")  # <--- EKLENDİ
        b_cl.setStyleSheet(style_ice)
        b_cl.clicked.connect(lambda: self.map.clear_ghost_trail_by_id(bid))

        g.addWidget(b_fz, 2, 0);
        g.addWidget(b_cl, 2, 1)

        # 4. Acil Kapatma (KOYU KIRMIZI)
        b_estop = QtWidgets.QPushButton("🛑 ACİL DURDUR")
        b_estop.setToolTip("DİKKAT: Motorlara giden gücü anında keser!")  # <--- EKLENDİ
        b_estop.setStyleSheet(self._get_btn_style("#b71c1c", "#ff1744", "#d50000"))
        b_estop.clicked.connect(lambda: self._confirm_estop(bid))
        g.addWidget(b_estop, 3, 0, 1, 2)

        # 5. Bilgi
        info_text = "[↑]İleri  [↓]Geri  [←][→]Yön  [SPACE]Dur"
        lbl_info = QtWidgets.QLabel(info_text)
        lbl_info.setAlignment(QtCore.Qt.AlignCenter)
        lbl_info.setStyleSheet(
            "color:#90a4ae; font-size:9px; border:1px dashed #546e7a; border-radius:4px; padding:2px; background:transparent;")
        g.addWidget(lbl_info, 4, 0, 1, 2)

    def _get_btn_style(self, bg_color, border_color, hover_color):
        """
        Verilen renklere göre Modern HUD butonu stili döndürür.
        """
        return f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                font-weight: bold;
                border: 2px solid {border_color};
                border-radius: 4px;
                padding: 4px;
                min-width: 24px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border: 2px solid white;
                color: white;
            }}
            QPushButton:pressed {{
                background-color: {bg_color};
                border: 2px solid {bg_color};
                padding-top: 6px;
                padding-left: 6px;
            }}
        """

    # --- MANTIK FONKSİYONLARI ---
    @QtCore.Slot(dict)
    def on_packet(self, d: dict):
        bid = int(d.get("id", 1))  # <--- ÇÖZÜM: Kesinlikle tam sayıya (1 veya 2) çevir!

        # İHA (Drone) paketi ise işle ve dön
        if bid == 3:
            drone_color_tr = d.get("drone_color", "BELIRSIZ")

            # LED yanıp sönme efekti ve timer reset
            self.iha_led.setStyleSheet("background-color:#00e676; border-radius:8px;")
            QtCore.QTimer.singleShot(100, lambda: self.iha_led.setStyleSheet("background-color:#1b5e20; border-radius:8px;"))
            if hasattr(self, 'drone_heartbeat_timer'):
                self.drone_heartbeat_timer.start(2500)

            if self.chk_drone_active.isChecked():
                self.lbl_iha_color.setText(drone_color_tr)

                # Türkçe rengi İngilizceye çevir
                mapped_color = "red"
                if drone_color_tr == "KIRMIZI": mapped_color = "red"
                elif drone_color_tr == "YESIL": mapped_color = "green"
                elif drone_color_tr == "SIYAH": mapped_color = "black"

                # Tekneye (IDA 1) yeni hedef rengi gönder. State-tracking to avoid redundant messages.
                if not hasattr(self, 'last_sent_drone_color') or self.last_sent_drone_color != mapped_color:
                    if self.worker_1:
                        self.worker_1.queue_send({"target_id": 1, "cmd": "set_target_color", "color": mapped_color})
                        self.last_sent_drone_color = mapped_color
            return

        task_str = d.get("task", "")

        # LED
        led = self.usv1_led if bid == 1 else getattr(self, 'usv2_led', self.usv1_led)
        led.setStyleSheet("background-color:#00e676; border-radius:10px;")
        QtCore.QTimer.singleShot(100, lambda: led.setStyleSheet("background-color:#1b5e20; border-radius:10px;"))
        self.heartbeat_timer.start(2500)

        # UI Update
        ui = self.usv1_ui if bid == 1 else self.usv2_ui
        # Eğer pakette görev listesi varsa hafızaya al ve haritayı güncelle

        # Anahtar ismini kendi protokolüne göre değiştirebilirsin: "mission_list", "points" vs.
        # Gelen veri: "GÖREV_NOKTALARI": {"GPS1": {"lat":..., "lon":...}, ...}
        if "GÖREV_NOKTALARI" in d:
            incoming_list = []
            try:
                points_dict = d["GÖREV_NOKTALARI"]

                # GPS1'den GPS12'ye kadar sırayla veriyi çek
                for i in range(1, 13):
                    key = f"GPS{i}"
                    if key in points_dict:
                        pt = points_dict[key]

                        # Veri dict formatında mı ve koordinat var mı?
                        if isinstance(pt, dict) and "lat" in pt and "lon" in pt:
                            lat = float(pt["lat"])
                            lon = float(pt["lon"])

                            # Sadece koordinatı olan (0,0 olmayan) noktaları listeye al
                            if lat != 0.0 or lon != 0.0:
                                incoming_list.append((lat, lon))

                # Hafızayı güncelle
                if bid == 1:
                    self.verified_mission_1 = incoming_list
                else:
                    self.verified_mission_2 = incoming_list

                # Haritayı yeniden çiz (Yeşil noktalar görünsün)
                self._refresh_map_mission(bid)

            except Exception as e:
                print(f"Görev listesi ayrıştırma hatası: {e}")

        # --- YARDIMCI GÜNCELLEME FONKSİYONU ---
        def s(k, v="-"):
            raw_val = d.get(v, "-")

            # Eğer veri yoksa veya None ise
            if raw_val is None or raw_val == "-":
                ui[k].setText("-")
                return

            try:
                # Sayısal veri mi diye kontrol et
                val_float = float(raw_val)

                # PWM gibi tam sayılar için (Örn: 1500.0 -> 1500)
                if val_float.is_integer():
                    txt = f"{int(val_float)}"
                else:
                    # HIZ, Heading gibi floatlar için (Örn: 2.5432 -> 2.54)
                    txt = f"{val_float:.2f}"

                # Yine de çok uzun bir şey oluşursa (güvenlik)
                if len(txt) > 8:
                    txt = txt[:8]

                ui[k].setText(txt)

            except (ValueError, TypeError):
                # Sayı değilse (Metin ise, örn: Görev Adı)
                txt = str(raw_val)
                # Görev ismi çok uzunsa sonunu kısalt (TASK_... -> TASK..)
                if len(txt) > 12:
                    txt = txt[:10] + ".."
                ui[k].setText(txt)

        s("ARKA_SOL_PWM", "pwm_L")
        s("ARKA_SAĞ_PWM", "pwm_R")
        s("ÖN_SOL_PWM", "pwm_FL")
        s("ÖN_SAĞ_PWM", "pwm_FR")
        s("DÜMEN_PWM", "pwm_STEER")
        s("HIZ", "spd")
        s("HDG", "hdg")
        s("HEDEF_HDG", "trg_hdg")
        s("ACI_FARKI", "err_ang")
        s("KONTROL_HATA", "ctrl_err")
        s("HEDEFE_MESAFE", "dist")
        s("MEVCUT_GOREV", "task")
        s("MANUEL", "mod")
        s("ZAMAN", "t_ms")
        s("FPS", "FPS")
        s("SENSOR_SAGLIK", "hlth")

        pos = d.get("MEVCUT_KONUM", {})
        lat = pos.get("lat", 0.0)
        lon = pos.get("lon", 0.0)
        if lat and lon:
            ui["IDA_KONUM"].setText(f"{lat:.5f},\n{lon:.5f}")

        t_pos = d.get("HEDEF_KONUM", {})
        t_lat = t_pos.get("lat", 0.0)
        t_lon = t_pos.get("lon", 0.0)
        if t_lat and t_lon:
            ui["HEDEF_KONUM"].setText(f"{t_lat:.5f},\n{t_lon:.5f}")

        if lat and lon:

            # Yazı boyutunu biraz küçültelim ki sığsın
            self.map.update_current_position(lat, lon, boat_id=bid)
            # Log
            logbox = self.usv1_log if bid == 1 else self.usv2_log
            # logbox.appendPlainText(f"DATA: {lat},{lon}") # İstersen aç

        # 3. GEOFENCE KORUMA MANTIĞI (Her iki tekne için)
        # Sadece "AKTİF" işaretliyse ve tekne OTONOM moddaysa çalışır.
        is_manual = bool(d.get("mod", True))  # Varsayılan True (Güvenlik)

        if self.chk_fence_active.isChecked() and not is_manual:
            pos = d.get("MEVCUT_KONUM")
            if isinstance(pos, dict):
                cur_lat = pos.get("lat")
                cur_lon = pos.get("lon")
                fence_pts = self.map.get_fence_points()

                # Çit en az 3 noktadan oluşmalı
                if cur_lat and cur_lon and len(fence_pts) >= 3:
                    # Nokta poligonun içinde DEĞİLSE (Dışındaysa)
                    if not is_point_in_polygon(cur_lat, cur_lon, fence_pts):
                        msg = f"⚠️ SINIR İHLALİ! USV{bid} DURDURULUYOR!"
                        self.on_status(msg)

                        # 1. Manuel moda al (Otonomdan çıkar)
                        self._send_cmd(bid, "set_manual", True)

                        # 2. Motorları durdur (PWM 1500)
                        self._send_manual_pwm(bid, 1500, 1500)

                        # 3. Sesli Uyarı (Opsiyonel)
                        QtWidgets.QApplication.beep()

    def _mk_btn(self, t, b, l, r):
        btn = QtWidgets.QPushButton(t)
        btn.clicked.connect(lambda: self._send_manual_pwm(b, l, r))
        return btn

    def _send_cmd(self, bid, cmd, val):
        if self.worker_1: self.worker_1.queue_send({"target_id": 1, "cmd": cmd, "value": val})
        self.on_status(f"TX[USV1]: {cmd}->{val}")

    def _send_manual_pwm(self, bid, l, r):
        if self.worker_1: self.worker_1.queue_send({"target_id": 1, "cmd": "manual_control", "left": l, "right": r})

    def _duplicate_selected_row(self, bid):
        """Seçili satırı kopyalar ve hemen altına yeni bir satır olarak ekler."""
        t = self.usv1_table if bid == 1 else self.usv2_table
        current = t.currentRow()

        # Eğer seçim yoksa uyar ve çık
        if current < 0:
            self.on_status(f"UYARI: Çoğaltılacak satır seçili değil (USV{bid}).")
            return

        # Seçili satırın lat ve lon değerlerini al
        lat = t.item(current, 1).text()
        lon = t.item(current, 2).text()

        # Bir alt satıra yeni bir satır aç
        target_row = current + 1
        t.insertRow(target_row)

        # Sinyalleri geçici durdur (harita gereksiz yere art arda tetiklenmesin)
        t.blockSignals(True)
        t.setItem(target_row, 0, QtWidgets.QTableWidgetItem(""))  # Label kısmı fonksiyonla dolacak
        t.setItem(target_row, 1, QtWidgets.QTableWidgetItem(lat))
        t.setItem(target_row, 2, QtWidgets.QTableWidgetItem(lon))
        t.blockSignals(False)

        # Tablo numaralarını düzelt ve haritayı yeniden çiz
        self._update_table_labels(bid)
        self._refresh_map_mission(bid)

        # Kullanıcı art arda kopyalayabilsin diye yeni satırı otomatik seçili yap
        t.selectRow(target_row)

    def _delete_selected_row(self, bid):
        """Tablodan seçili olan satırları siler."""
        t = self.usv1_table if bid == 1 else self.usv2_table

        # Çoklu seçim ihtimaline karşı tüm seçili satırları al
        # (Tersten siliyoruz ki indeksler kaymasın)
        rows = sorted(set(index.row() for index in t.selectedIndexes()), reverse=True)

        if not rows:
            # Eğer seçim yoksa, o anki aktif hücrenin satırını silmeyi dene
            current = t.currentRow()
            if current >= 0:
                rows = [current]
            else:
                self.on_status(f"UYARI: Silinecek satır seçili değil (USV{bid}).")
                return

        # Silme İşlemi
        for r in rows:
            t.removeRow(r)

        # Numaraları ve Haritayı Güncelle
        self._update_table_labels(bid)
        self._refresh_map_mission(bid)


    def _add_wp_row(self, bid):
        t = self.usv1_table if bid == 1 else self.usv2_table
        r = t.rowCount();
        t.insertRow(r)
        t.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r + 1)))
        t.setItem(r, 1, QtWidgets.QTableWidgetItem("0.0"));
        t.setItem(r, 2, QtWidgets.QTableWidgetItem("0.0"))
        self._refresh_map_mission(bid)
        self._update_table_labels(bid)

    def _update_table_labels(self, bid):
        """Tablonun 'No' sütununa numaraları ve etiketleri yazar."""
        t = self.usv1_table if bid == 1 else self.usv2_table

        if bid == 1:
            tags = self.MISSION_TAGS_1
        else:
            tags = self.MISSION_TAGS_2

        # --- GÜNCELLEME BURADA ---
        font_small = QtGui.QFont()
        font_small.setPointSize(8)  # Fontu 7'den 8'e çıkardık
        font_small.setBold(True)

        for i in range(t.rowCount()):
            label_text = str(i + 1)
            if i < len(tags):
                # Görev adını kısalt: TASK1_START -> T1_S
                short_tag = tags[i].replace("TASK", "T").replace("START", "S").replace("END", "E").replace("MID", "M")
                label_text += f".{short_tag}"

            item = t.item(i, 0)
            if not item:
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(Qt.ItemIsEnabled)  # Sadece okunabilir
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                t.setItem(i, 0, item)

            item.setFont(font_small)
            item.setText(label_text)

            # Satır yüksekliğini de 18px yapalım
            t.setRowHeight(i, 18)

    def _move_row(self, bid, direction):
        """
        Seçili satırı yukarı (-1) veya aşağı (+1) taşır.
        """
        table = self.usv1_table if bid == 1 else self.usv2_table
        current_row = table.currentRow()

        # Seçim yoksa veya geçersizse çık
        if current_row < 0: return

        # Hedef satır
        target_row = current_row + direction

        # Sınır kontrolü (Listenin dışına çıkma)
        if target_row < 0 or target_row >= table.rowCount():
            return

        # --- VERİ TAKASI (SWAP) ---
        # Lat ve Lon değerlerini al
        lat_curr = table.item(current_row, 1).text()
        lon_curr = table.item(current_row, 2).text()

        lat_target = table.item(target_row, 1).text()
        lon_target = table.item(target_row, 2).text()

        # Yerlerini değiştirerek geri yaz
        table.item(current_row, 1).setText(lat_target)
        table.item(current_row, 2).setText(lon_target)

        table.item(target_row, 1).setText(lat_curr)
        table.item(target_row, 2).setText(lon_curr)

        # Seçimi de taşı ki art arda basabilsin
        table.selectRow(target_row)

        # Haritayı güncelle (Çünkü sıra değişince etiketler de yer değiştirmeli)
        self._refresh_map_mission(bid)

    # MainWindow sınıfının içinde bu fonksiyonu bul ve değiştir:
    def _send_mission(self, bid):
        t = self.usv1_table if bid == 1 else self.usv2_table
        cnt = t.rowCount()

        if cnt == 0:
            self.on_status(f"UYARI: USV{bid} listesi boş, gönderilemedi.")
            return

        self.on_status(f"USV{bid}: {cnt} nokta hazırlanıyor...")

        sent_count = 0
        for i in range(cnt):
            try:
                # Hücrelerin boş olup olmadığını kontrol et
                item_lat = t.item(i, 1)
                item_lon = t.item(i, 2)

                if item_lat and item_lon and item_lat.text() and item_lon.text():
                    lat = float(item_lat.text())
                    lon = float(item_lon.text())

                    w = self.worker_1 if bid == 1 else self.worker_2
                    if w:
                        w.queue_send({
                            "target_id": bid,
                            "cmd": "set_gps",
                            "index": i + 1,
                            "lat": lat,
                            "lon": lon
                        })
                        sent_count += 1
                else:
                    self.on_status(f"HATA: Satır {i + 1} boş veya geçersiz!")
            except ValueError:
                self.on_status(f"HATA: Satır {i + 1} sayısal değil!")
            except Exception as e:
                self.on_status(f"HATA: {e}")

        if sent_count > 0:
            self.on_status(f"TX: USV{bid} için {sent_count} komut kuyruğa eklendi.")

    def _save_mission(self, bid):
        t = self.usv1_table if bid == 1 else self.usv2_table
        d = [{"lat": t.item(i, 1).text(), "lon": t.item(i, 2).text()} for i in range(t.rowCount())]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kaydet", f"usv{bid}.json")
        if path:
            with open(path, "w") as f: json.dump(d, f)

    def _load_mission(self, bid):
        t = self.usv1_table if bid == 1 else self.usv2_table
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Yükle")
        if path:
            with open(path, "r") as f:
                data = json.load(f)
            t.setRowCount(0)
            for p in data:
                r = t.rowCount();
                t.insertRow(r)
                t.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r + 1)))
                t.setItem(r, 1, QtWidgets.QTableWidgetItem(str(p["lat"])));
                t.setItem(r, 2, QtWidgets.QTableWidgetItem(str(p["lon"])))

        self._refresh_map_mission(bid)
        self._update_table_labels(bid)

    @QtCore.Slot(int, float, float, int)
    def _on_map_gps_selected(self, idx, lat, lon, target_row):
        """
        idx: 1 (IDA1), 2 (IDA2), 3 (Her İkisi)
        target_row: -1 ise listenin sonuna yeni satır ekler.
                    0, 1, 2 vb. ise o satırdaki mevcut GPS verisini günceller.
        """

        targets = []
        if idx == 1:
            targets = [1]
        elif idx == 2:
            targets = [2]
        elif idx == 3:
            targets = [1, 2]

        for target in targets:
            t = self.usv1_table if target == 1 else self.usv2_table

            if target_row == -1:
                # --- YENİ SATIR EKLEME MODU ---
                r = t.rowCount()
                t.insertRow(r)
                t.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r + 1)))
                t.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{lat:.7f}"))
                t.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{lon:.7f}"))
                self.on_status(f"Haritadan USV{target} listesine yeni nokta eklendi.")

            else:
                # --- MEVCUT SATIRI GÜNCELLEME MODU ---
                if target_row < t.rowCount():
                    # Sinyal döngüsünü engellemek için geçici olarak kapat
                    t.blockSignals(True)
                    t.item(target_row, 1).setText(f"{lat:.7f}")
                    t.item(target_row, 2).setText(f"{lon:.7f}")
                    t.blockSignals(False)
                    self.on_status(f"USV{target} - {target_row + 1}. Nokta haritadan güncellendi.")

            # Güncellemeler (Haritayı ve Tablo numaralarını yenile)
            self._refresh_map_mission(target)
            self._update_table_labels(target)

    # --- Diğer Yardımcı Fonksiyonlar (Eskilerden kalanlar) ---
    def _toggle_fence_draw(self, c):
        if c:
            self.btn_fence_draw.setText("Bitir"); self.map.start_fence_drawing()
        else:
            self.btn_fence_draw.setText("Çiz"); self.map.finish_fence_drawing()

    def _on_drone_active_changed(self, state):
        if not self.chk_drone_active.isChecked():
            # Kullanıcı checkbox'ı kapattıysa, varsayılan rengi göster
            default_color = getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red')
            tr_default = default_color.upper()
            if default_color == 'red': tr_default = "KIRMIZI"
            elif default_color == 'green': tr_default = "YESIL"
            elif default_color == 'black': tr_default = "SIYAH"

            self.lbl_iha_color.setText(f"Varsayılan: {tr_default}")

            # Tekneye (IDA 1) varsayılan hedef rengi gönder ki sıfırlansın
            if self.worker_1:
                self.worker_1.queue_send({"target_id": 1, "cmd": "set_target_color", "color": default_color.lower()})
                self.last_sent_drone_color = default_color.lower()

    def _on_hb_timeout(self):
        self.usv1_led.setStyleSheet("background-color:#b71c1c; border-radius:10px;")

    def _on_drone_hb_timeout(self):
        self.iha_led.setStyleSheet("background-color:#b71c1c; border-radius:8px;")
        # Drone bağlantısı koparsa varsayılan renge dön
        if getattr(self, 'chk_drone_active', None) and self.chk_drone_active.isChecked():
            default_color = getattr(cfg, 'TASK3_KAMIKAZE_COLOR', 'red')
            tr_default = default_color.upper()
            if default_color == 'red': tr_default = "KIRMIZI"
            elif default_color == 'green': tr_default = "YESIL"
            elif default_color == 'black': tr_default = "SIYAH"

            self.lbl_iha_color.setText(f"Varsayılan: {tr_default}")

            if self.worker_1 and getattr(self, 'last_sent_drone_color', None) != default_color.lower():
                self.worker_1.queue_send({"target_id": 1, "cmd": "set_target_color", "color": default_color.lower()})
                self.last_sent_drone_color = default_color.lower()

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()] or ["COM3"]
        self.cmb_port_1.clear()
        self.cmb_port_1.addItems(ports)

    def _toggle(self):
        if not self._connected:
            # --- BAĞLANMA İŞLEMİ ---
            port1 = self.cmb_port_1.currentText().strip()
            baud = int(self.cmb_baud.currentText())
            csv_path = "telemetry_log.csv" if self.chk_csv.isChecked() else None

            # WORKER 1 (IDA 1)
            self.worker_1 = SerialWorker(port1, baud, csv_path)
            self.worker_1.packet.connect(self.on_packet)
            self.worker_1.status.connect(self.on_status)
            self.worker_1.link.connect(self.on_link)
            self.worker_1.start()

            # POLLING
            self._polling_timer = QtCore.QTimer(self)
            self._polling_timer.timeout.connect(self._perform_polling)
            self._polling_timer.start(100)

            self._connected = True
            self.btn_connect.setText("BAĞLI")
            self.btn_connect.setStyleSheet("background-color: #00e676; color: black; font-weight: bold;")
            self.on_status(f"BAĞLANDI: {port1} @ {baud}")

        else:
            # --- KOPARMA İŞLEMİ ---
            if self.worker_1:
                try:
                    self.worker_1.packet.disconnect()
                except:
                    pass
                self.worker_1.stop()
                if not self.worker_1.wait(1000): self.worker_1.terminate()
                self.worker_1 = None

            if hasattr(self, '_polling_timer'):
                self._polling_timer.stop()

            self._connected = False
            self.btn_connect.setText("BAĞLAN")
            self.btn_connect.setStyleSheet("background-color: #455a64; color: white;")
            self.on_status("Bağlantı kesildi.")

    def _perform_polling(self):
        """Her iki araca da kendi özel hatlarından AYNI ANDA durum sorgusu atar."""
        if self.worker_1: self.worker_1.queue_send({"target_id": 1, "cmd": "report_status"})

    def _on_map_changed(self):
        d = self.MAP_PRESETS[self.cmb_maps.currentText()]
        self.map.change_map(res_path(d['file']), d['n'], d['s'], d['w'], d['e'])

    def _update_ring_settings(self):
        self.map.ring_step_m = self.sp_ring_step.value();
        self.map.ring_count = self.sp_ring_count.value()
        if self.map.rings_enabled: self.map.update_range_rings()

    def _toggle_rings(self, c):
        self.map.rings_enabled = c; self.map.update_range_rings()

    def _start_course_marking(self, n):
        self.map.start_defining_course(n)
        self.on_status(f"Parkur {n} Seçim Modu")

        # Sekme indeksini bul (A=0, B=1, C=2)
        idx = ["A", "B", "C"].index(n)

        # O anki sekmeye otomatik geçiş yap (Kullanıcı başka yerdeyse bile)
        self.tabs_course.setCurrentIndex(idx)

        # --- DİNAMİK RENK DEĞİŞTİRME ---
        # Sadece seçili olan sekme başlığı (QTabBar::tab:selected) SARI olsun.
        # Diğerleri varsayılan (koyu) kalsın.

        # Renk Kodları:
        # Seçili (Aktif): #ffeb3b (Sarı), Yazı: Siyah
        # Normal: #455a64 (Gri), Yazı: Beyaz

        self.tabs_course.setStyleSheet(f"""
            QTabBar::tab {{
                background: #455a64;
                color: white;
                padding: 8px 20px;
                margin: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: #ffeb3b;
                color: black;
                font-weight: bold;
            }}
        """)

    def _update_grid_params(self, n):
        ui = self.course_ui[n];
        self.map.courses[n].update({"rows": ui["sp_row"].value(), "cols": ui["sp_col"].value()})
        self.map.render_course_grid(n)

    def _save_courses(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kaydet", "courses.json")
        if p:
            with open(p, "w") as f:
                d = {k: {"corners": v["corners"], "rows": v["rows"], "cols": v["cols"]} for k, v in
                     self.map.courses.items()}
                json.dump(d, f)

    def _save_single_course(self, name):
        """Sadece seçilen parkuru (A, B veya C) kaydeder."""
        data = self.map.courses.get(name)
        if not data or not data["corners"]:
            self.on_status(f"UYARI: Parkur {name} için veri yok, kaydedilmedi.")
            return

        # Dosya adı önerisi: parkur_A.json
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"Parkur {name} Kaydet", f"parkur_{name}.json",
                                                        "JSON (*.json)")
        if path:
            to_save = {
                "name": name,
                "corners": data["corners"],
                "rows": self.course_ui[name]["sp_row"].value(),
                "cols": self.course_ui[name]["sp_col"].value()
            }
            try:
                with open(path, "w") as f:
                    json.dump(to_save, f, indent=4)
                self.on_status(f"Parkur {name} kaydedildi: {os.path.basename(path)}")
            except Exception as e:
                self.on_status(f"Kaydetme Hatası: {e}")

    def _load_single_course(self, name):
        """Seçilen parkur verisini dosyadan yükler."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Parkur {name} Yükle", "", "JSON (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    d = json.load(f)

                # Veriyi hafızaya al
                self.map.courses[name]["corners"] = d.get("corners", [])

                # Spinboxları güncelle
                r = d.get("rows", 4)
                c = d.get("cols", 4)
                self.course_ui[name]["sp_row"].setValue(r)
                self.course_ui[name]["sp_col"].setValue(c)

                # Haritayı güncelle (Köşeler ve Izgara)
                self.map.courses[name]["rows"] = r
                self.map.courses[name]["cols"] = c
                self.map.render_course_grid(name)

                self.on_status(f"Parkur {name} yüklendi.")
            except Exception as e:
                self.on_status(f"Yükleme Hatası: {e}")

    def _clear_single_course(self, name):
        """Sadece seçilen parkuru haritadan ve hafızadan siler."""
        # 1. Hafızayı temizle
        self.map.courses[name]["corners"] = []

        # 2. Görseli temizle
        self.map.clear_course_visuals(name)

        self.on_status(f"Parkur {name} silindi.")

    def _load_courses(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Yükle")
        if p:
            with open(p, "r") as f:
                d = json.load(f)
                for k, v in d.items():
                    if k in self.map.courses:
                        self.map.courses[k].update(v)
                        self.course_ui[k]["sp_row"].setValue(v["rows"]);
                        self.course_ui[k]["sp_col"].setValue(v["cols"])
                        self.map.render_course_grid(k)

    @QtCore.Slot(str)
    def on_status(self, s):
        self.usv1_log.appendPlainText(s)

    @QtCore.Slot(bool)
    def on_link(self, b):
        self.on_status("LINK OK" if b else "LINK LOST")

    def _layout_rope_stripes(self):
        w = self.content.width();
        h = self.content.height();
        t = self._rope_thk
        self.rope_top.setGeometry(0, 0, w, t);
        self.rope_bottom.setGeometry(0, h - t, w, t)
        self.rope_left.setGeometry(0, 0, t, h);
        self.rope_right.setGeometry(w - t, 0, t, h)

    def eventFilter(self, o, e):
        if o == self.content and e.type() == QtCore.QEvent.Resize: self._layout_rope_stripes()
        return super().eventFilter(o, e)

    def closeEvent(self, e):
        if self.worker_1: self.worker_1.stop()
        e.accept()

    def _fmt_float(self, v):
        try:
            return f"{float(v):.1f}"
        except:
            return "-"


def main():
    app = QtWidgets.QApplication(sys.argv)
    icon_path = res_path("cross.ico").replace("\\", "/")
    app.setWindowIcon(QtGui.QIcon(icon_path))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()