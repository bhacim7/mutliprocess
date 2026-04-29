import pyzed.sl as sl


class TimestampHandler:
    """Sensör verisinin (IMU vb.) eski mi yoksa yeni mi olduğunu kontrol eder."""

    def __init__(self):
        self.t_imu = sl.Timestamp()

    def is_new(self, sensor):
        if isinstance(sensor, sl.IMUData):
            new_ = (sensor.timestamp.get_microseconds() > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_


def initialize_camera():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init_params.depth_minimum_distance = 0.20
    init_params.depth_maximum_distance = 20
    init_params.camera_disable_self_calib = False
    init_params.depth_stabilization = 50
    init_params.sensors_required = False
    init_params.enable_image_enhancement = True
    init_params.async_grab_camera_recovery = False

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise Exception(f"ZED kamera açılamadı: {err}")
    return zed


def initialize_positional_tracking(zed):
    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    tracking_parameters.enable_pose_smoothing = True
    tracking_parameters.set_floor_as_origin = False
    tracking_parameters.enable_area_memory = True
    tracking_parameters.enable_imu_fusion = True
    tracking_parameters.set_as_static = False
    tracking_parameters.depth_min_range = 0.40
    tracking_parameters.mode = sl.POSITIONAL_TRACKING_MODE.GEN_1

    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Konumsal takip başlatılamadı: {err}")
        zed.close()
        exit()


def temiz_kapat(cam: sl.Camera):
    """Açık modüller varsa sessizce kapatıp kamerayı devreden çıkarır."""
    for fn in ("disable_recording", "disable_spatial_mapping", "disable_positional_tracking"):
        try:
            getattr(cam, fn)()
        except Exception:
            pass
    try:
        cam.close()
    except Exception:
        pass