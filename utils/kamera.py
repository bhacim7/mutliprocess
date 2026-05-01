# Legacy imports or stub functions needed by camera_process or others
class TimestampHandler:
    def __init__(self):
        self.t_imu = 0

    def is_new(self, sensor):
        if isinstance(sensor, int) or isinstance(sensor, float):
             # Just a stub check if we pass timestamps directly
             if sensor > self.t_imu:
                 self.t_imu = sensor
                 return True
             return False

        # If it's a PyZED sensor object
        if hasattr(sensor, 'timestamp'):
            if sensor.timestamp.get_microseconds() > self.t_imu:
                self.t_imu = sensor.timestamp.get_microseconds()
                return True
        return False
