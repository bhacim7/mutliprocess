class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0.0
        self.estimated_error = 1.0
        self.is_initialized = False

    def update(self, measurement):
        if not self.is_initialized:
            self.estimated_value = measurement
            self.is_initialized = True
            return self.estimated_value

        # Handle wrap-around for heading
        diff = measurement - self.estimated_value
        while diff > 180.0:
            diff -= 360.0
        while diff < -180.0:
            diff += 360.0

        priori_error = self.estimated_error + self.process_variance
        kalman_gain = priori_error / (priori_error + self.measurement_variance)

        self.estimated_value = self.estimated_value + kalman_gain * diff

        # Keep within 0-360
        self.estimated_value = self.estimated_value % 360.0

        self.estimated_error = (1.0 - kalman_gain) * priori_error

        return self.estimated_value
