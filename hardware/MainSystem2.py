import serial
import time
import dronekit
from dronekit import connect, VehicleMode

class USVController:
    """
    Interface to communicate with the Flight Controller (e.g., OrangeCube) via MAVLink.
    """
    def __init__(self, port, baud=57600):
        self.port = port
        self.baud = baud
        self.pwms = {1: 1500, 3: 1500} # left=1, right=3
        print(f"[USVController] Initializing on {port} at {baud} baud. Waiting for connection...")
        try:
            self.vehicle = connect(port, baud=baud, wait_ready=True)
            print("[USVController] Connection successful.")
        except Exception as e:
            print(f"[USVController] Connection failed: {e}")
            self.vehicle = None

    def get_current_position(self):
        """Returns latitude and longitude."""
        if self.vehicle and self.vehicle.location.global_frame:
            return self.vehicle.location.global_frame.lat, self.vehicle.location.global_frame.lon
        return 0.0, 0.0

    def get_horizontal_speed(self):
        """Returns ground speed in m/s."""
        if self.vehicle:
            return self.vehicle.groundspeed
        return 0.0

    def get_heading(self):
        """Returns compass heading in degrees."""
        if self.vehicle:
            return self.vehicle.heading
        return 0.0

    def get_servo_pwm(self, channel):
        """Returns the last commanded PWM for a given channel."""
        return self.pwms.get(channel, 1500)

    def set_servo(self, channel, pwm):
        """
        Commands the hardware to set a specific PWM on a servo channel.
        """
        self.pwms[channel] = pwm
        if self.vehicle:
            msg = self.vehicle.message_factory.command_long_encode(
                0, 0,    # target_system, target_component
                dronekit.mavutil.mavlink.MAV_CMD_DO_SET_SERVO, # command
                0,       # confirmation
                channel, # param1: Servo number
                pwm,     # param2: PWM value
                0, 0, 0, 0, 0 # params 3-7 not used
            )
            self.vehicle.send_mavlink(msg)

    def set_mode(self, mode):
        """Changes the vehicle flight mode (e.g., MANUAL, GUIDED)."""
        print(f"[USVController] Mode set to {mode}")
        if self.vehicle:
            self.vehicle.mode = VehicleMode(mode)

    def arm_vehicle(self):
        """Arms the thrusters."""
        print("[USVController] Vehicle armed")
        if self.vehicle:
            self.vehicle.armed = True

    def disarm_vehicle(self):
        """Disarms the thrusters for safety."""
        print("[USVController] Vehicle disarmed")
        if self.vehicle:
            self.vehicle.armed = False

    def get_gps_fix_type_verbose(self):
        """Returns a human-readable GPS fix type."""
        if self.vehicle and self.vehicle.gps_0:
            return f"Fix: {self.vehicle.gps_0.fix_type}"
        return "No Fix"
