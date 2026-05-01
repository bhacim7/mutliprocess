import time
import config as cfg

def set_motor_power(left_pwm, right_pwm):
    """
    Direct control of motor relays or ESCs via Edge Device GPIO (e.g., Jetson/Raspberry Pi).
    """
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        # Assumes config has PIN values
        left_pin = getattr(cfg, 'LEFT_MOTOR_PIN', 12)
        right_pin = getattr(cfg, 'RIGHT_MOTOR_PIN', 13)
        GPIO.setup(left_pin, GPIO.OUT)
        GPIO.setup(right_pin, GPIO.OUT)

        # In a real scenario, hardware PWM would be set up here.
        # This is a basic outline representing the requested implementation.
    except ImportError:
        pass
