import RPi.GPIO as GPIO
import time

# Pin definitions
AIN1 = 17
AIN2 = 27
PWMA = 18
STBY = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup([AIN1, AIN2, STBY], GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)

# Enable driver
GPIO.output(STBY, GPIO.HIGH)

# PWM setup
pwm = GPIO.PWM(PWMA, 1000)  # 1 kHz
pwm.start(0)

print("Motor ON")
GPIO.output(AIN1, GPIO.HIGH)
GPIO.output(AIN2, GPIO.LOW)
pwm.ChangeDutyCycle(50)  # 50% speed

time.sleep(2)

print("Motor OFF")
pwm.ChangeDutyCycle(0)
GPIO.output(AIN1, GPIO.LOW)
GPIO.output(AIN2, GPIO.LOW)

pwm.stop()
GPIO.cleanup()
