from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time

# Direction pins
AIN1 = DigitalOutputDevice(17)
AIN2 = DigitalOutputDevice(27)

# PWM speed pin
PWMA = PWMOutputDevice(18)

# Standby pin
STBY = DigitalOutputDevice(22)

# Enable motor driver
STBY.on()

print("Motor forward")
AIN1.on()
AIN2.off()
PWMA.value = 0.5   # 50% speed

time.sleep(2)

print("Motor stop")
PWMA.value = 0
AIN1.off()
AIN2.off()

STBY.off()
