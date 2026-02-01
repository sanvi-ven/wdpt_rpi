from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time

# Direction pins (start low for safety)
AIN1 = DigitalOutputDevice(17, initial_value=False)
AIN2 = DigitalOutputDevice(27, initial_value=False)

# PWM speed pin (start at 0)
PWMA = PWMOutputDevice(18, initial_value=0)

# Standby pin (start disabled)
STBY = DigitalOutputDevice(22, initial_value=False)


def run_motor():
	try:
		# ensure safe state before enabling driver
		AIN1.off()
		AIN2.off()
		PWMA.value = 0

		# enable motor driver
		print("Enabling motor driver (STBY on)")
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
	finally:
		# always disable driver on exit
		print("Disabling motor driver (STBY off)")
		STBY.off()


if __name__ == "__main__":
	run_motor()
