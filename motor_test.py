from gpiozero import PWMOutputDevice, DigitalOutputDevice
import time

# Direction pins (start low for safety)
AIN1 = DigitalOutputDevice(17, initial_value=False)
AIN2 = DigitalOutputDevice(27, initial_value=False)

# PWM speed pin (start at 0)
PWMA = PWMOutputDevice(18, initial_value=0)

# Standby pin (start disabled)
STBY = DigitalOutputDevice(22, initial_value=False)

# basic runtime checks
if not hasattr(STBY, 'on'):
	# gpiozero didn't initialize the device as expected
	raise RuntimeError('gpiozero STBY device not available')

# warn if STBY is configured active-low (polarity mismatch risk)
if not getattr(STBY, 'active_high', True):
	print("Warning: STBY appears to be active-low (active_high=False). Confirm wiring/polarity.")


def run_motor():
	try:
		# ensure safe state before enabling driver
		AIN1.off()
		AIN2.off()
		PWMA.value = 0

		# enable motor driver
		print("Enabling motor driver (STBY on)")
		STBY.on()

		# small delay to allow the GPIO level to settle, then verify the pin changed
		time.sleep(0.05)
		# verify physical pin level matches expected polarity
		if getattr(STBY, 'active_high', True):
			# active-high: expect a non-zero value after on()
			if getattr(STBY, 'value', 0) == 0:
				raise RuntimeError('STBY did not go high after STBY.on(); check wiring/power/polarity')
		else:
			# active-low: expect a zero value after on()
			if getattr(STBY, 'value', 1) == 1:
				raise RuntimeError('STBY did not go low after STBY.on() (active-low); check wiring/polarity')

		print("Motor forward")
		AIN1.off()
		AIN2.on()
		PWMA.value = 0.5   # 50% speed

		time.sleep(30)

		print("Motor stop")
		PWMA.value = 0
		AIN1.off()
		AIN2.off()
	finally:
		# always disable driver on exit
		print("Disabling motor driver (STBY off)")
		STBY.off()

		# explicitly close gpiozero devices to release GPIO cleanly
		try:
			PWMA.close()
		except Exception:
			pass
		try:
			AIN1.close()
		except Exception:
			pass
		try:
			AIN2.close()
		except Exception:
			pass
		try:
			STBY.close()
		except Exception:
			pass


if __name__ == "__main__":
	run_motor()
