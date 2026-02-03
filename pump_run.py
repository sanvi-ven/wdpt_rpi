import time
import serial

# This is usually ttyACM0 or ttyUSB0
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # let Arduino reset

def _now():
	return time.strftime('%Y-%m-%d %H:%M:%S')

print(f"{_now()}  Pump ON -> sending '1'")
ser.write(b'1')
ser.flush()

# try to read an optional ACK from the Arduino (non-blocking due to timeout)
try:
	resp = ser.readline().decode('utf-8', errors='ignore').strip()
	if resp:
		print(f"{_now()}  Arduino replied: {resp}")
	else:
		print(f"{_now()}  No immediate Arduino reply (continue)")
except Exception as e:
	print(f"{_now()}  Error reading ACK: {e}")

print(f"{_now()}  Sleeping 12 seconds")
start = time.time()
time.sleep(0.6)
elapsed = time.time() - start
print(f"{_now()}  Woke after {elapsed:.2f} seconds")

print(f"{_now()}  Pump OFF -> sending '0'")
ser.write(b'0')
ser.flush()

try:
	resp = ser.readline().decode('utf-8', errors='ignore').strip()
	if resp:
		print(f"{_now()}  Arduino replied: {resp}")
	else:
		print(f"{_now()}  No immediate Arduino reply after OFF")
except Exception as e:
	print(f"{_now()}  Error reading ACK after OFF: {e}")

ser.close()
#----
