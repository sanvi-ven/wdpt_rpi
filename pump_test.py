import serial
import time

# This is usually ttyACM0 or ttyUSB0
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # let Arduino reset

print("Pump ON")
ser.write(b'1')
time.sleep(3)

print("Pump OFF")
ser.write(b'0')

ser.close()
