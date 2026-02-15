# motor_server_fixed.py
import socket
import threading
from time import sleep
from gpiozero import OutputDevice, PWMOutputDevice
import serial

# ---------------- MOTOR SETUP ----------------
stby = OutputDevice(12)
pwm = PWMOutputDevice(18)  # PWM for wheels
stby.on()
pwm.value = 0.5  # default speed

# Driver 1
d1_ain1 = OutputDevice(16)
d1_ain2 = OutputDevice(25)
d1_bin1 = OutputDevice(5)
d1_bin2 = OutputDevice(6)

# Driver 2
d2_ain1 = OutputDevice(24)
d2_ain2 = OutputDevice(23)
d2_bin1 = OutputDevice(27)
d2_bin2 = OutputDevice(22)

# Driver 3: Tiller
d3_ain1 = OutputDevice(4)
d3_ain2 = OutputDevice(17)

# ---------------- ARDUINO ----------------
ARDUINO_PORT = "/dev/ttyACM0"
arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
sleep(2)

# ---------------- MOTOR FUNCTIONS ----------------
def forward(duration=2):
    print("Forward start")
    d1_ain1.on(); d1_ain2.off(); d1_bin1.off(); d1_bin2.on()
    d2_ain1.off(); d2_ain2.on(); d2_bin1.on(); d2_bin2.off()
    threading.Thread(target=stop_after, args=(duration,)).start()

def shallow_till(duration=3):
    print("Shallow till start")
    pwm = PWMOutputDevice(13) # separate PWM for tiller if needed
    pwm.value = 0.5   # 50% speed
    d3_ain1.on(); d3_ain2.off()
    threading.Thread(target=stop_after, args=(duration,)).start()

def deep_till(duration=5):
    pwm = PWMOutputDevice(13) # separate PWM for tiller if needed
    pwm.value = 0.5   # 50% speed
    print("Deep till start")
    d3_ain1.on(); d3_ain2.off()
    threading.Thread(target=stop_after, args=(duration,)).start()

def run_pump(duration=5):
    print("Pump start")
    arduino.write(b'ON\n')
    threading.Thread(target=pump_off_after, args=(duration,)).start()

def stop_after(duration):
    sleep(duration)
    stop()

def pump_off_after(duration):
    sleep(duration)
    arduino.write(b'OFF\n')

def stop():
    for dev in [d1_ain1, d1_ain2, d1_bin1, d1_bin2,
                d2_ain1, d2_ain2, d2_bin1, d2_bin2,
                d3_ain1, d3_ain2]:
        dev.off()
    print("Motors stopped")

# ---------------- SOCKET SERVER ----------------
HOST = ''
PORT = 5005
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"Motor server listening on port {PORT}...")

def handle_client(conn, addr):
    print(f"Connected by {addr}")
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            command = data.decode().strip().upper()
            print(f"Received command: {command}")

            if command == "FORWARD":
                forward()
            elif command == "STOP" or command == "NO_TILL":
                stop()
            elif command == "SHALLOW_TILL":
                shallow_till()
            elif command == "DEEP_TILL":
                deep_till()
            elif command == "PUMP_ON":
                run_pump()
            else:
                print(f"Unknown command: {command}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()
        print(f"Connection with {addr} closed.")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()
except KeyboardInterrupt:
    print("Shutting down server...")
finally:
    stop()
    arduino.close()
    server.close()
