# motor_server.py
import socket
import threading
from time import sleep
from gpiozero import OutputDevice, PWMOutputDevice
import serial

# ---------------- MOTOR SETUP ----------------
stby = OutputDevice(12)
pwm = PWMOutputDevice(18)  # PWM for wheels

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

# ---------------- ARDUINO SETUP ----------------
ARDUINO_PORT = "/dev/ttyACM0"
arduino = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
sleep(2)  # wait for Arduino to initialize

# ---------------- MOTOR FUNCTIONS ----------------
def forward(duration=2):
    stby.on()
    pwm.value = 0.5
    # Driver 1
    d1_ain1.on()
    d1_ain2.off()
    d1_bin1.off()
    d1_bin2.on()
    # Driver 2
    d2_ain1.off()
    d2_ain2.on()
    d2_bin1.on()
    d2_bin2.off()
    sleep(duration)
    stop()
    pwm.off()
    stby.off()

def stop():
    for dev in [d1_ain1, d1_ain2, d1_bin1, d1_bin2,
                d2_ain1, d2_ain2, d2_bin1, d2_bin2,
                d3_ain1, d3_ain2]:
        dev.off()

def shallow_till(duration=3):
    stby.on()
    pwm.value = 0.5
    d3_ain1.on()
    d3_ain2.off()
    sleep(duration)
    stop()
    pwm.off()
    stby.off()

def deep_till(duration=5):
    stby.on()
    pwm.value = 0.5
    d3_ain1.on()
    d3_ain2.off()
    sleep(duration)
    stop()
    pwm.off()
    stby.off()

def run_pump(duration=5):
    """Send command to Arduino to run pump"""
    arduino.write(b'PUMP_ON\n')
    sleep(duration)
    arduino.write(b'PUMP_OFF\n')

# ---------------- SOCKET SERVER ----------------
HOST = ''  # listen on all interfaces
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
            elif command == "STOP":
                stop()
            elif command == "SHALLOW_TILL":
                shallow_till()
            elif command == "DEEP_TILL":
                deep_till()
            elif command == "PUMP_ON":
                run_pump()
            elif command == "NO_TILL":
                stop()
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
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()
except KeyboardInterrupt:
    print("Shutting down server...")
finally:
    server.close()
    stop()
    arduino.close()
