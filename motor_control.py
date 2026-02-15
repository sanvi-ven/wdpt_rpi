# motor_server_fixed.py
import socket
import threading
from time import sleep
from gpiozero import OutputDevice, PWMOutputDevice
import serial

# ---------------- MOTOR SETUP ----------------
stby = OutputDevice(12)
stby.on()


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
wheel_pwm = PWMOutputDevice(18)     # Wheels PWM
tiller_pwm = PWMOutputDevice(13)    # Tiller PWM


def forward(duration=1):
    print("Forward start")
    stby.on()
    wheel_pwm.value = 0.5
    # Correct forward for both sides
    d1_ain1.off(); d1_ain2.on(); d1_bin1.on(); d1_bin2.off()
    d2_ain1.on(); d2_ain2.off(); d2_bin1.off(); d2_bin2.on()
    threading.Thread(target=stop_after, args=(duration,)).start()

def shallow_till(duration=0.25):
    print("Shallow till start")
    stby.on()
    tiller_pwm.value = 0.5

    # Move tiller down briefly
    d3_ain1.on()
    d3_ain2.off()
    sleep(0.2)  # pulse duration for tiller
    d3_ain1.off()
    d3_ain2.off()
    tiller_pwm.value = 0

    # Then move wheels forward using existing forward() function
    forward(duration)

def deep_till(duration=0.5):
    print("Deep till start")
    stby.on()
    tiller_pwm.value = 0.5

    # Move tiller down longer if needed
    d3_ain1.on()
    d3_ain2.off()
    sleep(0.5)  # longer pulse for deep till
    d3_ain1.off()
    d3_ain2.off()
    tiller_pwm.value = 0

    # Then move wheels forward
    forward(duration)


def run_pump(duration=2):
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
