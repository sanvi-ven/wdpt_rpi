# import socket
# from gpiozero import OutputDevice
# from time import sleep

# # ----- MOTOR SETUP -----
# stby = OutputDevice(12)
# stby.on()   # Enable motor drivers

# # DRIVER 1
# d1_ain1 = OutputDevice(16)
# d1_ain2 = OutputDevice(25)
# d1_bin1 = OutputDevice(5)
# d1_bin2 = OutputDevice(6)

# # DRIVER 2
# d2_ain1 = OutputDevice(24)
# d2_ain2 = OutputDevice(23)
# d2_bin1 = OutputDevice(27)
# d2_bin2 = OutputDevice(22)

# # DRIVER 3: TILLER
# d3_ain1 = OutputDevice(4)
# d3_ain2 = OutputDevice(17)

# def forward(duration=2):
#     # Driver 1
#     d1_ain1.on()
#     d1_ain2.off()
#     d1_bin1.off()
#     d1_bin2.on()
#     # Driver 2
#     d2_ain1.off()
#     d2_ain2.on()
#     d2_bin1.on()
#     d2_bin2.off()
#     sleep(duration)
#     stop()

# def shallow_till(duration=2):
#     d3_ain1.on()
#     d3_ain2.off()
#     sleep(duration)
#     d3_ain1.off()
#     d3_ain2.off()

# def deep_till(duration=4):
#     d3_ain1.on()
#     d3_ain2.off()
#     sleep(duration)
#     d3_ain1.off()
#     d3_ain2.off()

# def stop():
#     for m in [d1_ain1,d1_ain2,d1_bin1,d1_bin2,d2_ain1,d2_ain2,d2_bin1,d2_bin2,d3_ain1,d3_ain2]:
#         m.off()

# # ----- SOCKET SERVER -----
# HOST = ''  # listen on all interfaces
# PORT = 5005
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(1)
# print("Motor server running, waiting for Mac connection...")

# conn, addr = s.accept()
# print('Connected by', addr)

# try:
#     while True:
#         data = conn.recv(1024).decode().strip()
#         if not data: break
#         print("Command received:", data)
#         if data == "forward":
#             forward()
#         elif data == "shallow_till":
#             shallow_till()
#         elif data == "deep_till":
#             deep_till()
#         elif data == "no_till":
#             stop()
# finally:
#     conn.close()
#     stop()
#     stby.off()



import socket
from gpiozero import OutputDevice, PWMOutputDevice
from time import sleep

# ---------- MOTOR SETUP ----------
stby = OutputDevice(12)
stby.on()
pwm = PWMOutputDevice(18)
pwm.value = 0.5

# DRIVER 1
d1_ain1 = OutputDevice(16)
d1_ain2 = OutputDevice(25)
d1_bin1 = OutputDevice(5)
d1_bin2 = OutputDevice(6)
# DRIVER 2
d2_ain1 = OutputDevice(24)
d2_ain2 = OutputDevice(23)
d2_bin1 = OutputDevice(27)
d2_bin2 = OutputDevice(22)
# DRIVER 3: Tiller
d3_ain1 = OutputDevice(4)
d3_ain2 = OutputDevice(17)

# ---------- MOTOR COMMANDS ----------
def move_forward(duration=2):
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

def shallow_till(duration=0.5):
    d3_ain1.on()
    d3_ain2.off()
    sleep(duration)
    d3_ain1.off()
    d3_ain2.off()

def deep_till(duration=1):
    d3_ain1.on()
    d3_ain2.off()
    sleep(duration)
    d3_ain1.off()
    d3_ain2.off()

def stop():
    d1_ain1.off()
    d1_ain2.off()
    d1_bin1.off()
    d1_bin2.off()
    d2_ain1.off()
    d2_ain2.off()
    d2_bin1.off()
    d2_bin2.off()
    d3_ain1.off()
    d3_ain2.off()
    pwm.off()
    stby.off()

# ---------- SOCKET SERVER ----------
HOST = "0.0.0.0"
PORT = 5005

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
print(f"Motor server listening on {PORT}...")

conn, addr = s.accept()
print(f"Connected by {addr}")

try:
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print(f"Received: {data}")
        if data == "NO_TILL":
            stop()
        elif data == "SHALLOW_TILL":
            move_forward(2)
            shallow_till()
        elif data == "DEEP_TILL":
            move_forward(2)
            deep_till()
finally:
    stop()
    conn.close()
    s.close()
