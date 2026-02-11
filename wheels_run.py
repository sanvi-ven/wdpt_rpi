from gpiozero import OutputDevice, PWMOutputDevice
from time import sleep

# ----- STANDBY -----
stby = OutputDevice(12)
stby.on()   # Enable motor drivers

# ----- PWM (shared) -----
pwm = PWMOutputDevice(18)
pwm.value = 0.5   # 50% speed (0 to 1)

# ----- DRIVER 1 -----
d1_ain1 = OutputDevice(16)
d1_ain2 = OutputDevice(25)
d1_bin1 = OutputDevice(5)
d1_bin2 = OutputDevice(6)

# ----- DRIVER 2 -----
d2_ain1 = OutputDevice(24)
d2_ain2 = OutputDevice(23)
d2_bin1 = OutputDevice(27)
d2_bin2 = OutputDevice(22)

def forward():
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

def stop():
    d1_ain1.off()
    d1_ain2.off()
    d1_bin1.off()
    d1_bin2.off()

    d2_ain1.off()
    d2_ain2.off()
    d2_bin1.off()
    d2_bin2.off()

# ---- Run ----
forward()
sleep(2)   # move forward for 2 seconds
stop()

pwm.off()
stby.off()
