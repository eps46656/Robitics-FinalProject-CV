import config
import socket
import cv2 as cv

class PoseEstimatorClient:
    def __init__(self, addr, port):
        self.s = socket.socket()
        self.s.connect((addr, port))

    def Estimate(self, img):
        x = int(input("x: "))
        print(f"out x: {x}")
        self.s.send(x.to_bytes(8, "little"))

def main():
    s = socket.socket()
    host = socket.gethostname()
    port = config.PORT
    s.connect((host, port))

    while True:
        x = int(input("x: "))
        print(f"out x: {x}")
        s.send(x.to_bytes(8, "little"))

        if x == 0:
            break

    s.close()

if __name__ == "__main__":
    main()
