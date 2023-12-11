import os

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import cv2 as cv
import socket

from SocketBuffer import *

class PoseEstimatorClient:
    def __init__(self, addr, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((addr, port))

        self.sb_byte = SocketBufferByte(self.s)
        self.sb_arr = SocketBufferArr(self.sb_byte)
        self.sb_img = SocketBufferImage(self.sb_byte)

    def Estimate(self):
        self.sb_byte.Send(int(0).to_bytes(1, "little"))

        poses = self.sb_arr.Recv(None)
        scores = self.sb_arr.Recv(None)

        return poses, scores

def main():
    s = socket.socket()
    host = socket.gethostname()
    port = config.PORT
    s.connect((host, port))

    sb_byte = SocketBufferByte(s)
    sb_img = SocketBufferImage(sb_byte)

    rgb_in = cv.VideoCapture(f"{DIR}/anno_v.mp4")

    while True:
        ret, rgb_frame = rgb_in.read()

        if not ret:
            break

        print(f"send")
        sb_img.Send(rgb_frame)

    print(f"EOF")

    s.close()

if __name__ == "__main__":
    main()
