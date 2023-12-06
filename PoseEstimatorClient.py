import os

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import socket
import cv2 as cv
from SocketBuffer import *

class PoseEstimatorClient:
    def __init__(self, addr, port):
        self.s = socket.socket()
        self.s.connect((addr, port))

        self.sb_byte = SocketBufferByte(self.s)
        self.sb_arr = SocketBufferArr(self.sb_byte)
        self.sb_img = SocketBufferImage(self.sb_byte)

    def Estimate(self, img):
        self.sb_img.Send(img)

        poses = self.sb_arr.Recv()
        scores = self.sb_arr.Recv()

        return poses, scores

def main():
    s = socket.socket()
    host = socket.gethostname()
    port = config.PORT
    s.connect((host, port))

    sb_byte = SocketBufferByte(s)
    sb_img = SocketBufferImage(sb_byte)

    print(__file__)
    print(DIR)
    print(f"{DIR}/anno_v.mp4")

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
