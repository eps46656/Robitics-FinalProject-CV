import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import cv2 as cv
import socket

from PoseEstimator import PoseEstimator
from SocketBuffer import *

def main():
    host = config.CSIE_SERVER_HOST
    port = config.PORT

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    csie_sb_byte = SocketBufferByte(s)
    csie_sb_arr = SocketBufferArr(csie_sb_byte)
    csie_sb_img = SocketBufferImage(csie_sb_byte)

    pe = PoseEstimator()

    print(f"pe node is OK")

    while True:
        print(f"receiving img from csie node")

        try:
            img = csie_sb_img.Recv(None)
        except Exception as e:
            print(f"exception message: {e}")
            break

        print(f"received img from csie node")

        poses, scores = pe.Estimate(img)

        print(f"sending poses and scores to csie node")

        try:
            csie_sb_arr.Send(poses)
            csie_sb_arr.Send(scores)
        except Exception as e:
            print(f"exception message: {e}")
            break

if __name__ == "__main__":
    main()
