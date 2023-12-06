import os

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import sys

from queue import Queue
from collections import deque
import cv2 as cv
import socket
from Server import *
from SocketBuffer import *
from PoseEstimator import PoseEstimator, AnnoPoses

def main():
    print(f"{sys.argv}")

    s = socket.socket()
    host = sys.argv[1]
    port = int(sys.argv[2])
    s.connect((host, port))

    sb_byte = SocketBufferByte(s)
    sb_image = SocketBufferImage(sb_byte)
    sb_arr = SocketBufferArr(sb_byte)

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if True:
            sb_image.Send(img)

            poses = sb_arr.Recv(time.time() + 1000)
            scores = sb_arr.Recv(time.time() + 1000)

            anno_img = AnnoPoses(img, poses, scores)
        else:
            anno_img = img

        anno_frame = cv.cvtColor(anno_img, cv.COLOR_RGB2BGR)

        cv.imshow("anno_frame", anno_frame)

        if cv.waitKey(50) & 0xff == ord("q"):
            break

    s.close()

if __name__ == "__main__":
    main()
