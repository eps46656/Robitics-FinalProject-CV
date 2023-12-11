import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config

from queue import Queue
from collections import deque
import cv2 as cv
import socket

import numpy as np

from Server import *
from SocketBuffer import *
from AnnoPoses import AnnoPoses
from PoseEstimatorClient import PoseEstimatorClient

def main():
    host = socket.gethostname()
    port = config.PORT

    pec = PoseEstimatorClient(host, port)

    H = 480
    W = 640

    try:
        while True:
            poses, scores = pec.Estimate()

            img = np.zeros((H, W, 3), dtype=np.uint8)

            anno_img = AnnoPoses(img, poses, scores)

            anno_frame = cv.cvtColor(anno_img, cv.COLOR_RGB2BGR)

            cv.imshow("anno_frame", anno_frame)

            if cv.waitKey(50) & 0xff == ord("q"):
                break
    except Exception as e:
        print(f"exception message = {e}")

if __name__ == "__main__":
    main()
