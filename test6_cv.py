import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config

from queue import Queue
from collections import deque
import cv2 as cv
import socket

from utils import *
from Server import *
from SocketBuffer import *
from PoseEstimator import PoseEstimator
from PoseEstimatorClient import PoseEstimatorClient

def main():
    pe = PoseEstimator()

    cap = cv.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            poses, scores = pe.Estimate(img)
            anno_img = AnnoPoses(img, poses, scores)

            anno_frame = cv.cvtColor(anno_img, cv.COLOR_RGB2BGR)

            cv.imshow("anno_frame", anno_frame)

            if cv.waitKey(50) & 0xff == ord("q"):
                break
    except Exception as e:
        print(f"exception message = {e}")

if __name__ == "__main__":
    main()
