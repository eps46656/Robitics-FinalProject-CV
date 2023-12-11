import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config

from queue import Queue
from collections import deque
import cv2 as cv
import socket

from Server import *
from SocketBuffer import *
from PoseEstimatorServer import PoseEstimatorServer

def main():
    host = socket.gethostname()
    port = config.PORT

    pe_server = PoseEstimatorServer(host, port)

    cap = cv.VideoCapture(0)

    pe_server.Start()

    try:
        while True:
            ret, frame = cap.read()

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            pe_server.Estimate(img)

            time.sleep(1/ 20)
    except Exception as e:
        print(f"exception message = {e}")

    pe_server.Stop()

if __name__ == "__main__":
    main()
