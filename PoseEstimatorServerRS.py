import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import cv2 as cv
import socket
import threading

from Server import *
from PoseEstimator import PoseEstimator
from SocketBuffer import *

class PoseEstimatorServerRS:
    def __init__(self, host, port):
        self.server = Server(host, port, self.SessionFunc)
        self.pe = PoseEstimator()
        self.pose = None

        self.lock = threading.Lock()

    def Start(self):
        self.server.Start()

    def Stop(self):
        self.server.Stop()

    def UpdatePose(self, pose):
        self.lock.acquire()

        self.pose = pose.copy()

        self.lock.release()

    def SessionFunc(self, _, conn, addr, end_callback):
        sb_byte = SocketBufferByte(conn)
        sb_arr = SocketBufferArr(sb_byte)

        while self.server.is_active():
            try:
                sb_byte.Recv(1, None)
            except Exception as e:
                print(f"exception when sending poses and scores: {e}")
                break

            try:
                self.lock.acquire()
                sb_arr.Send(self.pose.copy())
                self.lock.release()
            except Exception as e:
                print(f"exception when sending poses and scores: {e}")
                break

        end_callback()

def main():
    print(sys.argv)

    host = socket.gethostname()
    port = int(sys.argv[2])

    pose_es = PoseEstimatorServer(host, port)

    pose_es.Start()

    try:
        input()
    except:
        pass

    pose_es.Stop()

if __name__ == "__main__":
    main()
