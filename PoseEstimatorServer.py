import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import cv2 as cv
import socket

from Server import *
from PoseEstimator import PoseEstimator
from SocketBuffer import *

class PoseEstimatorServer:
    def __init__(self, host, port):
        self.server = Server(host, port, self.SessionFunc)
        self.pe = PoseEstimator()
        self.poses = None
        self.scores = None

    def Start(self):
        self.server.Start()

    def Stop(self):
        self.server.Stop()

    def Estimate(self, img):
        self.poses, self.scores = self.pe.Estimate(img)
        print(f"PoseEstimatorServer: Estimate: {time.time()}")

    def SessionFuncD(self, _, conn, addr, end_callback):
        sb_byte = SocketBufferByte(conn)
        sb_arr = SocketBufferArr(sb_byte)

        while self.server.is_active():
            try:
                arr = sb_arr.Recv(time.time() + 20 / 1000)
            except TimeoutException:
                continue
            except Exception as e:
                print(f"exception message: {e}")
                break

            print(f"recved arr")

            sb_arr.Send(arr)

        end_callback()

    def SessionFuncC(self, _, conn, addr, end_callback):
        sb_byte = SocketBufferByte(conn)
        sb_arr = SocketBufferArr(sb_byte)
        sb_img = SocketBufferImage(sb_byte)

        while self.server.is_active():
            try:
                img = sb_img.Recv(time.time() + 20 / 1000)
            except TimeoutException:
                continue
            except Exception as e:
                print(f"exception message: {e}")
                break

            print(f"recved img")

            # rgb_frames_tmp.append(img)

        end_callback()

    def SessionFunc(self, _, conn, addr, end_callback):
        sb_byte = SocketBufferByte(conn)
        sb_arr = SocketBufferArr(sb_byte)

        while self.server.is_active():
            try:
                sb_byte.Recv(1, None)
                print(f"ajbkaodw")
            except Exception as e:
                print(f"exception when sending poses and scores: {e}")
                break

            try:
                sb_arr.Send(self.poses)
                sb_arr.Send(self.scores)
            except Exception as e:
                print(f"exception when sending poses and scores: {e}")
                break

        end_callback()

    def SessionFuncB(self, _, conn, addr, end_callback):
        ss = SocketBufferByte(conn)

        while self.server.is_active():
            try:
                x = int.from_bytes(
                    ss.Recv(8, time.time() + 20 / 1000), "little")
                # x = int.from_bytes(conn.recv(8), "little")
                print(f"{addr}: received {x}")

                if x == 0:
                    break
            except TimeoutException:
                continue
            except Exception as e:
                print(f"exception message: {e}")
                break

        end_callback()

    def SessionFuncA(self, _, conn, addr, end_callback):
        conn.settimeout(2)

        while self.server.is_active():
            try:
                x = int.from_bytes(conn.recv(1024), "little")
                print(f"{addr}: received {x}")

                if x == 0:
                    break
            except socket.timeout:
                continue
            except Exception as e:
                print(f"exception message: {e}")
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
