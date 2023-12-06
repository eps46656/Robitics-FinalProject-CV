import os

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import sys

import config
import socket

from Server import *
from PoseEstimator import PoseEstimator

from SocketBuffer import *

import cv2 as cv

rgb_frames_tmp = list()

class PoseEstimatorServer:
    def __init__(self, host, port):
        self.server = Server(host, port, self.SessionFunc)
        self.pe = PoseEstimator()

    def Start(self):
        self.server.Start()

    def Stop(self):
        self.server.Stop()

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

            rgb_frames_tmp.append(img)

        end_callback()

    def SessionFunc(self, _, conn, addr, end_callback):
        sb_byte = SocketBufferByte(conn)
        sb_arr = SocketBufferArr(sb_byte)
        sb_img = SocketBufferImage(sb_byte)

        while self.server.is_active():
            try:
                img = sb_img.Recv(time.time() + 20 / 1000)
            except TimeoutException:
                continue
            except Exception as e:
                print(f"exception when receiving image: {e}")
                break

            poses, scores = self.pe.Estimate(img)

            try:
                sb_arr.Send(poses)
                sb_arr.Send(scores)
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
    host = sys.argv[1]
    port = int(sys.argv[2])

    pose_es = PoseEstimatorServer(host, port)

    pose_es.Start()

    try:
        input("")
    except:
        pass

    pose_es.Stop()

if __name__ == "__main__":
    main()
