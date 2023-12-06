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

import pyrealsense2 as rs

def main():
    print(f"{sys.argv}")

    s = socket.socket()
    host = sys.argv[1]
    port = int(sys.argv[2])
    s.connect((host, port))

    sb_byte = SocketBufferByte(s)
    sb_image = SocketBufferImage(sb_byte)
    sb_arr = SocketBufferArr(sb_byte)

    rs_pipe = rs.pipeline()
    rs_cfg  = rs.config()

    rs_cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

    rs_pipe.start(rs_cfg)

    while True:
        frame = rs_pipe.wait_for_frames()

        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        img = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)

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
