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
from PoseEstimator import PoseEstimator, AnnoPoses
from PoseEstimatorClient import PoseEstimatorClient

import pyrealsense2 as rs

def main():
    host = config.CSIE_SERVER_HOST
    port = config.PORT

    pec = PoseEstimatorClient(host, port)

    rs_pipe = rs.pipeline()
    rs_cfg  = rs.config()

    rs_cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

    rs_pipe.start(rs_cfg)

    try:
        while True:
            frame = rs_pipe.wait_for_frames()

            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()

            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            poses, scores = pec.Estimate(img)
            anno_img = AnnoPoses(img, poses, scores)

            anno_frame = cv.cvtColor(anno_img, cv.COLOR_RGB2BGR)

            cv.imshow("anno_frame", anno_frame)

            if cv.waitKey(50) & 0xff == ord("q"):
                break
    except Exception as e:
        print(f"exception message = {e}")

if __name__ == "__main__":
    main()
