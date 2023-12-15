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
from PoseEstimatorServerRS import PoseEstimatorServerRS

import pyrealsense2 as rs

def main():
    img_h = 480
    img_w = 640
    fps = 30

    host = "10.42.0.1"
    port = config.PORT

    pes = PoseEstimatorServerRS(host, port)

    pe = PoseEstimator()

    rs_pipe = rs.pipeline()
    rs_cfg  = rs.config()

    rs_cfg.enable_stream(rs.stream.color, img_w, img_h, rs.format.bgr8, fps)
    rs_cfg.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16, fps)

    rs_pipe.start(rs_cfg)

    camera_params = NPLoad(f"{DIR}camera_params.npy").items()
    camera_mat = camera_params["camera_mat"]
    camera_distort = camera_params["camera_distort"]

    inv_camera_mat = np.linalg.inv(camera_mat)

    T_camera_to_base = NPLoad(f"{DIR}/T_camera_to_base.npy")

    try:
        while True:
            frame = rs_pipe.wait_for_frames()

            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()

            depth_scale = depth_frame.get_units() # m / unit
            scale = 100 * scale # mm / unit

            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = np.asanyarray(depth_frame.get_data())

            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            depth = (depth_frame * scale).astype(np.float32)

            poses, scores = pe.Estimate(img)
            pose = poses[scores.argmax()]
            # [17, 3]

            pose_3d = np.empty((17, 3), np.float32)

            for k in range(17):
                pose_w = pose[k][0] * img_w
                pose_h = pose[k][1] * img_h

                d = depth[max(0, min(int(round(pose_w)), img_w-1)),
                          max(0, min(int(round(pose_h)), img_h-1))]

                x = np.array([pose_w * d, pose_h * d, d]).reshape((3, 1))
                x = (inv_camera_mat @ x).reshape(-1)
                x = np.array([x[0], x[1], x[2], 1]).reshape((3, 1))
                x = (T_camera_to_base @ x).reshape(-1)

                pose_3d[k, :] = x[:3]

            pes.UpdatePose(pose_3d)

            if cv.waitKey(50) & 0xff == ord("q"):
                break
    except Exception as e:
        print(f"exception message = {e}")

if __name__ == "__main__":
    main()
