#!/home/whcheng/.pyenv/shims/python
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import numpy as np
import cv2 as cv
import math

import glob

from scipy.spatial.transform import Rotation as scipy_Rotation

from utils import *


GLOBAL = EMPTY_CLASS()

GLOBAL.chessboard = EMPTY_CLASS()

GLOBAL.chessboard.img_h = 480
GLOBAL.chessboard.img_w = 640
GLOBAL.chessboard.pattern_h = 6
GLOBAL.chessboard.pattern_w = 8
GLOBAL.chessboard.grid_len = 25

GLOBAL.RAD_TO_DEG = 180 / math.pi
GLOBAL.DEG_TO_RAD = math.pi / 180



def PointsToHomoPoints(points):
    # points[P, N]

    P, N = points.shape

    ret = np.empty((P+1, N), dtype=points.dtype)
    ret[:P, :] = points
    ret[P, :] = 1

    return ret

def FindCornerPoints(imgs):
    per_obj_points = np.empty(
        (GLOBAL.chessboard.pattern_h * GLOBAL.chessboard.pattern_w, 3),
        dtype=np.float32)
    per_obj_points[:, :2] = \
        np.mgrid[:GLOBAL.chessboard.pattern_h, :GLOBAL.chessboard.pattern_w] \
        .transpose().reshape((-1, 2)) * GLOBAL.chessboard.grid_len
    per_obj_points[:, 2] = 0

    obj_points = list()
    img_points = list()

    for img in imgs:
        img = img.copy()

        cv.imshow("img", img)
        cv.waitKey(0)

        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        ret, corners = cv.findChessboardCorners(
            gray,
            (GLOBAL.chessboard.pattern_h, GLOBAL.chessboard.pattern_w),
            None)

        if not ret:
            continue

        obj_points.append(per_obj_points)

        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 0.001)).reshape((-1, 2))

        img = cv.drawChessboardCorners(
            img, (GLOBAL.chessboard.pattern_h, GLOBAL.chessboard.pattern_w),
            corners, ret)

        cv.imshow("checkerboard img", img)
        cv.waitKey(0)

        img_points.append(corners)

    obj_points = np.stack(obj_points, axis=0, dtype=np.float32)
    img_points = np.stack(img_points, axis=0, dtype=np.float32)

    return obj_points, img_points

def CameraCalib(imgs):
    obj_points, img_points = FindCornerPoints(imgs)

    ret, camera_mat, camera_distort, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points,
        (GLOBAL.chessboard.img_w, GLOBAL.chessboard.img_h),
        None, None)

    return camera_mat, camera_distort

def HandEyeCalib(camera_mat, camera_distort, Ts_base_to_gripper, imgs):
    obj_points, img_points = FindCornerPoints(imgs)

    Ts_obj_to_camera = list()

    for T_base_to_gripper, img, cur_obj_points, cur_img_points \
    in zip(Ts_base_to_gripper, imgs, obj_points, img_points):
        success, rvec, tvec = cv.solvePnP(
            objectPoints=cur_obj_points,
            imagePoints=cur_img_points,
            cameraMatrix=camera_mat,
            distCoeffs=camera_distort,
            flags=cv.SOLVEPNP_EPNP)

        T = np.identity(4)
        T[:3, :3] = cv.Rodrigues(rvec)[0].reshape((3, 3))
        T[:3, 3] = tvec.reshape(-1)

        Ts_obj_to_camera.append(T)

    Ts_obj_to_camera = np.stack(Ts_obj_to_camera, axis=0)

    R, t = cv.calibrateHandEye(
        R_gripper2base=Ts_base_to_gripper[:, :3, :3],
        t_gripper2base=Ts_base_to_gripper[:, :3, 3],
        R_target2cam=Ts_obj_to_camera[:, :3, :3],
        t_target2cam=Ts_obj_to_camera[:, :3, 3],
        # method=cv.CALIB_HAND_EYE_DANIILIDIS
        method=cv.CALIB_HAND_EYE_TSAI
    )

    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(-1)

    T_camera_to_base = T

    return T_camera_to_base

def main():
    camera_mat, camera_distort = None, None

    if False:
        camera_mat, camera_distort = CameraCalib(
            [ReadImg(filename)
            for filename in \
                glob.glob(f"{DIR}/camera_calib_images/frame-*_Color.png")])
    else:
        camera_params = NPLoad(f"{DIR}/camera_params.npy").item()
        camera_mat = camera_params["camera_mat"]
        camera_distort = camera_params["camera_distort"]

    print("camera_mat")
    print(camera_mat)

    print("camera_distort")
    print(camera_distort)

    NPSave(f"{DIR}/camera_params", {"camera_mat": camera_mat,
                                    "camera_distort": camera_distort,})

    rot = np.array([90, 0, -60])
    base_loc_beg = np.array([227, -412, 177])
    base_loc_end = np.array([160, -381, 551.8])

    Ts_base_to_gripper = list()

    for t in np.linspace(0, 1, 9):
        loc = base_loc_beg * (1 - t) + base_loc_end * t

        T = np.identity(4)
        T = GetHomoRotMat([1, 0, 0], rot[0] * GLOBAL.DEG_TO_RAD) @ T
        T = GetHomoRotMat([0, 1, 0], rot[1] * GLOBAL.DEG_TO_RAD) @ T
        T = GetHomoRotMat([0, 0, 1], rot[2] * GLOBAL.DEG_TO_RAD) @ T
        T = GetHomoTransMat(loc) @ T

        T = np.linalg.inv(T)

        Ts_base_to_gripper.append(T.copy())

    Ts_base_to_gripper = np.stack(Ts_base_to_gripper, axis=0)

    T_camera_to_base = HandEyeCalib(
        camera_mat, camera_distort,
        Ts_base_to_gripper,
        [ReadImg(f"{DIR}/hand_eye_calib_images/img_{i}.png")
         for i in range(9)])

    print("T_camera_to_base")
    print(T_camera_to_base)

if __name__ == "__main__":
    main()
