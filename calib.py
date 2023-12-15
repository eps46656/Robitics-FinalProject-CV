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

        # cv.imshow("img", img)
        # cv.waitKey(0)

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

    obj_points = np.stack(obj_points, axis=0).astype(np.float64)
    img_points = np.stack(img_points, axis=0).astype(np.float64)

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
        print(f"img.shape = {img.shape}")
        print(f"cur_obj_points.shape = {cur_obj_points.shape}")
        print(f"cur_img_points.shape = {cur_img_points.shape}")
        print(f"camera_mat = {camera_mat}")
        print(f"camera_distort = {camera_distort}")
        success, rvec, tvec = cv.solvePnP(
            objectPoints=cur_obj_points,
            imagePoints=cur_img_points[:, None, :],
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
        # method=cv.CALIB_HAND_EYE_TSAI
        # method=cv.CALIB_HAND_EYE_PARK
        method=cv.CALIB_HAND_EYE_HORAUD
        # method=cv.CALIB_HAND_EYE_ANDREFF
        # method=cv.CALIB_HAND_EYE_DANIILIDIS
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

    target_Cpose = [[227, -412, 177, 90, 0, -60],
                    [284, -449, 273, 84, 10, -60],
                    [249.45, -398.59, 265.26, 89.45, -1.52, -79.46],
                    [265, -451, 422, 90, -4, -34],
                    [201, -437, 443, 92, -7, -54],
                    [122, -450, 508, 90, -15, -45],
                    [174.59, -502.0, 486.36, 89.41, 25.81, -45.86],
                    [146.82, -434.45, 502.51, 76.4, 17.13, -88.37],]

    img_num = len(target_Cpose)

    Ts_base_to_gripper = list()

    imgs = list()

    for i in range(img_num):
        if i == 3:
            continue

        if i == 5:
            continue

        if i == 6:
            continue

        imgs.append(ReadImage(f"{DIR}/img_{i}.png"))

        pose = target_Cpose[i]

        T = np.identity(4)
        T = GetHomoRotMat([1, 0, 0], pose[3] * GLOBAL.DEG_TO_RAD) @ T
        T = GetHomoRotMat([0, 1, 0], pose[4] * GLOBAL.DEG_TO_RAD) @ T
        T = GetHomoRotMat([0, 0, 1], pose[5] * GLOBAL.DEG_TO_RAD) @ T
        T = GetHomoTransMat(pose[0:3]) @ T

        T = np.linalg.inv(T)

        Ts_base_to_gripper.append(T.copy())

    Ts_base_to_gripper = np.stack(Ts_base_to_gripper, axis=0)

    T_camera_to_base = HandEyeCalib(
        camera_mat, camera_distort,
        Ts_base_to_gripper,
        imgs)

    print("T_camera_to_base")
    print(T_camera_to_base)

    NPSave(f"{DIR}/T_camera_to_base.npy", T_camera_to_base)

if __name__ == "__main__":
    main()
