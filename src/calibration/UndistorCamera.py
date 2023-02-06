import os
import cv2
import yaml
import glob
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

parser = argparse.ArgumentParser()
parser.add_argument("-yaml", "--yaml_file", help="Path of yaml file", type=str)
args = parser.parse_args()

# =================檢查 Yaml file 是否存在 ====================

yaml_path = args.yaml_file
assert os.path.isfile(yaml_path), "Yaml File Not Exist!!"
with open(yaml_path, "r") as stream:
    yaml_data = yaml.load(stream)

# ==================讀取Yaml內參數=============================
FILE_PATH = yaml_data['Calibration']['FILE_PATH']
NUM_IMAGES = 90
IMAGE_SHAPE = [640, 400]
NEW_IMAGE_SHAPE = [640, 400]
FIND_ALGO = "findChessboardCornersSB"
DISTORTION_MODEL = "Fisheye"
USE_DEFAULT_RECTIFY = True
# LEFT CAMERA: 0.15168719955162607
LEFT_INTRINSIC = [278.08321774513394, 279.113841127447, 309.19580091877737, 207.67952497223772, 0.0]
LEFT_DISTORTION = [-0.020306250909190218, 0.056496479357930875, -0.06768354885591037, 0.036626224288176815]
LEFT_QUATERNION = [0.00014795886006828917, -0.18390070023464902, -0.05418164685810092, 0.9814503857583049]
LEFT_PROJECTION = [685.5153012854004,685.5153012854004,318.80980587005615,160.96755317687987]

# ==================讀取Yaml內參數=============================
if FIND_ALGO == "findCirclesGrid":
    FIND_PATTERN_VERSION = 0
elif FIND_ALGO == "findChessboardCorners":
    FIND_PATTERN_VERSION = 1
elif FIND_ALGO == "findChessboardCornersSB":
    FIND_PATTERN_VERSION = 2
else:
    assert False, 'Do not support {}'.format(FIND_ALGO)

K1 = np.array([[ LEFT_INTRINSIC[0], LEFT_INTRINSIC[4], LEFT_INTRINSIC[2]],
               [                0., LEFT_INTRINSIC[1], LEFT_INTRINSIC[3]],
               [                0.,                0.,                1.]])
D1 = np.array(LEFT_DISTORTION)
R1 = Rot.from_quat(LEFT_QUATERNION).as_matrix()
P1 = np.array([[ LEFT_PROJECTION[0],                 0., LEFT_PROJECTION[2], 0.],
               [                 0., LEFT_PROJECTION[1], LEFT_PROJECTION[3], 0.],
               [                 0.,                 0.,                 1., 0.]])

IMAGE_WIDTH = IMAGE_SHAPE[0]
IMAGE_HEIGHT = IMAGE_SHAPE[1]
USE_CIRCLES_GRID = FIND_PATTERN_VERSION == 2
PATTERN_TYPE = "CirclesGrid" if USE_CIRCLES_GRID else "Checkerboard"
USE_FISHEYE_API = DISTORTION_MODEL == 'Fisheye'

def Show():
    cv2.namedWindow('original left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('undistort left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rectify left', cv2.WINDOW_NORMAL)

    image_paths = sorted(glob.glob(FILE_PATH))
    undistorted_fn = cv2.fisheye.undistortImage if USE_FISHEYE_API else cv2.undistort
    rectify_fn = cv2.fisheye.initUndistortRectifyMap if not USE_DEFAULT_RECTIFY else cv2.initUndistortRectifyMap
    mapx1, mapy1 = rectify_fn(K1, D1, R1, P1, NEW_IMAGE_SHAPE, cv2.CV_16SC2)

    undistorted_fn = cv2.fisheye.undistortImage if USE_FISHEYE_API else cv2.undistort
    rectify_fn = cv2.fisheye.initUndistortRectifyMap if not USE_DEFAULT_RECTIFY else cv2.initUndistortRectifyMap
    mapx1, mapy1 = rectify_fn(K1, D1, R1, P1, NEW_IMAGE_SHAPE, cv2.CV_16SC2)

    for fname in image_paths:
        print(fname)
        imgL = cv2.imread(fname)
        imgL_undisto = undistorted_fn(imgL, K1, D1, Knew=K1)
        imgL_rectify = cv2.remap(imgL, mapx1, mapy1, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow('original left', imgL)
        cv2.imshow('undistort left', imgL_undisto)
        cv2.imshow('rectify left', imgL_rectify)

        while True:
            key = cv2.waitKey(10)
            if key == ord('n'):
                break
            elif key == ord('q'):
                return

# print('K1 \n {}'.format(K1))
# print('D1 \n {}'.format(D1))
# print('R1 \n {}'.format(R1))
# print('P1 \n {}'.format(P1))
# Show()