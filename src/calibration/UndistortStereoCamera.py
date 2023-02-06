import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation as Rot

###############################################################################
### Copy Paste
###############################################################################
# findChessboardCornersSB + findChessboardCornersSB
FILE_PATH = "{}_6DoF_EVT_PD03/{:08d}.png"
NUM_IMAGES = 90
IMAGE_SHAPE = [640, 400]
NEW_IMAGE_SHAPE = [640, 400]
FIND_ALGO = "findChessboardCornersSB"
DISTORTION_MODEL = "Fisheye"
USE_DEFAULT_RECTIFY = True
# LEFT CAMERA: 0.15168719955162607
LEFT_INTRINSIC = [278.08321774513394, 279.113841127447, 309.19580091877737, 207.67952497223772, 0.0]
LEFT_DISTORTION = [-0.020306250909190218, 0.056496479357930875, -0.06768354885591037, 0.036626224288176815]
# RIGHT CAMERA: 0.32748615807380177
RIGHT_INTRINSIC = [278.94802029213525,280.06413943948945,323.82576383382235,189.79463491445,0.0]
RIGHT_DISTORTION = [-0.008572059203577632, 0.030199938052149225, -0.02935876766158547, 0.011172720793932037]
# N_OK 19
# STEREO CAMERA: 0.17819081837645334
# R Angle [  5.83703274 -41.88027972 -14.88965658]
# T [-105.5723082    11.29983573  -40.86152588]
# validROI1 (0, 0, 640, 352)
# validROI2 (0, 0, 640, 352)
# Quaternion Format is (x, y, z, w)
# LEFT Angle [  1.24228059 -21.15933621  -6.55173034]
LEFT_QUATERNION = [0.00014795886006828917, -0.18390070023464902, -0.05418164685810092, 0.9814503857583049]
# RIGHT Angle [ 1.11765387 21.05103554  6.10273377]
RIGHT_QUATERNION = [-0.00014795886006828944, 0.18291619025332448, 0.05055392386298552, 0.9818278597765704]
BASELINE = 113.76670358072406
LEFT_PROJECTION = [685.5153012854004,685.5153012854004,318.80980587005615,160.96755317687987]
RIGHT_PROJECTION = [685.5153012854004,685.5153012854004,318.80980587005615,160.96755317687987]
###############################################################################

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
K2 = np.array([[ RIGHT_INTRINSIC[0], RIGHT_INTRINSIC[4], RIGHT_INTRINSIC[2]],
               [                 0., RIGHT_INTRINSIC[1], RIGHT_INTRINSIC[3]],
               [                 0.,                 0.,                 1.]])
D2 = np.array(RIGHT_DISTORTION)
R2 = Rot.from_quat(RIGHT_QUATERNION).as_matrix()
P2 = np.array([[ RIGHT_PROJECTION[0],                  0., RIGHT_PROJECTION[2], -BASELINE*RIGHT_PROJECTION[0]],
               [                  0., RIGHT_PROJECTION[1], RIGHT_PROJECTION[3], 0.],
               [                  0.,                  0.,                  1., 0.]])
IMAGE_WIDTH = IMAGE_SHAPE[0]
IMAGE_HEIGHT = IMAGE_SHAPE[1]
USE_CIRCLES_GRID = FIND_PATTERN_VERSION == 0
PATTERN_TYPE = "CirclesGrid" if USE_CIRCLES_GRID else "Checkerboard"
USE_FISHEYE_API = DISTORTION_MODEL == 'Fisheye'

def Show():
    cv2.namedWindow('original left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('undistort left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rectify left', cv2.WINDOW_NORMAL)
    cv2.namedWindow('original right', cv2.WINDOW_NORMAL)
    cv2.namedWindow('undistort right', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rectify right', cv2.WINDOW_NORMAL)
    
    lxx, rxx = 0, IMAGE_WIDTH
    yy, ww, hh = 0, IMAGE_WIDTH, IMAGE_HEIGHT
    image_paths = [ FILE_PATH.format(PATTERN_TYPE, n) for n in range(NUM_IMAGES) ]
    undistorted_fn = cv2.fisheye.undistortImage if USE_FISHEYE_API else cv2.undistort
    rectify_fn = cv2.fisheye.initUndistortRectifyMap if not USE_DEFAULT_RECTIFY else cv2.initUndistortRectifyMap
    mapx1, mapy1 = rectify_fn(K1, D1, R1, P1, NEW_IMAGE_SHAPE, cv2.CV_16SC2)
    mapx2, mapy2 = rectify_fn(K2, D2, R2, P2, NEW_IMAGE_SHAPE, cv2.CV_16SC2)
    
    for fname in image_paths:
        print(fname)
        img = cv2.imread(fname)
        imgL, imgR = img[yy:(yy+hh), lxx:(lxx+ww)], img[yy:(yy+hh), rxx:(rxx+ww)]
        imgL_undisto = undistorted_fn(imgL, K1, D1, Knew=K1)
        imgR_undisto = undistorted_fn(imgR, K2, D2, Knew=K2)
        imgL_rectify = cv2.remap(imgL, mapx1, mapy1, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        imgR_rectify = cv2.remap(imgR, mapx2, mapy2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow('original left', imgL)
        cv2.imshow('undistort left', imgL_undisto)
        cv2.imshow('rectify left', imgL_rectify)
        cv2.imshow('original right', imgR)
        cv2.imshow('undistort right', imgR_undisto)
        cv2.imshow('rectify right', imgR_rectify)
        
        while True:
            key = cv2.waitKey(10)
            if key == ord('n'):
                break
            elif key == ord('q'):
                return
    return
        
print('K1 \n {}'.format(K1))
print('D1 \n {}'.format(D1))
print('R1 \n {}'.format(R1))
print('P1 \n {}'.format(P1))
print('K2 \n {}'.format(K2))
print('D2 \n {}'.format(D2))
print('R2 \n {}'.format(R2))
print('P2 \n {}'.format(P2))
Show()