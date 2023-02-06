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
NUM_IMAGES = yaml_data['Calibration']['NUM_IMAGES']
IMAGE_SHAPE = yaml_data['Calibration']['IMAGE_SHAPE']
BORDER_THRESHOLD = yaml_data['Calibration']['BORDER_THRESHOLD']
CAMERA_FIND_ALGO = yaml_data['Calibration']['CAMERA_FIND_ALGO']
STEREO_FIND_ALGO = yaml_data['Calibration']['STEREO_FIND_ALGO']
SUBPIX_SIZE = yaml_data['Calibration']['SUBPIX_SIZE']
DISTORTION_MODEL = yaml_data['Calibration']['DISTORTION_MODEL']
FIX_SKEW = yaml_data['Calibration']['FIX_SKEW']
USE_IMAGE_FILTER = yaml_data['Calibration']['USE_IMAGE_FILTER']
NEW_IMAGE_SHAPE = yaml_data['Calibration']['NEW_IMAGE_SHAPE']
NEW_IMAGE_BALANCE = yaml_data['Calibration']['NEW_IMAGE_BALANCE']
MAX_REPROJ_ERROR = yaml_data['Calibration']['MAX_REPROJ_ERROR']
USE_DEFAULT_RECTIFY = yaml_data['Calibration']['USE_DEFAULT_RECTIFY']
INTRINSIC_OUTPUT_FILE = yaml_data['Calibration']['INTRINSIC_OUTPUT_FILE']
# ===========================================================

# https://euratom-software.github.io/calcam/html/intro_theory.html
# Default Module: Rectilinear Lens Distortion Model
# Fisheye Module: Fisheye Lens Distirtion Model
if DISTORTION_MODEL == 'Fisheye':
    USE_FISHEYE_API = True
    CAMERA_CALIBRATION_FLAGS = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    STEREO_CALIBRATION_FLAGS = cv2.fisheye.CALIB_FIX_INTRINSIC + cv2.fisheye.CALIB_CHECK_COND
    if FIX_SKEW:
        CAMERA_CALIBRATION_FLAGS += cv2.fisheye.CALIB_FIX_SKEW
        STEREO_CALIBRATION_FLAGS += cv2.fisheye.CALIB_FIX_SKEW
    RECTIFY_FLAGS = cv2.fisheye.CALIB_ZERO_DISPARITY
else:
    USE_FISHEYE_API = False
    if DISTORTION_MODEL == 'RectilinearK4':
        CAMERA_CALIBRATION_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST \
            + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    elif DISTORTION_MODEL == 'RectilinearK6':
        CAMERA_CALIBRATION_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL
    elif DISTORTION_MODEL == 'Rectilinear5':
        CAMERA_CALIBRATION_FLAGS = 0
    elif DISTORTION_MODEL == 'Rectilinear14':
        CAMERA_CALIBRATION_FLAGS = cv2.CALIB_RATIONAL_MODEL \
            + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL 
    elif DISTORTION_MODEL == 'Rectilinear0':
        CAMERA_CALIBRATION_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST \
            + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
    else:
        assert False, 'Do not support {}'.format(DISTORTION_MODEL)
    STEREO_CALIBRATION_FLAGS = cv2.CALIB_FIX_INTRINSIC + CAMERA_CALIBRATION_FLAGS
    RECTIFY_FLAGS = cv2.CALIB_ZERO_DISPARITY 
SUBPIX_WIN_SIZE = (SUBPIX_SIZE, SUBPIX_SIZE)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
CAMERA_CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
STEREO_CALIBRATION_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
IMAGE_WIDTH = IMAGE_SHAPE[0]
IMAGE_HEIGHT = IMAGE_SHAPE[1]
    
def SwitchFindAlgo(find_algo):
    global FIND_PATTERN_VERSION, USE_CIRCLES_GRID, PATTERN_TYPE, PATTERN_COL, PATTERN_ROW, FIND_FLAGS, OBJP
    if find_algo == "findCirclesGrid":
        FIND_PATTERN_VERSION = 0
    elif find_algo == "findChessboardCorners":
        FIND_PATTERN_VERSION = 1
    elif find_algo == "findChessboardCornersSB":
        FIND_PATTERN_VERSION = 2
    else:
        assert False, 'Do not support {}'.format(find_algo)
    USE_CIRCLES_GRID = FIND_PATTERN_VERSION == 0
    PATTERN_TYPE = "CirclesGrid" if USE_CIRCLES_GRID else "Checkerboard"
    if USE_CIRCLES_GRID:
        PATTERN_ROW = 13
        PATTERN_COL = 5
        SQUARE_SIZE = 40
    else:
        PATTERN_ROW = 18
        PATTERN_COL = 11
        SQUARE_SIZE = 15
    if USE_CIRCLES_GRID:
        shift = SQUARE_SIZE / 2
        OBJP = np.zeros((PATTERN_ROW*PATTERN_COL, 3), np.float32)
        objp_index = 0
        for row in range(PATTERN_ROW):
            for col in range(PATTERN_COL):
                if row % 2 == 0:
                    OBJP[objp_index] = (row * shift, col * SQUARE_SIZE, 0)
                else:
                    OBJP[objp_index] = (row * shift, shift + col * SQUARE_SIZE, 0)
                objp_index += 1
    else:
        OBJP = np.zeros((PATTERN_ROW*PATTERN_COL, 3), np.float32)
        OBJP[:, :2] = np.mgrid[0:PATTERN_COL, 0:PATTERN_ROW].T.reshape(-1, 2) * SQUARE_SIZE
    if USE_FISHEYE_API:
        OBJP = OBJP[None,:,:]
    if FIND_PATTERN_VERSION == 0:
        FIND_FLAGS = cv2.CALIB_CB_ASYMMETRIC_GRID
    elif FIND_PATTERN_VERSION == 1:
        FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    elif FIND_PATTERN_VERSION == 2:
        FIND_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    else:
        assert False 

def FindPattern(gray):
    if FIND_PATTERN_VERSION == 0:
        gray = cv2.equalizeHist(gray)
    if USE_IMAGE_FILTER:
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.bilateralFilter(gray, 5, 25, 75)
    if FIND_PATTERN_VERSION == 0:
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        ret, corners = cv2.findCirclesGrid(binary, (PATTERN_COL, PATTERN_ROW), flags=FIND_FLAGS)
        if not ret:
            ret, corners = cv2.findCirclesGrid(gray, (PATTERN_COL, PATTERN_ROW), flags=FIND_FLAGS)
    elif FIND_PATTERN_VERSION == 1:
        ret, corners = cv2.findChessboardCorners(gray, (PATTERN_COL, PATTERN_ROW), flags=FIND_FLAGS)
    elif FIND_PATTERN_VERSION == 2:
        ret, corners = cv2.findChessboardCornersSB(gray, (PATTERN_COL, PATTERN_ROW), flags=FIND_FLAGS)
    if (not USE_CIRCLES_GRID) and ret and SUBPIX_SIZE > 0:
        corners = cv2.cornerSubPix(gray, corners, SUBPIX_WIN_SIZE, (-1, -1), SUBPIX_CRITERIA)
    if ret:
        minx = corners[:,:,0].min()
        maxx = corners[:,:,0].max()
        miny = corners[:,:,1].min()
        maxy = corners[:,:,1].max()
        if (minx < BORDER_THRESHOLD) or (miny < BORDER_THRESHOLD) or \
            (IMAGE_WIDTH-maxx < BORDER_THRESHOLD) or (IMAGE_HEIGHT-maxy < BORDER_THRESHOLD):
            ret = False
    return ret, corners

def CalibrateCamera(use_left):
    objpoints = []  # 3d point in real world space
    imgpoints = []
    xx = 0 if use_left else IMAGE_WIDTH
    yy, ww, hh = 0, IMAGE_WIDTH, IMAGE_HEIGHT
    image_paths = sorted(glob.glob(FILE_PATH))
    # image_paths = [ FILE_PATH.format(PATTERN_TYPE, n) for n in range(NUM_IMAGES) ]

    for fname in image_paths:
        print('fname:', fname)
        img = cv2.imread(fname)
        img = img[yy:(yy+hh), xx:(xx+ww)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = FindPattern(gray)
        if ret:
            objpoints.append(OBJP)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (PATTERN_COL, PATTERN_ROW), corners, ret)
            cv2.imshow('img', img)
            key = cv2.waitKey(5)
            # if key == ord('n'):
            #     continue
    
    head = "LEFT" if use_left else "RIGHT"
    calibrate_fn = cv2.fisheye.calibrate if USE_FISHEYE_API else cv2.calibrateCamera
    
    try:
        ret, mtx, disto, rvecs, tvecs = calibrate_fn(
                objpoints, imgpoints, IMAGE_SHAPE, None, None,
                flags=CAMERA_CALIBRATION_FLAGS, criteria=CAMERA_CALIBRATION_CRITERIA)
        print("# {} CAMERA: {}".format(head, ret))
        print("{}_INTRINSIC = [{},{},{},{},{}]".format(head, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2], mtx[0, 1]))
        print('{}_DISTORTION = {}'.format(head, disto.flatten().tolist()))
        d = {'{}_INTRINSIC'.format(head): [ float(mtx[0, 0]), float(mtx[1, 1]), float(mtx[0, 2]), float(mtx[1, 2]), float(mtx[0, 1])],
             '{}_DISTORTION'.format(head) : disto.flatten().tolist(),
             }
        if os.path.isfile(INTRINSIC_OUTPUT_FILE) != True:
            os.mknod(INTRINSIC_OUTPUT_FILE)
        with open(INTRINSIC_OUTPUT_FILE, 'w') as f:
            yaml.dump(d, f)

    except Exception as err_msg:
        ret, mtx, disto = -1, None, None
        print("# {} CAMERA: FAILED".format(head))
        print("# {}".format(err_msg)) 
    return ret, mtx, disto

def TestStereoCalibrate(lp, rp):
    try:
        if USE_FISHEYE_API:
            ret, _, _, _, _, R, T = cv2.fisheye.stereoCalibrate(
                OBJP.reshape(1,1,-1,3), lp.reshape(1,1,-1,2), rp.reshape(1,1,-1,2), K1, D1, K2, D2, IMAGE_SHAPE, 
                flags=STEREO_CALIBRATION_FLAGS, criteria=STEREO_CALIBRATION_CRITERIA)
        else:
            ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
                [OBJP], [lp], [rp], K1, D1, K2, D2, IMAGE_SHAPE, 
                flags=STEREO_CALIBRATION_FLAGS, criteria=STEREO_CALIBRATION_CRITERIA)
        quat = Rot.from_matrix(R).as_quat()
        # print('# Angle {} T {} Err {}'.format(quat, T.flatten(), ret))
    except Exception as err:
        # print("# {}".format(str(err).replace("\n", " "))) 
        return False
    return ret < MAX_REPROJ_ERROR and abs(quat[3]) > 0.9

def CalibrateStereo():
    objpoints = []  # 3d point in real world space
    rpoints = []  # 2d points in image plane.
    lpoints = []  # 2d points in image plane.
    lxx, rxx = 0, IMAGE_WIDTH
    yy, ww, hh = 0, IMAGE_WIDTH, IMAGE_HEIGHT
    image_paths = [ FILE_PATH.format(PATTERN_TYPE, n) for n in range(NUM_IMAGES) ]

    for fname in image_paths:
        img = cv2.imread(fname)
        gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgL, imgR = img[yy:(yy+hh), lxx:(lxx+ww)], img[yy:(yy+hh), rxx:(rxx+ww)]
        grayL, grayR = gImg[yy:(yy+hh), lxx:(lxx+ww)], gImg[yy:(yy+hh), rxx:(rxx+ww)]

        retL, cornersL = FindPattern(grayL)
        retR, cornersR = FindPattern(grayR)
        if retL and retR:
            retS = TestStereoCalibrate(cornersL, cornersR)
        else:
            retS = False
        if retL and retR and retS:
            objpoints.append(OBJP)
            lpoints.append(cornersL)
            rpoints.append(cornersR)
            imgL = cv2.drawChessboardCorners(imgL, (PATTERN_COL, PATTERN_ROW), cornersL, retL)
            cv2.imshow('imgL', imgL)
            imgR = cv2.drawChessboardCorners(imgR, (PATTERN_COL, PATTERN_ROW), cornersR, retR)
            cv2.imshow('imgR', imgR)
            key = cv2.waitKey(5)
            
    num_ok = len(objpoints)
    print("# N_OK {}".format(num_ok))
    
    if num_ok == 0:
        print("# Calibrate Stereo Failed")
        return
    if USE_FISHEYE_API:
        objpoints = np.asarray([objpoints], dtype=np.float64)
        rpoints = np.asarray([rpoints], dtype=np.float64)
        lpoints = np.asarray([lpoints], dtype=np.float64)
        objpoints = np.reshape(objpoints, (num_ok, 1, PATTERN_ROW*PATTERN_COL, 3))
        rpoints = np.reshape(rpoints, (num_ok, 1, PATTERN_ROW*PATTERN_COL, 2))
        lpoints = np.reshape(lpoints, (num_ok, 1, PATTERN_ROW*PATTERN_COL, 2))
        ret, _, _, _, _, rot, trans = cv2.fisheye.stereoCalibrate(
            objpoints, lpoints, rpoints, K1, D1, K2, D2, IMAGE_SHAPE, 
            flags=STEREO_CALIBRATION_FLAGS, criteria=STEREO_CALIBRATION_CRITERIA)
    else:
        ret, _, _, _, _, rot, trans, _, _ = cv2.stereoCalibrate(
            objpoints, lpoints, rpoints, K1, D1, K2, D2, IMAGE_SHAPE, 
            flags=STEREO_CALIBRATION_FLAGS, criteria=STEREO_CALIBRATION_CRITERIA)
    
    print("# STEREO CAMERA: {}".format(ret))
    print('# R Angle {}'.format(Rot.from_matrix(rot).as_euler('xyz', degrees=True)))
    print('# T {}'.format(trans.flatten()))
    flag = (not USE_DEFAULT_RECTIFY) and USE_FISHEYE_API
    
    if flag:
        r1_, r2_, p1_, p2_, q_ = cv2.fisheye.stereoRectify(
            K1, D1, K2, D2, IMAGE_SHAPE, rot, trans,
            flags=RECTIFY_FLAGS, newImageSize=NEW_IMAGE_SHAPE, balance=NEW_IMAGE_BALANCE, fov_scale=0)
    else:
        if USE_FISHEYE_API:
            d1 = np.zeros((14,), np.float32)
            d2 = np.zeros((14,), np.float32)
            d1[0], d1[1], d1[4], d1[5] = D1[0], D1[1], D1[2], D1[3]
            d2[0], d2[1], d2[4], d2[5] = D2[0], D2[1], D2[2], D2[3]
        else:
            d1, d2 = D1.copy(), D2.copy()
        r1_, r2_, p1_, p2_, q_, validROI1, validROI2 = cv2.stereoRectify(
            K1, d1, K2, d2, IMAGE_SHAPE, rot, trans, 
            flags=RECTIFY_FLAGS, newImageSize=NEW_IMAGE_SHAPE, alpha=NEW_IMAGE_BALANCE)
        print("# validROI1 {}".format(validROI1))
        print("# validROI2 {}".format(validROI2))
    print('# Quaternion Format is (x, y, z, w)')
    print("# LEFT Angle {}".format(Rot.from_matrix(r1_).as_euler('xyz', degrees=True)))
    print("LEFT_QUATERNION = {}".format(Rot.from_matrix(r1_).as_quat().tolist()))
    print("# RIGHT Angle {}".format(Rot.from_matrix(r2_).as_euler('xyz', degrees=True)))
    print("RIGHT_QUATERNION = {}".format(Rot.from_matrix(r2_).as_quat().tolist()))
    baseline = 1.0/q_[3,2]
    print("BASELINE = {}".format(baseline))
    print("LEFT_PROJECTION = [{},{},{},{}]".format(p1_[0, 0], p1_[1, 1], p1_[0, 2], p1_[1, 2]))
    print("RIGHT_PROJECTION = [{},{},{},{}]".format(p2_[0, 0], p2_[1, 1], p2_[0, 2], p2_[1, 2]))
    # print('P1 \n {}'.format(p1_))
    # print('P2 \n {}'.format(p2_))
    # print('Q \n {}'.format(Q))
    return

print('# {} + {}'.format(CAMERA_FIND_ALGO, STEREO_FIND_ALGO))
print('FILE_PATH = "{}"'.format(FILE_PATH))
print('NUM_IMAGES = {}'.format(NUM_IMAGES))
print('IMAGE_SHAPE = {}'.format(IMAGE_SHAPE))
print('NEW_IMAGE_SHAPE = {}'.format(NEW_IMAGE_SHAPE))
print('FIND_ALGO = "{}"'.format(CAMERA_FIND_ALGO))
print('DISTORTION_MODEL = "{}"'.format(DISTORTION_MODEL))
print('USE_DEFAULT_RECTIFY = {}'.format(USE_DEFAULT_RECTIFY or not USE_FISHEYE_API))

SwitchFindAlgo(CAMERA_FIND_ALGO)
retL, K1, D1 = CalibrateCamera(use_left=True)
# retR, K2, D2 = CalibrateCamera(use_left=False)
#
# if retL > 0 and retR > 0:
#     SwitchFindAlgo(STEREO_FIND_ALGO)
#     CalibrateStereo()