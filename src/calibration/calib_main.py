import argparse
import numpy as np
from calib_func import calibrate_camera

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", help="Path of images folder", type=str)
parser.add_argument("-co", "--calib_output", help="Path of calib output file", type=str)
args = parser.parse_args()

mtx, dist = calibrate_camera(images_folder=args.folder)
np.save(args.calib_output+"mtx", mtx)
np.save(args.calib_output+"dist", dist)
