from utils import logger
import argparse
import cv2
import numpy as np
from image_processing.image_processor import load_yaml_config, read_and_binarize_images, ensure_folder
from gesture_analysis.hand_detect import (detect_joints,
                                          cal_angle_rotatematrix,
                                          rotate_points,
                                          createBoundingBox,
                                          extract_largest_contour_mask)

parser = argparse.ArgumentParser(description='Process yaml config file path.')
# 加入命令列參數 --config，並設定目標變數名稱為 config_path
parser.add_argument('--config', dest='config_path', required=True, help='Path to the yaml config file')
# 解析命令列參數
args = parser.parse_args()
# 將命令列參數中的 YAML 檔案位置存到變數 yaml_config_path
yaml_config_path = args.config_path
# 使用 logger 記錄器，記錄 YAML 檔案位置
logger.logger.info("Yaml 檔案位置: {}".format(yaml_config_path))
# 使用 load_yaml_config 函數讀取 YAML 配置檔，並存到變數 yaml_data
yaml_data = load_yaml_config(yaml_config_path)
# 使用 read_and_binarize_images 函數，讀取並二值化圖片
images_list, bin_image_list = read_and_binarize_images(yaml_data)

for index, (image, bin_image) in enumerate(zip(images_list, bin_image_list)):
    # 創建圖片的拷貝以進行處理
    current_image = np.copy(image)
    # 檢測關節點，並存在 image 和 points 變量中
    joint_detected_image, joints = detect_joints(current_image)
    # 計算旋轉角度和旋轉矩陣，並存在 angle_rad、angle_deg 和 rotate_matrix 變量中
    rotate_matrix = cal_angle_rotatematrix(joint_detected_image, joints)
    # 對關節點進行旋轉
    rotated_joints = rotate_points(joints, rotate_matrix)
    # 創建手部邊界框
    bounding_box_image, bounding_rect = createBoundingBox(joint_detected_image, rotated_joints, rotate_matrix)
    # 儲存 bbox 結果的圖片
    cv2.imwrite(f"{yaml_data['bbox_output_dir']}/{index}.png", bounding_box_image)

    # 處理手臂遮罩
    binary_current_image = np.copy(bin_image)
    arm_mask = extract_largest_contour_mask(binary_current_image, bounding_rect)
    cv2.imwrite(f"{yaml_data['arm_mask_output_dir']}/{index}.png", arm_mask)
