import cv2
import argparse
import numpy as np
from autolabel import annotation
from utils.file_manage import load_yaml_config
from utils.logger import configure_logging
from image_processing.image_processor import (process_and_save_images,
                                              show_images)
from autolabel.hand_detect import (detect_joints,
                                   cal_angle_rotatematrix,
                                   rotate_points,
                                   createBoundingBox,
                                   extract_largest_contour_mask,
                                   save_mask_image)
logger = configure_logging(__name__)

parser = argparse.ArgumentParser(description='Process yaml config file path.')
# 加入命令列參數 --config，並設定目標變數名稱為 config_path
parser.add_argument('--config', dest='config_path', required=True, help='Path to the yaml config file')
# 解析命令列參數
args = parser.parse_args()
# 將命令列參數中的 YAML 檔案位置存到變數 yaml_config_path
yaml_config_path = args.config_path
# 使用 load_yaml_config 函數讀取 YAML 配置檔，並存到變數 yaml_data
yaml_data = load_yaml_config(yaml_config_path)
# 使用 read_and_binarize_images 函數，讀取並二值化圖片
images_list, bin_image_list, dir_map = process_and_save_images(yaml_data)

for index, (image, bin_image) in enumerate(zip(images_list, bin_image_list)):
    # 創建圖片的拷貝以進行處理
    current_image = np.copy(image)
    # 檢測關節點，並存在 image 和 points 變量中
    joint_detected_image, joints = detect_joints(current_image)
    if len(joints) != 21:
        continue

    # 創建手部邊界框
    bounding_box_image, bboxRadius = createBoundingBox(joint_detected_image, joints)

    # 處理手臂遮罩
    binary_current_image = np.copy(bin_image)
    arm_mask = extract_largest_contour_mask(binary_current_image, joints, bboxRadius)

    # unimask
    unit_mask = save_mask_image(arm_mask, yaml_data)

    # 產生覆蓋在原圖上的手臂 mask 圖片
    overlay = cv2.addWeighted(image[:, :, 0], 0.4, arm_mask, 0.3, 0)

    # 呈現所有處理結果
    gama = cv2.imread(f"{dir_map['contrast_dir']}/{index}.png")
    vis_output = show_images(image, joint_detected_image, arm_mask, overlay, bounding_box_image, gama)

    # 儲存結果的圖片
    # cv2.imwrite(f"{dir_map['bbox_images_output']}/{index}.png", bounding_box_image)
    # cv2.imwrite(f"{dir_map['arm_mask_output']}/{index}.png", arm_mask)
    np.save(f"{dir_map['unit_mask_dir']}/{index}.npy", unit_mask)
    cv2.imwrite(f"{dir_map['merge_vis_dir']}/{index}.png", vis_output)
    cv2.imwrite(f"{dir_map['bin_vis_dir']}/{index}.png", bin_image)

annotation.sub_annotation(yaml_data, dir_map)
annotation.annotation_res(yaml_data)
# 將每張圖片彙整成視頻
# image_2_video(f"{yaml_data['bbox_output_dir']}", (640, 400))
