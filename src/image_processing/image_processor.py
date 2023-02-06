import os
import re
import yaml
import glob
import numpy as np
import cv2
import shutil
from utils import logger


def sort_by_number(filename):
    """
    根據檔案名的數字對檔案名進行排序

    :param filename: 檔案名字符串 (str)
    :return: 檔案名數字部分的整數 (int)
    """

    # Extract the numerical part of the filename using a regular expression
    number = int(re.findall(r'\d+', filename)[0])
    return number


def ensure_folder(folder: str):
    """
    創建資料夾，如果已存在就刪除重新創建

    :param folder: str 要創建的資料夾路徑
    """

    if not os.path.isdir(folder):
        os.mkdir(folder)
    else:
        shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)
    index = -1
    if folder.split('/')[index] == "":
        index -= 1
    logger.logger.info("{}資料夾創建完成".format(folder.split('/')[index]))


def read_and_binarize_images(yaml_data):
    """讀取影像目錄內的影像，按順序更名後並存放在指定路徑
    :param yaml_data: 影像目錄的路徑 (str)
    :return: images: 影像的清單 (list)
    """
    image_dir = yaml_data['image_dir']

    # 存儲影像的列表
    images = []
    bin_images = []
    # 輸出路徑
    images_output = yaml_data['output_raw_dir']
    ensure_folder(images_output)
    # 二值化影像的輸出路徑
    bin_images_output = yaml_data['output_bin_dir']
    ensure_folder(bin_images_output)
    # 儲存 bbox 結果的圖片
    bbox_images_output = yaml_data['bbox_output_dir']
    ensure_folder(bbox_images_output)
    # 儲存 arm_mask 結果的圖片
    arm_mask_output = yaml_data['arm_mask_output_dir']
    ensure_folder(arm_mask_output)
    # 儲存 unit_mask 結果的圖片
    unit_mask_output = yaml_data['unit_mask_output_dir']
    ensure_folder(unit_mask_output)

    # 獲取所有影像文件名稱的列表
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    # 按數字大小排序
    image_files.sort(key=sort_by_number)
    # 計數器，用於按順序命名影像
    index = 0
    for index, filename in enumerate(image_files):
        # 判斷是否為jpg或png影像
        if filename.endswith('.jpg') or filename.endswith('.png'):
            try:
                # 開啟影像文件
                image = cv2.imread(filename)
                # 儲存讀取的影像
                cv2.imwrite(f"{images_output}/{index}.png", image)
                # 將影像添加到列表中
                images.append(image)
                # 影像二值化
                bin_image = binarize(image, threshold=30)
                # 將影像添加到列表中
                bin_images.append(bin_image)
                # 儲存讀取的影像
                cv2.imwrite(f"{bin_images_output}/{index}.png", bin_image)
                # 增加計數器
                index += 1
            except Exception as e:
                print(f"發生錯誤：{e}")
    return images, bin_images


# def binarize(image, threshold: int):
#     """
#     進行二值化的函數
#     :param image: numpy.ndarray, 需要進行二值化的圖片
#     :param threshold: int, 閾值
#     :return: numpy.ndarray, 二值化後的結果
#     """
#     # 檢查 threshold 是否為 int
#     if not isinstance(threshold, int):
#         raise TypeError("threshold should be an integer.")
#
#     try:
#         # 將圖片中的像素值大於閾值的設置為1，否則設置為0
#         binarized = np.where(image > threshold, 255, 0)
#         return binarized
#     except Exception as e:
#         logger.logger.error("binarization failed: " + str(e))
#         return None

def binarize(image, threshold: int):
    """
    進行二值化的函數
    :param image: numpy.ndarray, 需要進行二值化的圖片
    :param threshold: int, 閾值
    :return: numpy.ndarray, 二值化後的結果
    """

    try:
        # 將圖片轉換成灰階色彩
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 透過阈值二值化
        ret, mask = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        return mask
    except Exception as e:
        logger.logger.error("binarization failed: " + str(e))
        return None
def load_yaml_config(file_path):

    """
    讀取 yaml 配置檔
    :param file_path: yaml 配置檔路徑
    :return: 讀取到的 yaml 配置資料
    """

    # 檢查檔案是否存在
    assert os.path.isfile(file_path), "Yaml File Not Exist!!"
    # 讀取 yaml 檔案
    with open(file_path, "r") as stream:
        yaml_data = yaml.full_load(stream)
    return yaml_data



