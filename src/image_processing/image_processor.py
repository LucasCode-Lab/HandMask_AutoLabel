import os
import re
import yaml
import glob
import numpy as np
import cv2
import shutil
from natsort import natsorted
from utils import logger

def ensure_folder(folder: str):
    """
    創建資料夾，如果已存在就刪除重新創建

    :param folder: str 要創建的資料夾路徑
    """

    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    else:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
    index = -1
    if folder.split('/')[index] == "":
        index -= 1
    logger.logger.info("{}資料夾創建完成".format(folder.split('/')[index]))

def gen_folder(yaml_data):

    user = yaml_data['Class_param']['User']
    gesture_id = yaml_data['Class_param']["gesture_id"]
    angle_id = yaml_data['Class_param']["angle_id"]
    directory_path = {}

    # 儲存各種影像輸出路徑
    for path in yaml_data["output_dir"]:
        if yaml_data["output_dir"][path] != "":
            output = yaml_data["output_dir"][path].format(user, gesture_id, angle_id)
            ensure_folder(output)
            directory_path[path] = output
        else:
            continue
    return directory_path

def read_and_binarize_images(yaml_data):

    """
    讀取影像目錄內的影像，按順序更名後並存放在指定路徑
    :param yaml_data: 影像目錄的路徑 (str)
    :return: images: 影像的清單 (list)
    """

    image_dir = yaml_data['image_dir']

    # 存儲影像的列表
    images = []
    bin_images = []

    # 產生輸出路徑
    dir_map = gen_folder(yaml_data)
    # 獲取所有影像文件名稱的列表
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    # 按數字大小排序
    image_files.sort(key=sort_by_number)
    for index, filename in enumerate(image_files):
        # 判斷是否為jpg或png影像
        if filename.endswith('.jpg') or filename.endswith('.png'):
            try:
                # 開啟影像文件
                image = cv2.imread(filename)
                # 儲存讀取的影像
                cv2.imwrite(f"{dir_map['raw_dir']}/{index}.png", image)
                # 將影像添加到列表中
                images.append(image)
                # 增加影像對比
                image = cv2.convertScaleAbs(image, alpha=2, beta=0)
                # image = increase_contrast_in_roi(image, roi_size=(200, 300))
                cv2.imwrite(f"{dir_map['contrast_dir']}/{index}.png", image)
                # 影像二值化
                bin_image = binarize(image, threshold=47)
                # 進行腐蝕運算。
                ero_image = erosion(bin_image, kernel_size=3, iterations=1)
                # 找出最大輪廓範圍
                img_large = find_largest_contour(ero_image)
                # 對輪廓邊緣進行模糊化
                kernel_size = 5
                # img_blurred = cv2.GaussianBlur(ero_image, (kernel_size, kernel_size), 0)
                img_blurred = cv2.medianBlur(img_large, kernel_size)
                # 將影像添加到列表中
                bin_images.append(img_blurred)
                # 儲存讀取的影像
                # cv2.imwrite(f"{bin_images_output}/{index}.png", img_blurred)
                # 增加計數器
                index += 1
            except Exception as e:
                print(f"發生錯誤：{e}")
    return images, bin_images, dir_map


def increase_contrast_in_roi(image, roi_size=(100, 100), alpha=2.5, beta=0):

    """
    增加 ROI 區域的對比度。
    :param image: 輸入的圖像，必須是一個 NumPy 陣列。
    :param roi_size: ROI 區域的大小，預設值為 (100, 100)。
    :param alpha: 對比度系數，預設值為 2.5。
    :param beta: 亮度系數，預設值為 0。
    :return: 增加對比度後的圖像，為一個 NumPy 陣列。
    """

    # 創建 ROI
    rows, cols = image.shape[:2]
    roi = image[rows - roi_size[0]:rows, cols - roi_size[1]:cols]

    # 對 ROI 增加對比度
    roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)

    # 將 ROI 複制回圖片中
    image[rows - roi_size[0]:rows, cols - roi_size[1]:cols] = roi

    return image


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

import numpy as np


def gamma_correction(image, gamma=1.0):
    """
    進行 gamma correction 的函數
    :param image: numpy.ndarray, 需要進行 gamma correction 的圖片
    :param gamma: float, gamma 值
    :return: numpy.ndarray, gamma correction 後的結果
    """
    # 建立 gamma correction 的 lookup table
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    # 使用 LUT 對圖片進行 gamma correction
    img_corrected = cv2.LUT(image, look_up_table)
    return img_corrected


def erosion(img, kernel_size=3, iterations=1):
    """
    進行腐蝕運算。

    :param img: 要進行腐蝕運算的圖像，必須是一個 NumPy 陣列。
    :param kernel_size: 核心的大小，預設值為 3。
    :param iterations: 運算的次數，預設值為 1。

    :return: 腐蝕運算後的圖像，為一個 NumPy 陣列。
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def dilate(image, kernel_size=3, iterations=1):

    """
    進行膨脹運算。

    :param image: 輸入的圖像，必須是一個 NumPy 陣列。
    :param kernel_size: 膨脹核的大小。

    :return: 膨脹後的圖像，為一個 NumPy 陣列。
    """

    # Create a dilation kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)

    return dilated_image


def convert_scale_abs(image, alpha, beta):
    """
    縮放圖像並轉換為絕對值。

    參數：
    - image (np.array)：輸入圖像。
    - alpha (float)：縮放因子。
    - beta (float)：添加到縮放圖像上的常數。

    返回值：
    - scaled_image (np.array)：縮放和轉換後的圖像。
    """
    # 縮放並將圖像轉換為絕對值
    return cv2.convertScaleAbs(image, alpha, beta)

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


def show_images(*images):
    """
    結合多張圖片並顯示

    :param images: numpy.ndarray 要結合的多張圖片，每張圖片可以是灰階或彩色圖片
    :returns: numpy.ndarray 結合完的圖片
    """
    try:
        # 將圖片轉換為 numpy.ndarray 並強制轉換為 uint8 型態
        image_list = [np.asarray(image, dtype=np.uint8) for image in images]

        # 如果是灰階圖片，則將其轉換為彩色圖片
        image_list = [np.dstack((image, image, image)) if len(image.shape) == 2 else image for image in image_list]

        # 如果圖片數量為奇數且不止一張，則新增一張空白圖片
        if len(image_list) % 2 != 0 and len(image_list) > 1:
            image_list = np.concatenate((image_list, np.zeros_like(np.expand_dims(image_list[-1], axis=0))), axis=0)

        if len(image_list) > 1:
            # 將圖片數量平均分配為兩組
            n = len(image_list) // 2
            group1 = image_list[:n]
            group2 = image_list[n:]

            # 將同一組的圖片水平接起來
            row1 = cv2.hconcat(group1)
            row2 = cv2.hconcat(group2)

            # 將兩組圖片垂直接起來
            result = cv2.vconcat([row1, row2])
            return result
        else:
            return image_list[0]
    except Exception as e:
        print(f"Error: {e}")


def image_2_video(folder: str, size: (int, int)):

    """
    將圖片轉換為視頻檔案。

    :param folder: 存放圖片的文件夾的路徑。
    :param size: 視頻幀的大小。
    :return: None
    """

    # 取得幀的大小
    frameSize = size
    # 使用指定的輸出檔案名稱、FourCC代碼、每秒幀數和幀大小創建VideoWriter物件
    out = cv2.VideoWriter(folder + '/output_video.mp4', cv2.VideoWriter_fourcc(*'MPEG'), 10, frameSize)
    # 取得按名稱排序的文件名列表
    imgs = [cv2.imread(os.path.join(folder, filename)) for filename in natsorted(os.listdir(folder))]
    # 迭代圖像
    for img in imgs:
        # 寫入圖像到視頻檔案
        out.write(img)
    # 釋放VideoWriter物件
    out.release()


def sort_by_number(filename):
    """
    根據檔案名的數字對檔案名進行排序

    :param filename: 檔案名字符串 (str)
    :return: 檔案名數字部分的整數 (int)
    """

    # 使用正則表達式提取文件名的數字部分
    number = int(re.findall(r'(\d+)\.(png|jpg)', filename)[0][0])
    return number


def find_largest_contour(image):

    """
    在二值化圖像上找出最大的輪廓並在黑色背景上畫出白色的最大輪廓面積。
    參數：
    - image (np.array)：二值化圖像。

    返回值：
    - contour_image (np.array)：在黑色背景上畫出白色最大輪廓面積的圖像。
    """

    # 找到輪廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到面積最大的輪廓
    largest_contour = max(contours, key=cv2.contourArea)
    # 在黑色背景上畫出白色最大輪廓面積
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, [largest_contour], 0, 255, -1)

    return contour_image


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
