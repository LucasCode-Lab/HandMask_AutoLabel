import os
import re
import glob
from PIL import Image


def sort_by_number(filename):
    """
    根據檔案名的數字對檔案名進行排序

    :param filename: 檔案名字符串 (str)
    :return: 檔案名數字部分的整數 (int)
    """

    # Extract the numerical part of the filename using a regular expression
    number = int(re.findall(r'\d+', filename)[0])
    return number


def read_images(image_dir):
    """讀取影像目錄內的影像，按順序更名後並存放在指定路徑
    :param image_dir: 影像目錄的路徑 (str)
    :return: images: 影像的清單 (list)
    """
    # 存儲影像的列表
    images = []
    # 輸出路徑
    images_output = os.path.join(image_dir, 'images_output')

    # 檢查輸出路徑是否存在，不存在則創建
    if not os.path.exists(images_output):
        print("輸出路徑不存在，已自動創建：", images_output)
        os.makedirs(images_output)

    # 獲取所有影像文件名稱的列表
    image_files = glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    # 按數字大小排序
    image_files.sort(key=sort_by_number)
    # 計數器，用於按順序命名影像
    index = 0
    for filename in image_files:
        # 判斷是否為jpg或png影像
        if filename.endswith('.jpg') or filename.endswith('.png'):
            try:
                # 開啟影像文件
                image = Image.open(filename)
                # 將影像添加到列表中
                images.append(image)
                # 儲存讀取的影像
                image.save(os.path.join(images_output, "%d.png" % index))
                # 增加計數器
                index += 1
            except Exception as e:
                print(f"發生錯誤：{e}")
    return images





