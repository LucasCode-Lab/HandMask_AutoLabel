import os
import re
import json
import glob
import copy
import yaml
from typing import Dict, List
from utils.logger import logger
from typing_extensions import Any
from image_processing.image_processor import sort_by_number, ensure_folder


def sub_annotation(yaml_data: Dict[str, Any], dir_map: Dict[str, str]) -> None:
    """
    將 yaml_data 檔案中的 annotation 資訊寫入到指定路徑下的 data.yaml 檔案中
    並添加 image_path, mask_path, camera 和 accessories 屬性，
    然後將結果寫入 data.yaml 檔案中，供後續產生 annotation JSON 檔案使用。

    Args:
        yaml_data (Dict[str, Any]): 待處理的 yaml 資料。
        dir_map (Dict[str, str]): 路徑映射字典，包含 raw_dir 和 unit_mask_dir 。

    :param yaml_data: 原始 YAML 檔案的資料
    :type yaml_data: Dict
    :param dir_map: 存放寫入資料的目標路徑
    :type dir_map: Dict
    :return: None
    """
    if os.path.exists(dir_map["raw_dir"]):
        pass
    else:
        logger.logger.Error("sub_annotation raw_dir路徑找不到!!")
    if os.path.exists(dir_map["unit_mask_dir"]):
        pass
    else:
        logger.logger.Error("sub_annotation unit_mask路徑找不到!!")

    data = yaml_data.get("Class_param", {})
    data["image_path"] = dir_map["raw_dir"]
    data["mask_path"] = dir_map["unit_mask_dir"]
    data["camera"] = yaml_data['camera']
    data["annotation"] = yaml_data["Annotation"]
    with open(dir_map['dir'] + "/data.yaml", "w") as outfile:
        yaml.dump(data, outfile, sort_keys=False)


def search_sub_yaml(yaml_data: Dict[str, Any]) -> List[str]:
    """
    搜尋指定目錄下的所有 .yaml 檔案。

    Args:
        yaml_data (Dict[str, Any]): 待處理的 YAML 資料。

    Returns:
        List[str]: 所有符合條件的 .yaml 檔案路徑清單，每個路徑都是一個字串。
    """

    # 遞迴搜尋指定目錄下的所有檔案和子目錄
    yaml_path = [
        os.path.join(dir_path, filename)
        for dir_path, _, filenames in os.walk(yaml_data["sub_yaml"], topdown=False)
        for filename in filenames if filename.endswith(".yaml")
    ]
    # 印出所有找到的檔案路徑
    logger.info("找到的 YAML 檔案：\n{}".format("\n".join(yaml_path)))

    # 回傳所有符合條件的檔案路徑清單
    return yaml_path


def annotation_res(yaml_data: Dict[str, Any]) -> None:
    """
        根據 yaml_data 中的設定，建立三個 JSON 檔案：
            - images.json：包含所有圖片的相關訊息。
            - mask.json：包含所有遮罩圖的相關訊息。
            - annotation.json：包含所有標註資料的相關訊息。

        Args:
            yaml_data (Dict[str, Any]): 待處理的 YAML 資料。

        Returns:
            None
        """

    # 設定基礎資訊
    data = {
        "info": {
            "description": "CRI 2023 Hand Datasets",
            "version": "1.0",
            "date": "2023-01-31",
            "organization": "Coretronic Reality Inc. AR/MR AI Technology Division"
        }
    }
    # 設定手部類別資訊
    categories = [{}, {}]

    # 建立 image、mask、annotation 資料的副本
    data_image = data.copy()
    data_mask = data.copy()
    data_anno = data.copy()

    # 為三個副本設定基本資訊
    data_image.setdefault("images", [])
    data_mask.setdefault("mask", [])
    data_anno.setdefault("annotations", [])

    # 找到所有 sub yaml 檔案
    yaml_path = search_sub_yaml(yaml_data)

    # 設定批次處理的資料數量
    BATCH_SIZE = 5000

    # 設定全域 id
    global_id = 0

    # 確保三個資料夾存在, 若不存在則建立
    ensure_folder(yaml_data['annotation_json'] + "/images")
    ensure_folder(yaml_data['annotation_json'] + "/mask")
    ensure_folder(yaml_data['annotation_json'] + "/annotation")

    for image_path in yaml_path:
        # 深度拷貝手部類別資訊
        categories_copy = copy.deepcopy(categories)

        # 讀取 yaml 檔案
        with open(image_path, "r") as stream:
            yaml_file = yaml.full_load(stream)

        # 獲取圖片和遮罩路徑
        image_path = yaml_file['image_path']
        mask_path = yaml_file['mask_path']

        # 設定手部類別資訊
        categories_copy[0]["HandCategoriesId"] = yaml_file["annotation"]["HandType"]
        categories_copy[0]["GestureCategoriesId"] = yaml_file["gesture_id"]
        categories_copy[0]["AngleCategoriesId"] = yaml_file["angle_id"]
        categories_copy[1]["HandCategoriesId"] = yaml_file["annotation"]["ArmType"]

        # 獲取所有npy文件名稱的列表
        mask_files = glob.glob(os.path.join(mask_path, "*.npy"))

        # 按數字大小排序
        mask_files.sort(key=sort_by_number)

        # 逐一處理檔案
        for index, maskpath in enumerate(mask_files):
            # 取得影像編號
            number = int(re.findall(r'(\d+)\.(npy)', maskpath)[0][0])
            # 取得影像檔案路徑
            imagepath = image_path + "/{}.png".format(number)

            # 填入影像資訊到字典中
            data_image["images"].append({
                "image_path": imagepath,  # 影像檔案路徑
                "id": global_id,  # 影像編號
                "fid": number,  # 影像ID
                "camera": yaml_file["camera"]  # 相機資訊
            })

            # 填入標註資訊到字典中
            data_mask["mask"].append({
                "mask_path": maskpath,  # 標註檔案路徑
                "id": global_id,  # 標註編號
                "fid": number  # 影像ID
            })

            # 填入標註數據資訊到字典中
            data_anno["annotations"].append({
                "id": global_id,  # 標註編號
                "image_id": global_id,  # 影像編號
                "mask_id": global_id,  # 標註編號
                "categories": yaml_file["categories"],  # 物件類別
                "accessories": yaml_file["annotation"]["accessories"],  # 附屬品資訊
                "user": yaml_file["User"]  # 使用者資訊
            })

            # 更新全域編號
            global_id += 1

            # 如果已處理的資料筆數達到指定大小，就將字典轉換成 JSON 格式並寫入檔案中
            if global_id % BATCH_SIZE == 0:
                # 將字典轉換成 JSON 格式
                image_anno_json = json.dumps(data_image)
                mask_anno_json = json.dumps(data_mask)
                anno_json = json.dumps(data_anno)

                # 將 JSON 格式寫入文件
                with open(yaml_data['annotation_json'] + "/images/images_{}.json".format(global_id), "w") as f:
                    f.write(image_anno_json)
                with open(yaml_data['annotation_json'] + "/mask/mask_{}.json".format(global_id), "w") as f:
                    f.write(mask_anno_json)
                with open(yaml_data['annotation_json'] + "/annotation/annotation_{}.json".format(global_id), "w") as f:
                    print(yaml_data['annotation_json'] + "/annotation/annotation_{}.json".format(global_id))
                    f.write(anno_json)

                # 重新建立新的字典
                data_image = {"images": []}
                data_mask = {"mask": []}
                data_anno = {"annotations": []}

    # 將最後剩餘的字典輸出到 JSON 格式檔案
    if data_image["images"] and data_mask["mask"] and data_anno["annotations"]:
        image_anno_json = json.dumps(data_image)
        mask_anno_json = json.dumps(data_mask)
        anno_json = json.dumps(data_anno)
        with open(yaml_data['annotation_json'] + "/images/images_{}.json".format(global_id), "w") as f:
            f.write(image_anno_json)
        with open(yaml_data['annotation_json'] + "/mask/mask_{}.json".format(global_id), "w") as f:
            f.write(mask_anno_json)
        with open(yaml_data['annotation_json'] + "/annotation/annotation{}.json".format(global_id), "w") as f:
            f.write(anno_json)

