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


def search_yaml(yaml_data):
    yaml_path = []
    for dirpath, dirnames, filenames in os.walk(yaml_data["annotation_json"]):
        for filename in filenames:
            if filename.endswith(".yaml"):
                file_path = os.path.join(dirpath, filename)
                yaml_path.append(file_path)
    logger.info("Yaml位置：{}".format(yaml_path))
    return yaml_path

def annotation_res(yaml_data):
    data = {
        "info": {
            "description": "CRI 2023 Hand Datasets",
            "version": "1.0",
            "date": "2023-01-31",
            "organization": "Coretronic Reality Inc. AR/MR AI Technology Division"
        }
    }
    data.setdefault("images", [])

    yaml_path = search_yaml(yaml_data)
    global_id = 0
    for image_path in yaml_path:

        with open(image_path, "r") as stream:
            yaml_file = yaml.full_load(stream)
        image_path = yaml_file['image_path']
        # 獲取所有影像文件名稱的列表
        image_files = glob.glob(os.path.join(image_path, "*.png")) + glob.glob(os.path.join(image_path, "*.jpg"))
        # 按數字大小排序
        image_files.sort(key=sort_by_number)
        for index, filename in enumerate(image_files):
            data["images"].append({"image_path": filename, "id": global_id, "fid": index, "camera": yaml_file["camera"]})
            global_id += 1

    # 將字典轉換成 JSON 格式
    json_data = json.dumps(data)

    # 將 JSON 格式寫入文件
    with open(yaml_data['annotation_json']+"/data.json", "w") as f:
        f.write(json_data)
