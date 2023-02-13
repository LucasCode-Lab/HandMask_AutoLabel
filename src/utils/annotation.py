import glob
import os
import json
import yaml

from utils.logger import logger
from image_processing.image_processor import sort_by_number


def genannotation(yaml_data):
    user = yaml_data['Class_param']['User']
    gesture_id = yaml_data['Class_param']['gesture_id']
    angle_id = yaml_data['Class_param']['angle_id']
    # 創建yaml資訊
    data = {

        "image_path": yaml_data["output_dir"]["raw_dir"].format(user, gesture_id, angle_id),
        "camera": {
            "id": 0,
            "retify": False,
            "width": 640,
            "height": 400
        },
        "mask": {
            "mask_path": yaml_data["output_dir"]["unit_mask_dir"].format(user, gesture_id, angle_id)
        },
        "annotation": {
            "left_hand_accessories": yaml_data['Annotation']['left_hand_accessories'],
            "right_hand_accessories": yaml_data['Annotation']['right_hand_accessories'],
            "left_arm_accessories": yaml_data['Annotation']['left_arm_accessories'],
            "right_arm_accessories": yaml_data['Annotation']['right_arm_accessories']
        },
        "angle": {
            "angle_id": yaml_data['Class_param']['angle_id']
        }
    }

    # 寫入 YAML 檔
    with open(yaml_data['output_dir']['dir'].format(user, gesture_id, angle_id)+"/data.yaml", "w") as outfile:
        yaml.dump(data, outfile)


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
