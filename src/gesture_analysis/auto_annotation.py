import os
import json
import time
import yaml
from natsort import natsorted

yaml_path = "/home/lucas/hand/medi/config/create_annotation.yaml"
assert os.path.isfile(yaml_path), "Yaml File Not Exist!!"
with open(yaml_path, "r") as stream:
    yaml_data = yaml.load(stream)

sample_json_root = yaml_data["SAMPLE_JSON_ROOT"]
image_folder = yaml_data["IMAGE_FOLDER"]
assert os.path.isfile(sample_json_root), "Sample Json file not exist!!"
assert os.path.isfile(image_folder), "Images Folder not exist!!"

gesture_id = yaml_data["GESTURE_ID"]
category_id_arm = -1
category_id_hand = yaml_data["CATEGORY_ID"]
if category_id_hand == 3:
    category_id_arm = 7
elif category_id_hand == 4:
    category_id_arm = 8
elif category_id_hand == 1:
    category_id_arm = 5
elif category_id_hand == 2:
    category_id_arm = 6

f = open(sample_json_root)
p = json.load(f)

# 獲取現在時間
t = time.localtime()
current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)

# 添加 annotation images 資訊
image_name = os.listdir(image_folder)
for filename in natsorted(os.listdir(image_folder)):
    json_text_images = {
        "file_name": "/home/lucas/hand/medi/datasets/images/{}".format(filename),
        "id": "{}".format(filename.split('.')[0]),
        "camera": "fisheye",
        "retify": "False",
        "width": 640,
        "height": 400,
        "date_captured": "{}".format(current_time)
    }
    # 添加 annotation annotation 資訊
    json_text_annotation = {
        "image_id": "{}".format(filename.split('.')[0]),
        "category_id": category_id_hand,
        "gesture_id": gesture_id,
        "postition_id": 0
    }, {
        "image_id": "{}".format(filename.split('.')[0]),
        "category_id": category_id_arm,
    }
    p["images"].append(json_text_images)
    p["annotations"].append(json_text_annotation)
json_obj = json.dumps(p, indent=4)

json_root = "/home/lucas/hand/medi/annotation/"
new_json = "Hand_mask.json"
if not os.path.isfile(json_root + new_json):
    os.mknod(json_root + new_json)
g = open("/home/lucas/hand/medi/annotation/" + new_json, "r+")
g.write(json_obj)

g.close()
f.close()
