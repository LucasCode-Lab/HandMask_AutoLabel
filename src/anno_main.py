from autolabel import annotation
from utils.logger import configure_logging
import argparse
from utils.file_manage import load_yaml_config
logger = configure_logging(__name__)

parser = argparse.ArgumentParser(description='Process yaml config file path.')
# 加入命令列參數 --config，並設定目標變數名稱為 config_path
parser.add_argument('--config', dest='config_path', required=True, help='Path to the yaml config file')
# 解析命令列參數
args = parser.parse_args()
# 將命令列參數中的 YAML 檔案位置存到變數 yaml_config_path
yaml_config_path = args.config_path
# 使用 logger 記錄器，記錄 YAML 檔案位置
logger.info("Yaml 檔案位置: {}".format(yaml_config_path))
# 使用 load_yaml_config 函數讀取 YAML 配置檔，並存到變數 yaml_data
yaml_data = load_yaml_config(yaml_config_path)
annotation.annotation_res(yaml_data)
