import yaml
from utils import logger
import argparse
from image_processing.image_reader import load_yaml_config, read_and_binarize_images


parser = argparse.ArgumentParser(description='Process yaml config file path.')
parser.add_argument('--config', dest='config_path', required=True, help='Path to the yaml config file')
args = parser.parse_args()

yaml_config_path = args.config_path
logger.logger.info("Yaml 檔案位置: {}".format(yaml_config_path))
yaml_data = load_yaml_config(yaml_config_path)

image_dir = yaml_data['image_dir']
images_list, bin_image_list = read_and_binarize_images(image_dir)
