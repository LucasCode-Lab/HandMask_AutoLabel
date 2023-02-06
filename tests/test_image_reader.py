import os
import numpy as np
import unittest
import yaml
from PIL import Image
from src.image_processing.image_reader import read_and_binarize_images, binarize, load_yaml_config


class TestImageReader(unittest.TestCase):
    def test_load_images(self):
        # 定義影像目錄的路徑
        image_dir = "../images/test_images/"
        # 讀取影像
        images = read_and_binarize_images(image_dir)
        # 檢查是否有至少一張影像被讀取
        assert len(images) > 0, f"無法讀取影像: {image_dir}"

    def test_binarize(self):
        image = np.array([[100, 150, 200], [100, 150, 200], [100, 150, 200]], dtype=np.uint8)
        threshold = 150
        expected = np.array([[0, 0, 255], [0, 0, 255], [0, 0, 255]], dtype=np.uint8)

        result = binarize(image, threshold)
        self.assertTrue(np.array_equal(result, expected))


    def test_binarize_threshold_not_int(self):
        # Arrange
        image = np.array([[100, 200, 150], [120, 180, 210], [50, 60, 70]])
        threshold = "150"

        # Act and Assert
        with self.assertRaises(TypeError) as context:
            result = binarize(image, threshold)

        self.assertEqual(str(context.exception), "threshold should be an integer.")

    def test_load_yaml_config_success(self):
        file_path = 'sample_config.yaml'
        with open(file_path, 'w') as f:
            yaml.dump({'key': 'value'}, f)
        result = load_yaml_config(file_path)
        expected = {'key': 'value'}
        self.assertDictEqual(result, expected)
        os.remove(file_path)

    def test_load_yaml_config_fail(self):
        file_path = 'nonexistent_file.yaml'
        with self.assertRaises(AssertionError) as context:
            load_yaml_config(file_path)
        self.assertEqual(str(context.exception), "Yaml File Not Exist!!")


# Run the tests if the module is run as a script
if __name__ == '__main__':
    unittest.main()