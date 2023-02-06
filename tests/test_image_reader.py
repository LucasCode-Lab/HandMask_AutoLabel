import os
import unittest
from PIL import Image
from src.image_processing.image_reader import read_images


class TestImageReader(unittest.TestCase):
    def test_read_images(self):
        # 定義影像目錄的路徑
        image_dir = "../images/test_images/"
        # 讀取影像
        images = read_images(image_dir)
        # 檢查是否有至少一張影像被讀取
        assert len(images) > 0, f"無法讀取影像: {image_dir}"


# Run the tests if the module is run as a script
if __name__ == '__main__':
    unittest.main()