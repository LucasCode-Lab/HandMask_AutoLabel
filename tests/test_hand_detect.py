import cv2
import unittest
from ..src.autolabel.hand_detect import detect_joints

class TestJointDetection(unittest.TestCase):
    def setUp(self):
        # 在每次測試開始前設置初始條件，例如讀取圖像
        self.image = cv2.imread('test_image.jpg')

    def test_detect_joints(self):
        # 調用 detect_joints 函數
        processed_image, points = detect_joints(self.image)

        # 測試返回的關節點清單是否不是空列表
        self.assertTrue(points)
        # 測試返回的關節點清單長度是否等於 21
        self.assertEqual(len(points), 21)
        # 測試返回的圖像大小是否等於輸入圖像大小
        self.assertEqual(processed_image.shape, self.image.shape)

    def test_no_joints_detected(self):
        # 使用一張無關節點的圖像作為輸入
        image = cv2.imread('no_joints_image.jpg')
        # 調用 detect_joints 函數
        processed_image, points = detect_joints(image)

        # 測試返回的關節點清單是否是空列表
        self.assertEqual(points, [])
        # 測試返回的圖像大小是否等於輸入圖像大小
        self.assertEqual(processed_image.shape, image.shape)

if __name__ == '__main__':
    unittest.main()