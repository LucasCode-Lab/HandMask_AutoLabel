import mediapipe as mp
import cv2
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, min_detection_confidence=0.1, min_tracking_confidence=0.1)
mpDraw = mp.solutions.drawing_utils


def detect_joints(image):
    """
    功能：在輸入的影像中偵測出手部關節點，並將其在影像上標記出。
    輸入：
        image：numpy陣列，表示輸入的影像。
    輸出：
        image：numpy陣列，表示處理後的影像，在影像上標記出關節點。
        points：二維數組，表示關節點的横縱座標，以[x,y]表示。如果未偵測到任何關節點，則返回空列表。
    """

    image = image.copy()
    # 處理輸入圖像
    results = hands.process(image)

    # 初始化手部關節點的清單
    points = []

    # 檢查是否存在多個手部的關節點
    if results.multi_hand_landmarks:
        # 遍歷每個手部的關節點
        for handLms in results.multi_hand_landmarks:
            # 檢查是否有21個關節點
            if len(handLms.landmark) != 21:
                continue
            # 將每個關節點的橫縱座標計算出來，存儲到points清單中
            points = [[int(lm.x * image.shape[1]), int(lm.y * image.shape[0])] for lm in handLms.landmark]
            # 繪製關節點之間的連接
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            # 退出迴圈，僅處理第一個手部
            break
    # 如果未檢測到任何關節點，返回圖像和空清單
    if not points:
        return image, []
    # 在圖像中標記兩個手指關節點
    cv2.circle(image, tuple(points[0]), 2, (0, 255, 0), 4)
    cv2.circle(image, tuple(points[9]), 2, (0, 255, 0), 4)
    # 返回處理後的圖像和關節點清單
    return image, points


def wrist_finger_vector_angle(wrist, finger):
    """
    計算手腕與手指的向量夾角

    :param wrist: 手腕的座標 (ndarray)
    :param finger: 手指的座標 (ndarray)
    :return: 向量夾角的弧度值 (float)
    """
    # 計算手腕與手指的向量
    wrist_finger_vector = finger - wrist

    # 設定垂直向量
    vertical_vector = np.array([0, -1])

    # 計算內積並除以向量長度
    cosine_angle = np.dot(wrist_finger_vector, vertical_vector) / (
            np.linalg.norm(wrist_finger_vector) * np.linalg.norm(vertical_vector))

    # 計算 wrist_finger_vector 與 vertical_vector 的叉積
    cross_product = np.cross(wrist_finger_vector, vertical_vector)

    if cross_product > 0:
        # wrist_finger_vector 是從 vertical_vector 逆時針方向旋轉得到的
        angle = np.arccos(cosine_angle)
    else:
        # wrist_finger_vector 是從 vertical_vector 順時針方向旋轉得到的
        angle = -np.arccos(cosine_angle)

    # # 將cosine_angle轉換為弧度值並返回
    # angle = np.arccos(cosine_angle)
    return angle


def rotate_matrix(angle_rad):
    """
    計算旋轉矩陣

    :param angle_rad: 弧度制角度值
    :return: 旋轉矩陣
    """
    rotate_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                              [np.sin(angle_rad), np.cos(angle_rad)]])
    return rotate_matrix


def cal_angle_rotatematrix(image, points):
    """
    計算旋转矩陣

    :param image: 圖片
    :param points: 包含手指頭和腕部位置的點的列表
    :return: 旋转矩陣
    """
    angle_rad = wrist_finger_vector_angle(np.array(points[0]), np.array(points[9]))
    rotate_matrix_res = rotate_matrix(angle_rad)

    return rotate_matrix_res


def rotate_points(points, rotate_matrix):
    """
    對指節點做旋轉

    Parameters:
        points (List[Tuple[int, int]]): 要旋轉的指節點坐標列表
        rotate_matrix (np.ndarray): 旋轉矩陣

    Returns:
        List[Tuple[int, int]]: 旋轉後的指節點坐標列表
    """
    # 計算第0個指節點的坐標
    base_point = points[0]

    # 遍歷其他指節點
    new_points = [base_point]
    for point in points[1:]:
        # 計算向量
        b1 = point[0] - base_point[0]
        b2 = point[1] - base_point[1]

        # 旋轉向量，加上基準點坐標計算新的指節點坐標
        new_point = np.dot(rotate_matrix, [b1, b2])
        new_point = (new_point[0] + base_point[0], new_point[1] + base_point[1])
        new_points.append(new_point)

    return new_points


def drawPoints(image, points):
    """
    繪製點在圖像上

    Parameters:
        image (np.ndarray): 要繪製點的圖像
        points (List[Tuple[int, int]]): 要繪製的點的坐標列表

    Returns:
        np.ndarray: 繪製完點的圖像
    """
    # 定義特殊點的顏色
    color_map = {0: (0, 255, 0), 9: (0, 255, 0)}

    # 遍歷所有點
    for i, point in enumerate(points):
        # 取出特殊點或預設顏色
        color = color_map.get(i, (0, 255, 255))
        # 繪製點
        cv2.circle(image, (int(point[0]), int(point[1])), 2, color, 4)

    return image


def createBoundingBox(image: np.ndarray, joints: np.ndarray) -> tuple:
    """
    在關節周圍繪製邊界框，並返回邊界框半徑長度。

    :param image: 輸入圖像。
    :type image: numpy array

    :param joints: 手指節點位置。
    :type joints: numpy array

    :return: 包含邊界框圖像和邊界框半徑長度。
    :rtype: tuple
    """

    # image = drawPoints(image, joints)
    image_circle = image.copy()

    # 計算關節之間和中心關節之間的最大距離
    maxLength = np.linalg.norm(joints[9] - np.array(joints), axis=1).max()

    # 計算邊界框半徑
    bboxRadius1 = maxLength * 0.6 + 1
    bboxRadius2 = maxLength * 1.1 + 1
    bboxRadius3 = maxLength * 0.8 + 1

    # 在特定關節周圍繪製圓形
    cv2.circle(image_circle, (int(joints[0][0]), int(joints[0][1])), int(bboxRadius1), (0, 0, 0), -1)
    cv2.circle(image_circle, (int(joints[9][0]), int(joints[9][1])), int(bboxRadius2), (0, 0, 0), -1)
    cv2.circle(image_circle, (int(joints[4][0]), int(joints[4][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(image_circle, (int(joints[8][0]), int(joints[8][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(image_circle, (int(joints[12][0]), int(joints[12][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(image_circle, (int(joints[16][0]), int(joints[16][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(image_circle, (int(joints[20][0]), int(joints[20][1])), int(bboxRadius3), (0, 0, 0), -1)
    return image_circle, maxLength


def extract_largest_contour_mask(original_mask: np.ndarray, joints: np.ndarray, bboxRadius: float) -> np.ndarray:
    """
    從二元遮罩中提取最大的輪廓，方法是減去一些圓形框住的區域。

    :param original_mask: 原始二元遮罩。
    :type original_mask: np.ndarray
    :param joints: 關節點的座標數組。
    :type joints: np.ndarray
    :param bboxRadius: 矩形框的半徑大小。
    :type bboxRadius: float
    :return: 最大輪廓的二元遮罩。
    :rtype: np.ndarray

    """
    # 複製原始遮罩以避免就地修改
    mask = original_mask.copy()

    # 設定圓形範圍，並在遮罩中把它們填成黑色
    bboxRadius1 = bboxRadius * 0.8 + 1
    cv2.circle(mask, (int(joints[0][0]), int(joints[0][1])), int(bboxRadius1), (0, 0, 0), -1)
    bboxRadius2 = bboxRadius * 1.1 + 1
    cv2.circle(mask, (int(joints[9][0]), int(joints[9][1])), int(bboxRadius2), (0, 0, 0), -1)
    bboxRadius3 = bboxRadius * 0.8 + 1
    cv2.circle(mask, (int(joints[4][0]), int(joints[4][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(mask, (int(joints[8][0]), int(joints[8][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(mask, (int(joints[12][0]), int(joints[12][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(mask, (int(joints[16][0]), int(joints[16][1])), int(bboxRadius3), (0, 0, 0), -1)
    cv2.circle(mask, (int(joints[20][0]), int(joints[20][1])), int(bboxRadius3), (0, 0, 0), -1)

    # 在遮罩中找到所有輪廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找不到任何輪廓，則返回原始遮罩
    if not contours:
        return original_mask

    # 選擇面積最大的輪廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 從原始遮罩中減去最大的輪廓
    result = original_mask.copy()
    cv2.drawContours(result, [largest_contour], -1, -1, -1)
    cv2.drawContours(result, [largest_contour], -1, (127, 127, 127), -1)

    return result


def save_mask_image(mask_image, yaml_data):
    """
    根據注釋資料中指定的手部類型，對輸入的遮罩影像進行值映射。
    :param mask_image: 輸入的遮罩影像。
    :param yaml_data: 包含 YAML 格式注釋資料的字典。字典的結構應遵循 YAML 檔案中指定的格式。

    :return: 映射後的遮罩影像。
    """
    # 從注釋資料中獲取手部類型
    hand_type = yaml_data["Annotation"]["HandType"]
    # 根據手部類型映射遮罩值
    mapping = {1: {225: 1, 127: 2}, 3: {255: 3, 127: 4}}[hand_type]
    # 創建一個與輸入遮罩影像大小相同的數組並用零填充
    unit_mask = np.zeros_like(mask_image)
    # 迭代 mapping 字典中的值與映射值
    for value, mapped_value in mapping.items():
        # 將單位遮罩中與當前映射字典中的值相同的值替換為映射值
        unit_mask[mask_image == value] = mapped_value
    # 返回生成的單位遮罩
    return unit_mask
