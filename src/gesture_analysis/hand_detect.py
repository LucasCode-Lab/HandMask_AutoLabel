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

    # 將cosine_angle轉換為弧度值並返回
    angle = np.arccos(cosine_angle)
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


def createBoundingBox(image, joints, rotationMatrix):
    """
    Draw a bounding box around the given joints and return the box corners.

    Parameters:
    image (numpy array): The input image.
    joints (numpy array): The positions of joints.
    rotationMatrix (numpy array): The rotation matrix.

    Returns:
    numpy array, numpy array: The input image with bounding box and the box corners.
    """
    image = drawPoints(image, joints)

    # Calculate the maximum length between joints and the center joint
    maxLength = np.linalg.norm(joints[9] - np.array(joints), axis=1).max()
    # Calculate the bounding box radius
    bboxRadius = maxLength * 1.1 + 1
    leftUp = (int(joints[9][0] - bboxRadius), int(joints[9][1] - bboxRadius))
    rightDown = (int(joints[9][0] + bboxRadius), int(joints[9][1] + bboxRadius))
    # Calculate the corners of the bounding box
    bboxCorners = np.array([
        [leftUp[0], leftUp[1]],
        [leftUp[0], rightDown[1]],
        [rightDown[0], leftUp[1]],
        [rightDown[0], rightDown[1]],
    ]) - joints[0]
    bboxCorners = np.dot(np.linalg.inv(rotationMatrix), bboxCorners.T).T + joints[0]
    # Draw the bounding box on the image
    cv2.line(image, tuple(bboxCorners[0].astype(int)), tuple(bboxCorners[1].astype(int)), (0, 0, 255), 1)
    cv2.line(image, tuple(bboxCorners[0].astype(int)), tuple(bboxCorners[2].astype(int)), (0, 0, 255), 1)
    cv2.line(image, tuple(bboxCorners[1].astype(int)), tuple(bboxCorners[3].astype(int)), (0, 0, 255), 1)
    cv2.line(image, tuple(bboxCorners[2].astype(int)), tuple(bboxCorners[3].astype(int)), (0, 0, 255), 1)

    return image, bboxCorners