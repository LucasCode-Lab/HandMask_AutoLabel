import mediapipe as mp
import cv2

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
