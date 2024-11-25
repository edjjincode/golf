import numpy as np
import cv2
import os
import json

def calculate_angle(a, b, c):
    """
    세 점의 좌표를 사용하여 두 벡터 사이의 각도를 계산합니다.
    a, b, c는 각각 (x, y) 형태의 좌표입니다.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def save_debug_log(log_data, frame_count, log_dir):
    """
    디버그 로그 데이터를 JSON 파일로 저장합니다.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{frame_count}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)

def save_debug_frame(image, frame_count, frame_dir, detections=None, pose_landmarks=None):
    """
    디버그 프레임 이미지를 저장합니다.
    YOLO의 바운딩 박스와 MediaPipe 랜드마크를 시각화하여 저장합니다.
    """
    os.makedirs(frame_dir, exist_ok=True)
    debug_image = image.copy()

    # YOLO 바운딩 박스 그리기
    if detections:
        for det in detections:
            x_min, y_min, x_max, y_max = map(int, det["bbox"])
            cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(debug_image, det["class_name"], (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # MediaPipe 랜드마크 그리기
    if pose_landmarks:
        mp_drawing = __import__('mediapipe').solutions.drawing_utils
        mp_pose = __import__('mediapipe').solutions.pose
        mp_drawing.draw_landmarks(
            debug_image,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    frame_path = os.path.join(frame_dir, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
