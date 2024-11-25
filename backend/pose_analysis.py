from ultralytics import YOLO
import mediapipe as mp
import cv2
import os
import json
from utils import calculate_angle

# YOLO 및 MediaPipe 초기화
model = YOLO("/Users/jinchan/edjjincode/Project/final_golf/backend/models/best.pt")
classes = {0: "Address", 1: "Backswing", 2: "Downswing", 3: "Impact", 4: "Finish"}
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 디버그 저장 경로 설정
DEBUG_LOGS_DIR = "/Users/jinchan/edjjincode/Project/final_golf/backend/debug_logs"
DEBUG_FRAMES_DIR = "/Users/jinchan/edjjincode/Project/final_golf/backend/debug_frames"
os.makedirs(DEBUG_LOGS_DIR, exist_ok=True)
os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)

pose_angle_ranges = {
    "Address": {
        "ShoulderDifference": (0, 10),
        "HipDifference": (0, 10),
        "SpineAngle": (20, 30),
        "KneeAngle": (140, 160),
    },
    "Backswing": {
        "LeftArmAngle": (150, 180),
        "ShoulderRotation": (30, 70),
        "RightKneeAngle": (150, 170),
    },
    "Downswing": {
        "ArmAngleChange": (0, 10),
        "PelvisRotation": (10, 40),
    },
    "Impact": {
        "ArmStraightness": (170, 180),
        "HeadMovement": (0, 5),
    },
    "Finish": {
        "LeftFootWeight": (0.1, float("inf")),
        "FinalArmAngle": (160, 180),
    },
}

def analyze_frame(image, frame_count):
    """
    YOLO와 MediaPipe를 사용하여 자세를 분석하고, 디버그 데이터를 저장합니다.
    """
    # YOLO로 자세 분류
    yolo_results = model(image)[0].boxes.data.tolist()
    detected_pose = None
    detections = []
    for result in yolo_results:
        x_min, y_min, x_max, y_max, confidence, cls = result
        detected_pose = classes.get(int(cls), "Unknown")
        detections.append({
            "class_name": detected_pose,
            "bbox": [x_min, y_min, x_max, y_max],
            "confidence": confidence
        })

    if detected_pose == "Unknown":
        save_debug_log(frame_count, detections, {}, [])
        return {"pose": "Unknown", "metrics": {}}

    # MediaPipe로 포즈 추출
    pose_results = pose.process(image)
    if not pose_results.pose_landmarks:
        save_debug_log(frame_count, detections, {}, [])
        return {"pose": detected_pose, "metrics": {}}

    landmarks = pose_results.pose_landmarks.landmark
    metrics = calculate_pose_metrics(landmarks, detected_pose)

    # 디버그 데이터 저장
    save_debug_frame(image, detections, pose_results, frame_count)
    save_debug_log(frame_count, detections, metrics, landmarks)

    return {"pose": detected_pose, "metrics": metrics}

def calculate_pose_metrics(landmarks, detected_pose):
    """
    자세별 메트릭을 계산합니다.
    """
    metrics = {}

    if detected_pose == "Address":
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        metrics["ShoulderDifference"] = abs(right_shoulder[1] - left_shoulder[1])
        metrics["HipDifference"] = abs(right_hip[1] - left_hip[1])
        metrics["SpineAngle"] = calculate_angle(right_shoulder, right_hip, [right_hip[0], right_hip[1] + 1])
        metrics["KneeAngle"] = calculate_angle(right_hip, right_knee, right_ankle)

    elif detected_pose == "Backswing":
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

        metrics["LeftArmAngle"] = calculate_angle(left_shoulder, left_elbow, left_wrist)
        metrics["ShoulderRotation"] = calculate_angle(left_shoulder, right_shoulder, right_hip)
        metrics["RightKneeAngle"] = calculate_angle(right_hip, right_knee, [right_knee[0], right_knee[1] + 1])

    elif detected_pose == "Downswing":
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        metrics["ArmAngleChange"] = calculate_angle(left_hip, left_elbow, left_wrist)
        metrics["PelvisRotation"] = abs(right_hip[0] - left_hip[0])

    elif detected_pose == "Impact":
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        metrics["ArmStraightness"] = calculate_angle(right_shoulder, right_elbow, right_wrist)
        metrics["HeadMovement"] = abs(nose[1] - right_shoulder[1])

    elif detected_pose == "Finish":
        left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        metrics["LeftFootWeight"] = abs(left_foot[0] - right_foot[0])
        metrics["FinalArmAngle"] = calculate_angle(left_wrist, left_foot, right_foot)

    return metrics

def save_debug_frame(image, detections, pose_results, frame_count):
    """
    이미지에 YOLO 결과와 MediaPipe 랜드마크를 그려 디버그 프레임으로 저장합니다.
    """
    debug_image = image.copy()
    for det in detections:
        x_min, y_min, x_max, y_max = map(int, det["bbox"])
        cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(debug_image, det["class_name"], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if pose_results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(debug_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    frame_path = os.path.join(DEBUG_FRAMES_DIR, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

def save_debug_log(frame_count, detections, metrics, landmarks):
    """
    분석 결과를 JSON 파일로 저장합니다.
    """
    log_data = {
        "frame": frame_count,
        "detections": detections,
        "metrics": metrics,
        "landmarks": [
            {"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks
        ] if landmarks else []
    }
    log_path = os.path.join(DEBUG_LOGS_DIR, f"log_{frame_count}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=4)
