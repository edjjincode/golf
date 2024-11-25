# import os
# import io
# import json
# import base64
# import datetime
# from fastapi import FastAPI, WebSocket
# from PIL import Image, ImageDraw
# import numpy as np
# from ultralytics import YOLO
# import mediapipe as mp

# app = FastAPI()

# # 디버그 데이터 저장 경로
# DEBUG_FRAMES_DIR = "/Users/jinchan/edjjincode/Project/final_golf/backend/debug_frames"
# DEBUG_LOGS_DIR = "/Users/jinchan/edjjincode/Project/final_golf/backend/debug_logs"

# # 폴더 생성
# os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
# os.makedirs(DEBUG_LOGS_DIR, exist_ok=True)

# # YOLO 및 MediaPipe 초기화
# model = YOLO("/Users/jinchan/edjjincode/Project/final_golf/backend/models/best.pt")
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # 클래스 이름 정의
# classes = {0: "Address", 1: "Backswing", 2: "Downswing", 3: "Impact", 4: "Finish"}

# # 각도 계산 함수
# def calculate_angle(a, b, c):
#     ba = np.array(a) - np.array(b)
#     bc = np.array(c) - np.array(b)
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
#     return angle


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     frame_count = 0

#     try:
#         while True:
#             # WebSocket 데이터 수신
#             data = await websocket.receive_text()

#             # "end" 신호가 들어오면 루프 종료
#             if data == "end":
#                 print("WebSocket received end signal. Closing connection.")
#                 await websocket.close()
#                 break

#             # Base64 이미지 디코딩
#             image_data = base64.b64decode(data.split(",")[1])
#             image = Image.open(io.BytesIO(image_data)).convert("RGB")
#             image_np = np.array(image)
#             frame_count += 1

#             # 5 프레임마다 처리
#             if frame_count % 5 == 0:
#                 # YOLO 모델로 자세 분류
#                 yolo_results = model(image_np)[0].boxes.data.tolist()
#                 detections = []

#                 for result in yolo_results:
#                     x_min, y_min, x_max, y_max, confidence, cls = result
#                     class_id = int(cls)
#                     class_name = classes.get(class_id, "Unknown")
#                     detections.append({
#                         "class_id": class_id,
#                         "class_name": class_name,
#                         "bbox": [x_min, y_min, x_max, y_max]
#                     })

#                 # MediaPipe로 포즈 추출
#                 pose_results = pose.process(image_np)
#                 keypoints, angles = [], {}

#                 if pose_results.pose_landmarks:
#                     landmarks = pose_results.pose_landmarks.landmark

#                     # 특정 각도 계산 (예: 왼쪽 팔 각도)
#                     left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#                     left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#                     left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#                     angles["left_arm_angle"] = calculate_angle(left_shoulder, left_elbow, left_wrist)

#                 # 디버그 데이터 저장
#                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                 frame_path = os.path.join(DEBUG_FRAMES_DIR, f"frame_{timestamp}_{frame_count}.jpg")
#                 log_path = os.path.join(DEBUG_LOGS_DIR, f"log_{timestamp}_{frame_count}.json")

#                 image.save(frame_path)
#                 with open(log_path, "w") as log_file:
#                     json.dump({"detections": detections, "angles": angles}, log_file, indent=4)

#                 print(f"Saved frame to {frame_path}")
#                 print(f"Saved log to {log_path}")

#                 # WebSocket으로 분석 결과 전송
#                 await websocket.send_json({"detections": detections, "angles": angles})

#     except Exception as e:
#         print(f"WebSocket Error: {e}")
#         await websocket.close()

from fastapi import FastAPI, WebSocket
from pose_analysis import analyze_frame, pose_angle_ranges
import base64
import numpy as np
from PIL import Image
import io
import json

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    frame_count = 0  # 각 연결마다 독립적으로 관리
    results_summary = {}

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            if data.strip() == "end":
                print("WebSocket received end signal. Closing connection.")
                summary_text = format_results(results_summary)
                await websocket.send_text(summary_text)
                await websocket.close()
                break

            # Base64로 인코딩된 이미지를 디코딩
            try:
                image_data = base64.b64decode(data.split(",")[1])
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                image_np = np.array(image)
            except Exception as e:
                await websocket.send_text(f"Error decoding image: {e}")
                continue

            # 프레임 분석 및 결과 업데이트
            frame_count += 1
            try:
                result = analyze_frame(image_np, frame_count)
                update_results_summary(results_summary, result)
            except Exception as e:
                await websocket.send_text(f"Error analyzing frame: {e}")
                continue

    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()

# 요약 결과 업데이트
def update_results_summary(results_summary, result):
    pose = result["pose"]
    metrics = result["metrics"]

    if pose not in results_summary:
        results_summary[pose] = {}

    for metric, value in metrics.items():
        range_min, range_max = pose_angle_ranges[pose].get(metric, (None, None))
        in_range = range_min <= value <= range_max if range_min is not None and range_max is not None else True
        if metric not in results_summary[pose]:
            results_summary[pose][metric] = []
        results_summary[pose][metric].append((value, in_range, range_min, range_max))

# 결과 요약 형식화
def format_results(results_summary):
    formatted = ""
    for pose, metrics in results_summary.items():
        formatted += f"{pose}:\n"
        for metric, values in metrics.items():
            for value, in_range, range_min, range_max in values:
                if in_range:
                    formatted += f"  {metric} 값 {value:.2f}는 범위 {range_min}-{range_max} 안에 있습니다.\n"
                else:
                    adjustment = abs(value - (range_max if value > range_max else range_min))
                    formatted += f"  {metric} 값 {value:.2f}는 범위를 벗어났습니다. {adjustment:.2f} 정도 조정해야 합니다.\n"
    return formatted
