
# 분류 정확도는 이후에 생각하고 YOLO detection 모델

import torch
import cv2
import json
from ultralytics import YOLO

# YOLO 모델 로드 및 새로운 클래스 이름 설정
def load_yolo_model_with_new_classes(model_path, class_names_path):
    # YOLO 모델 로드
    model = YOLO(model_path)

    # 새로운 클래스 이름을 파일에서 다시 읽어오기
    with open(class_names_path, "r") as f:
        new_class_names = json.load(f)

    # 클래스 이름을 모델의 내부 속성에 설정
    model.model.names = new_class_names

    # 클래스 이름 확인 및 출력
    print("새로운 클래스 이름:")
    print(type(model.model.names), len(model.model.names))
    print(model.model.names)

    return model

# 동영상 처리 및 예측
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15
    frame_skip = round(fps / target_fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to achieve the target FPS
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Perform inference on the frame
        results = model(frame)[0]

        # Process the results
        detected_objects = []
        for result in results.boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0].item()
            cls = int(result.cls[0].item())

            # Print detected class for debugging
            print(f"Detected class: {cls}")

            # Check if cls exists in model.model.names
            if str(cls) in model.model.names:
                label = model.model.names[str(cls)]
            else:
                label = "unknown"

            # Append detected object details for output
            detected_objects.append((cls, label, conf))

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Print detected objects details
        for obj in detected_objects:
            cls, label, conf = obj
            print(f"Detected class: {cls}, Label: {label}, Confidence: {conf:.2f}")

            # Check if the detected label is an accident class
            if label in ['Car-to-Bicycle-Crash', 'Car-to-Car-Crash', 'Car-to-Motorcycle-Crash', 'Car-to-Pedestrian-Crash']:
                print(f"Frame {frame_count} - Accident: {label}")

        frame_count += 1

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 함수
if __name__ == "__main__":
    model_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\crash-detect code\\finally_yolo13.pt'
    class_names_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\crash-detect code\\class_names.json'
    #video_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\crash-detect code\\car-crash.mov'
    vedio_path='C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\crash-detect code\\car_bike.mp4'

    # YOLO 모델 로드 및 새로운 클래스 이름 설정
    model = load_yolo_model_with_new_classes(model_path, class_names_path)

    # 동영상 처리
    process_video(video_path, model)