import torch
import cv2
from ultralytics import YOLO

# YOLO 모델 로드
def load_yolo_model(model_path):
    model = YOLO(model_path)
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

            # Get the label from the model's class names
            label = model.model.names[cls] if cls in model.model.names else "unknown"

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
    model_path = 'best (1).pt'
    video_path = 'car-crash.mov'

    # YOLO 모델 로드
    model = load_yolo_model(model_path)

    # 동영상 처리
    process_video(video_path, model)
