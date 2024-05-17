import torch
import cv2
import base64
from ultralytics import YOLO
from torchvision.models import densenet121
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# YOLO 모델 로드
def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

# DenseNet 모델 로드
def load_densenet_model(densenet_model_path):
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    try:
        model.load_state_dict(torch.load(densenet_model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model.to(device)
    model.eval()
    return model, device

# 프레임을 base64 문자열로 변환
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_str = base64.b64encode(buffer).decode('utf-8')
    return frame_str

# 동영상 처리 및 예측
def process_video(video_path, densenet_model, yolo_model, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15
    frame_skip = round(fps / target_fps)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to achieve the target FPS
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        # Preprocess the frame for DenseNet
        input_tensor = transform(frame).unsqueeze(0).to(device)

        # Predict using DenseNet model
        with torch.no_grad():
            output = densenet_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            accident = 'No Accident' if predicted.item() == 1 else 'Accident'
            confidence = confidence.item()

        if accident == 'Accident':
            # Perform inference on the frame using YOLO
            results = yolo_model(frame)[0]
            yolo_class = "unknown"
            for result in results.boxes:
                cls = int(result.cls[0].item())
                if cls in yolo_model.model.names:
                    yolo_class = yolo_model.model.names[cls]
                    break

            # Convert frame to base64 string
            frame_str = frame_to_base64(frame)

            # Print the detected accident and confidence
            print(f'Frame {frame_count} - {accident} (Confidence: {confidence:.2f}), YOLO Class: {yolo_class}')

            # Prepare data for printing
            result = {
                "confidence": confidence,
                "accident_detected": accident,
                "frame": frame_count,
                "yolo_class": yolo_class,
                "frame_base64": frame_str
            }

            # Print the result
            print(result)

        frame_count += 1

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 함수
if __name__ == "__main__":
    yolo_model_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\crash-detect code\\YOLOv8 best.pt'
    densenet_model_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\code\\densenet_model10.pth'
    video_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\crash-detect code\\car-crash.mov'

    # YOLO 모델 로드
    yolo_model = load_yolo_model(yolo_model_path)

    # DenseNet 모델 로드
    densenet_model, device = load_densenet_model(densenet_model_path)

    # 동영상 처리 및 예측
    process_video(video_path, densenet_model, yolo_model, device)
