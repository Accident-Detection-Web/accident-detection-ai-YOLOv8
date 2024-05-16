

import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn
from PIL import Image

# Load the DenseNet model
def load_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    try:
        model.load_state_dict(torch.load('C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\code\\densenet_model10.pth', map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model.to(device)
    model.eval()
    return model, device

# Process and predict each frame
def process_video(video_path, model, device):
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print(f"Error opening video: {e}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15
    frame_skip = round(fps / target_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, target_fps, (width, height))

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

        # Preprocess the frame
        input_tensor = transform(frame).unsqueeze(0).to(device)

        # Predict using the model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            accident = 'No Accident' if predicted.item() == 1 else 'Accident'
            confidence = confidence.item()

        # Draw prediction on the frame
        if accident == 'Accident':
            cv2.putText(frame, f'Accident {confidence:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            print(f'Frame {frame_count} - {accident} (Confidence: {confidence:.2f})')
        else:
            cv2.putText(frame, 'No Accident', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(f'Frame {frame_count} - {accident}')

        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Main function to load the model and process the video
if __name__ == "__main__":
    model, device = load_model()
    video_path = 'C:\\2024\\On-Campus Activities\\Capstone\\2024_Capstone\\Capstone Project\\code\\car_car (1).mp4' # 이 영상에서의 대부분의 충돌을 잘 감지함  
    process_video(video_path, model, device)