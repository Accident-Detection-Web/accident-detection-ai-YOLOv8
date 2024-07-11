import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_cors import CORS
from flask_jwt_extended import *
from werkzeug.utils import secure_filename
import os
import subprocess
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
from ultralytics import YOLO
from datetime import datetime
import base64
import ffmpeg
import re
import mysql.connector
from mysql.connector import Error
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}

# JWT 매니저 설정
jwt = JWTManager(app)

# 허용된 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# DenseNet 모델 로드
def load_densenet_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# YOLO 모델 로드 및 새로운 클래스 이름 설정
def load_yolo_model_with_new_classes(model_path, class_names_path):
    model = YOLO(model_path)
    with open(class_names_path, "r") as f:
        new_class_names = json.load(f)
    model.model.names = new_class_names
    print("새로운 클래스 이름:")
    print(type(model.model.names), len(model.model.names))
    print(model.model.names)
    return model

# 도분초 방식의 gps 좌표를 십진수 방식으로 변환
def dms_to_decimal(dms):
    match = re.match(r'(\d+) deg (\d+)\u0027 (\d+\.\d+)" ([NSEW])', dms)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms}")    
    degrees = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)    
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal *= -1    
    return decimal

# 동영상 파일에서 gps추출
def extract_gps_data(video_path):
    command = ['exiftool', '-GPSLatitude', '-GPSLongitude', '-json', video_path]    
    result = subprocess.run(command, text=True, capture_output=True)    
    if result.stdout:
        metadata = json.loads(result.stdout)
        if metadata and 'GPSLatitude' in metadata[0] and 'GPSLongitude' in metadata[0]:
            latitude_dms = metadata[0]['GPSLatitude']
            longitude_dms = metadata[0]['GPSLongitude']
            latitude = dms_to_decimal(latitude_dms)
            longitude = dms_to_decimal(longitude_dms)
            return latitude, longitude    
    print("No GPS data available in the video.")
    return 0, 0

# database에 데이터 추가
def insert_accident_data(accident_info):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='ai_capstone',
            user='root',
            password='Abcd123@'
        )
        cursor = connection.cursor()
        insert_query = """
        INSERT INTO accidents (image, accident, latitude, longitude, date, sort, severity)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            accident_info['imagePath'], accident_info['accident'], accident_info['latitude'], 
            accident_info['longitude'], accident_info['date'], accident_info['sort'], accident_info['severity']
        ))
        connection.commit()
        print("Accident data inserted successfully.")
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

# 프레임을 base64 문자열로 변환
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    frame_str = base64.b64encode(buffer).decode('utf-8')
    return frame_str

# 비디오 처리
def process_video(source, densenet_model, yolo_model, device, gps_info):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return ['Failed to open video source']

    fps = cap.get(cv2.CAP_PROP_FPS)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_count = 0
    results = []
    folder_path = 'C:accident-detection-ai\\img'
    frame_times = []
    accident_count = 0
    frame_skip = 1  # 초기 frame_skip 값

    while cap.isOpened() and accident_count < 5:
        ret, frame = cap.read()
        if not ret:
            break        
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad(): 
                output = densenet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                accident = 0 if predicted.item() == 1 else 1.
                if accident == 1: # 사고발생하면
                    accident_count += 1
                    if(accident_count == 5):
                        #이미지 저장
                        filename = f'accident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        full_path = os.path.join(folder_path, filename)
                        cv2.imwrite(full_path, frame)

                        # YOLO 모델로 프레임 분석
                        results_yolo = yolo_model(frame)[0]
                        yolo_class = "unknown"
                        for result in results_yolo.boxes:
                            cls = int(result.cls[0].item())
                            if str(cls) in yolo_model.model.names:
                                yolo_class = yolo_model.model.names[str(cls)]
                                break

                        # 이미지 경로, 사고여부, GPS, 위도, 경도 정보 추가
                        lat, lon = gps_info if gps_info != 'No GPS info available' else ('0', '0')
                        accident_info = {
                            "imagePath": full_path,
                            "accident": accident,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "sort" : yolo_class,
                            "severity" : confidence.item()  # DenseNet에서 나온 충돌 Confidence 값을 넣음
                        }
                        #db에 내용추가
                        insert_accident_data(accident_info)
                        results.append(accident_info)
                        return results # 영상은 백에 데이터 전송 후 바로 종료
                    frame_skip = 1  # 사고가 발생하면 다음 프레임 검사
                else:
                    frame_skip = 5  # 사고가 없으면 5 프레임 후 검사        
        frame_count += 1
    cap.release()
    return results

# 비디오 링크 업로드 라우트
@app.route('/api/v1/public/upload-link', methods=['GET', 'POST'])
def upload_link():
    if 'video_link' not in request.json:
        return jsonify({'error': 'No video link provided'}), 400
    video_link = request.json['video_link']
    try:
        densenet_model, device = load_densenet_model()
        yolo_model = load_yolo_model_with_new_classes('path_to_yolo_model.pt', 'path_to_class_names.json')
        gps_info = request.form.get('gps_info', '')
        results = process_video(video_link, densenet_model, yolo_model, device, gps_info)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# 비디오 파일 업로드 라우트
@app.route('/api/v1/public/upload-video', methods=['GET', 'POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        densenet_model, device = load_densenet_model()
        yolo_model = load_yolo_model_with_new_classes('path_to_yolo_model.pt', 'path_to_class_names.json')
        gps_info = extract_gps_data(file_path)
        results = process_video(file_path, densenet_model, yolo_model, device, gps_info)
        return jsonify(results)
    else:
        flash('File not allowed or missing')
        return redirect(request.url)
    
if __name__ == '__main__':
    densenet_model, device = load_densenet_model()
    yolo_model = load_yolo_model_with_new_classes('path_to_yolo_model.pt', 'path_to_class_names.json')
    app.run(debug=True)
