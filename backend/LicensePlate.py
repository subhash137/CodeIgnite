from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import easyocr

import cv2
import numpy as np
from scipy.signal import convolve2d

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

model = YOLO('license_model.pt')  # Replace with your YOLOv8 model path
reader = easyocr.Reader(['en'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model(img)
        plates = []

        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                plate_img = img[y1:y2, x1:x2]
                result = reader.readtext(plate_img)
                text = result[0][1] if result else 'No text detected'

                plates.append({
                    'text': text,
                    'image': cv2.imencode('.png', plate_img)[1].tobytes()
                })

        response = {
            'plates': plates
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True,port=8080)
