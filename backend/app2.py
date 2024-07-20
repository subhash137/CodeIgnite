from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import torch
import os

app = Flask(__name__)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO model
model = YOLO('best.pt').to(device)  # Ensure the model is loaded onto the GPU if available

# Open the video file
video_path = 'video3.mp4'
cap = cv2.VideoCapture(video_path)

# Frame resizing factor
resize_factor = 0.5

# Frame skipping factor
frame_skip = 2
frame_count = 0

@app.route('/')
def index():
    return render_template('index.html')

def generate():
    global frame_count
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # Perform object detection
        results = model(resized_frame)

        # Process results
        for result in results:
            boxes = result.boxes.xyxy  # Extract bounding box coordinates
            confidences = result.boxes.conf  # Extract confidences
            class_ids = result.boxes.cls  # Extract class IDs

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box / resize_factor)  # Scale coordinates back to original frame size
                label = model.names[int(class_id)]
                confidence = float(confidence)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
