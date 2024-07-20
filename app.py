from flask import Flask, Response, render_template, send_from_directory, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import torch
import os
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app)  # Enable CORS for all routes

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

# Create a directory to save snapshots
snapshot_dir = 'snapshots'
os.makedirs(snapshot_dir, exist_ok=True)

# List to store detected accident snapshots
accident_snapshots = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snapshots/<filename>')
def snapshots(filename):
    return send_from_directory(snapshot_dir, filename)

@app.route('/get_accidents')
def get_accidents():
    return jsonify(accident_snapshots)

def generate():
    global frame_count
    snapshot_taken = False
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

        snapshot_taken = False

        # Variable to track if an accident is detected
        accident_detected = False

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

                if not snapshot_taken:
                    snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{frame_count}.jpg')
                    cv2.imwrite(snapshot_filename, frame)
                    print(f'Snapshot saved as: {snapshot_filename}')
                    snapshot_taken = True

                # Detect an accident (you can customize the condition based on your criteria)
                if label == 'accident' and confidence > 0.5:  # Assuming 'accident' is a class label
                    accident_detected = True
                    socketio.emit('accident_detected', {'message': 'Accident detected!'})

        # Take snapshot if an accident is detected
        if accident_detected:
            snapshot_filename = f'snapshot_{frame_count}.jpg'
            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            print(f'Snapshot saved as: {snapshot_path}')
            accident_snapshots.append(snapshot_filename)
            socketio.emit('accident_detected', {'message': 'Accident detected!'})

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

# if accident_detected:
#     socketio.emit('accident_detected', {'message': 'Accident detected!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
