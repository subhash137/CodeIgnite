from flask import Flask, Response, render_template, send_from_directory, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import torch
import os
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app) 


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('best.pt').to(device)  

video_path = 'video3.mp4'
cap = cv2.VideoCapture(video_path)

resize_factor = 0.5

frame_skip = 2
frame_count = 0

snapshot_dir = 'snapshots'
os.makedirs(snapshot_dir, exist_ok=True)

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

        resized_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        results = model(resized_frame)

        snapshot_taken = False

        accident_detected = False

        for result in results:
            boxes = result.boxes.xyxy  
            confidences = result.boxes.conf  
            class_ids = result.boxes.cls 

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box / resize_factor) 
                label = model.names[int(class_id)]
                confidence = float(confidence)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                if not snapshot_taken:
                    snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{frame_count}.jpg')
                    cv2.imwrite(snapshot_filename, frame)
                    print(f'Snapshot saved as: {snapshot_filename}')
                    socketio.emit('accident_detected', {'message': 'Accident detected!'})
                    snapshot_taken = True

                if label == model.names[int(class_id)] and confidence > 0.5: 
                    accident_detected = True

        print(f"Accident detected: {accident_detected}")
        if accident_detected:
            print("Viswaksena")
            snapshot_filename = f'snapshot_{frame_count}.jpg'
            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            print(f'Snapshot saved as: {snapshot_path}')
            accident_snapshots.append(snapshot_filename)
            print(accident_snapshots)
            socketio.emit('accident_detected', {'message': 'Accident detected!'})

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
