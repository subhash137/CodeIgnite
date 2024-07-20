from collections import defaultdict

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from flask import Flask, Response

app = Flask(__name__)

# Load the YOLO model with segmentation capabilities
model = YOLO("yolov8n-seg.pt")

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Open the video file
cap = cv2.VideoCapture("video1.mp4")

def generate():
    while True:
        # Read a frame from the video
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Create an annotator object to draw on the frame
        annotator = Annotator(im0, line_width=2)

        # Perform object tracking on the current frame
        results = model.track(im0, persist=True)

        # Check if tracking IDs and masks are present in the results
        if results[0].boxes.id is not None and results[0].masks is not None:
            # Extract masks and tracking IDs
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Annotate each mask with its corresponding tracking ID and color
            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=str(track_id))

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
