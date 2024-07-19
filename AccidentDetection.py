import cv2
from ultralytics import YOLO
import torch

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLO model
model = YOLO('best.pt').to(device)  # Ensure the model is loaded onto the GPU if available

# Open the video file
video_path = 'video3.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video writer initialized to save the output video
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Frame resizing factor
resize_factor = 0.5

# Frame skipping factor
frame_skip = 2
frame_count = 0

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
            print(label)

    # Write the frame to the output video
    out.write(frame)

    # Optionally, display the frame for preview
    cv2.imshow('YOLO Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
