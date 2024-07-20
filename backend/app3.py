# from flask import Flask, Response, render_template, send_from_directory, jsonify,request
# from flask_cors import CORS
# import cv2
# from ultralytics import YOLO
# import torch
# import os
# from flask_socketio import SocketIO
# import easyocr
# import cv2
# import numpy as np
# from scipy.signal import convolve2d
# from twilio.rest import Client
# from geopy.distance import geodesic

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")

# CORS(app) 

# # Twilio credentials
# account_sid = 'ACf265a7fc0a5c92850d9a664e796343e5'
# auth_token = '9adb31b646cec076031f28ed816e5fc0'
# twilio_number = '+12513253346'
# client = Client(account_sid, auth_token)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = YOLO('best.pt').to(device)  

# video_path = 'video3.mp4'
# cap = cv2.VideoCapture(video_path)

# resize_factor = 0.5

# frame_skip = 2
# frame_count = 0

# snapshot_dir = 'snapshots'
# os.makedirs(snapshot_dir, exist_ok=True)

# accident_snapshots = []

# # CCTV device location
# cctv_location = {'latitude': 12.9716, 'longitude': 77.5946} 

# # List of nearby hospitals with their locations and phone numbers
# hospitals = [
#     {'name': 'Hospital A', 'location': {'latitude': 12.9732, 'longitude': 77.6017}, 'phone': '+918121405305'},
#     {'name': 'Hospital B', 'location': {'latitude': 12.9751, 'longitude': 77.6092}, 'phone': '+918121405405'},
#     # Add more hospitals as needed
# ]

# def find_nearest_hospital(cctv_location):
#     nearest_hospital = None
#     min_distance = float('inf')
    
#     for hospital in hospitals:
#         distance = geodesic(
#             (cctv_location['latitude'], cctv_location['longitude']),
#             (hospital['location']['latitude'], hospital['location']['longitude'])
#         ).kilometers
#         if distance < min_distance:
#             min_distance = distance
#             nearest_hospital = hospital
    
#     return nearest_hospital

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/snapshots/<filename>')
# def snapshots(filename):
#     return send_from_directory(snapshot_dir, filename)

# @app.route('/get_accidents')
# def get_accidents():
#     return jsonify(accident_snapshots)

# # Function to send SMS
# def send_sms_to_hospital(message, hospital_number):
#     client.messages.create(
#         body=message,
#         from_=twilio_number,
#         to=hospital_number
#         media_url=[f'http://yourserver.com/snapshots/{os.path.basename(snapshot_path)}']
#     )

# def generate():
#     global frame_count
#     snapshot_taken = False
#     notification_sent = False
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         resized_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

#         results = model(resized_frame)

#         snapshot_taken = False

#         accident_detected = False

#         for result in results:
#             boxes = result.boxes.xyxy  
#             confidences = result.boxes.conf  
#             class_ids = result.boxes.cls 

#             for box, confidence, class_id in zip(boxes, confidences, class_ids):
#                 x1, y1, x2, y2 = map(int, box / resize_factor) 
#                 label = model.names[int(class_id)]
#                 confidence = float(confidence)

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
#                 # cv2.putText(frame, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#                 if not snapshot_taken:
#                     snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{frame_count}.jpg')
#                     cv2.imwrite(snapshot_filename, frame)
#                     print(f'Snapshot saved as: {snapshot_filename}')
#                     socketio.emit('accident_detected', {'message': 'Accident detected!'})
#                     snapshot_taken = True

#                 if label == model.names[int(class_id)] and confidence > 0.5: 
#                     accident_detected = True

#         print(f"Accident detected: {accident_detected}")
#         if accident_detected:
#             print("Viswaksena")
#             snapshot_filename = f'snapshot_{frame_count}.jpg'
#             snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
#             cv2.imwrite(snapshot_path, frame)
#             print(f'Snapshot saved as: {snapshot_path}')
#             accident_snapshots.append(snapshot_filename)
#             print(accident_snapshots)
#             if not notification_sent:
#                 nearest_hospital = find_nearest_hospital(cctv_location)
#                 message = f'Accident detected! Location: {cctv_location["latitude"]}, {cctv_location["longitude"]}'
#                 socketio.emit('accident_detected', {'message': message, 'location': cctv_location})
#                 send_sms_to_hospital(message, nearest_hospital['phone'])
#                 notification_sent = True
#             socketio.emit('accident_detected', {'message': 'Accident detected!'})

#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue

#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')



from flask import Flask, Response, render_template, send_from_directory, jsonify, request
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import torch
import os
from flask_socketio import SocketIO
import easyocr
import numpy as np
from scipy.signal import convolve2d
from twilio.rest import Client
from geopy.distance import geodesic
import base64
from vertexai.generative_models import GenerativeModel, Image
import uuid
import requests
from PIL import Image as PilImage



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'

socketio = SocketIO(app, cors_allowed_origins="*")

CORS(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Twilio credentials
account_sid = 'ACfffbe36870ac0c6e91de67059ebb2f45'
auth_token = '52f4ac8c8ba40adbe998f33886fd2bc9'
twilio_number = '+12076302613'
client = Client(account_sid, auth_token)

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

# CCTV device location
cctv_location = {'latitude': 12.9716, 'longitude': 77.5946}

# List of nearby hospitals with their locations and phone numbers
hospitals = [
    {'name': 'Hospital A', 'location': {'latitude': 12.9732, 'longitude': 77.6017}, 'phone': '+918121405305'},
    {'name': 'Hospital B', 'location': {'latitude': 12.9751, 'longitude': 77.6092}, 'phone': '+918121405405'},
    # Add more hospitals as needed
]

def find_nearest_hospital(cctv_location):
    nearest_hospital = None
    min_distance = float('inf')

    for hospital in hospitals:
        distance = geodesic(
            (cctv_location['latitude'], cctv_location['longitude']),
            (hospital['location']['latitude'], hospital['location']['longitude'])
        ).kilometers
        if distance < min_distance:
            min_distance = distance
            nearest_hospital = hospital

    return nearest_hospital

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/snapshots/<filename>')
def snapshots(filename):
    return send_from_directory(snapshot_dir, filename)

# @app.route('/get_accidents')
# def get_accidents():
#     return jsonify(accident_snapshots)

@app.route('/get_accidents')
def get_accidents():
    try:
        snapshots = os.listdir(snapshot_dir)
        return jsonify(snapshots)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# @app.route('/get_accidents')
# def get_accidents():
#     try:
#         snapshots = os.listdir(snapshot_dir)
#         return jsonify(snapshots)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# Function to send SMS
def send_sms_to_hospital(hospital_number, text_message=None, image_path=None):
    try:
        if image_path:
            if not os.path.exists(image_path):
                print(f"Error: Image file does not exist at path: {image_path}")
                return

            # Resize and compress the image
            with PilImage.open(image_path) as img:
                img.thumbnail((640, 480))  # Resize to max 640x480
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=70)
                buffer.seek(0)

            # Upload the image to a temporary file hosting service
            response = requests.post('https://tmpfiles.org/api/v1/upload', files={'file': ('image.jpg', buffer, 'image/jpeg')})
            
            if response.status_code == 200:
                image_url = response.json()['data']['url']
                # Replace 'upload' with 'dl' in the URL to get the direct download link
                image_url = image_url.replace('upload', 'dl')
                
                print(f"Image uploaded successfully. URL: {image_url}")
                
                # Combine text message and image URL
                full_message = f"{text_message}\n\nAccident snapshot: {image_url}"
                
                message = client.messages.create(
                    body=full_message,
                    from_=twilio_number,
                    to=hospital_number
                )
                print(f"Message with location and image URL sent successfully. SID: {message.sid}")
                
                # Fetch message details to confirm
                message_details = client.messages(message.sid).fetch()
                print(f"Message details: {message_details.to}, {message_details.status}")
                
            else:
                print(f"Failed to upload image. Status code: {response.status_code}")
                print(f"Response content: {response.text}")
        elif text_message:
            message = client.messages.create(
                body=text_message,
                from_=twilio_number,
                to=hospital_number
            )
            print(f"Text message sent successfully. SID: {message.sid}")

    except Exception as e:
        print(f"Error sending message: {str(e)}")
        print(f"Error details: {type(e).__name__}, {str(e)}")


def generate():
    global frame_count
    snapshot_taken = False
    notification_sent = False
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

        accident_image_sent = False

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

                # if not snapshot_taken:
                #     snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{frame_count}.jpg')
                #     cv2.imwrite(snapshot_filename, frame)
                #     print(f'Snapshot saved as: {snapshot_filename}')
                #     socketio.emit('accident_detected', {'message': 'Accident detected!'})
                #     snapshot_taken = True

                if label == model.names[int(class_id)] and confidence > 0.5:
                    accident_detected = True

        # print(f"Accident detected: {accident_detected}")
        # if accident_detected:
        #     snapshot_filename = f'snapshot_{frame_count}.jpg'
        #     snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
        #     cv2.imwrite(snapshot_path, frame)
        #     print(f'Snapshot saved as: {snapshot_path}')
        #     accident_snapshots.append(snapshot_filename)
        #     if not notification_sent:
        #         nearest_hospital = find_nearest_hospital(cctv_location)
        #         message = f'Accident detected! Location: {cctv_location["latitude"]}, {cctv_location["longitude"]}'
        #         socketio.emit('accident_detected', {'message': message, 'location': cctv_location})
        #         send_sms_to_hospital(message, nearest_hospital['phone'], snapshot_path)
        #         notification_sent = True

        # ret, jpeg = cv2.imencode('.jpg', frame)
        # if not ret:
        #     continue

        # frame = jpeg.tobytes()
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if accident_detected and not accident_image_sent and not notification_sent:
            snapshot_filename = f'accident_snapshot_{frame_count}.jpg'
            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            print(f'Accident snapshot saved as: {snapshot_path}')
            print(f'Snapshot file exists: {os.path.exists(snapshot_path)}')
            
            nearest_hospital = find_nearest_hospital(cctv_location)
            text_message = f'Accident detected!\nLocation: {cctv_location["latitude"]}, {cctv_location["longitude"]}\nMaps: http://maps.google.com/maps?q={cctv_location["latitude"]},{cctv_location["longitude"]} '

            socketio.emit('accident_detected', {'message': text_message, 'location': cctv_location})

            # Send text message with location and image URL
            send_sms_to_hospital(nearest_hospital['phone'], text_message=text_message, image_path=snapshot_path)
            notification_sent = True
            accident_image_sent = True

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

model2 = YOLO('license_model.pt')  # Replace with your YOLOv8 model path
reader = easyocr.Reader(['en'])

license_plate_db = {
    "CCC444": {
        "address": "123 Main St, Mumbai, India",
        "father_phone": "+918121405305",
        "father_name": "Anupam",
        "name": "Kalpana",
        # "father_name": "Michael Doe",
        # "father_phone": "+918121405305",
        # "address": "123 Main St, Anytown, USA"
    },
    "RT BB221": {
        "name": "Jane Smith",
        "father_name": "Robert Smith",
        "father_phone": "+918121405305",
        "address": "456 Oak Ave, Otherville, USA"
    }
}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        results = model2(img)
        plates = []

        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                plate_img = img[y1:y2, x1:x2]
                result = reader.readtext(plate_img)
                text = result[0][1] if result else 'No text detected'

                # Encode the image as base64
                _, buffer = cv2.imencode('.png', plate_img)
                plate_img_base64 = base64.b64encode(buffer).decode('utf-8')

                details = license_plate_db.get(text)

                plate_entry = {
                    'text': text,
                    'image': plate_img_base64
                }

                if details:
                    plate_entry['details'] = details
                    # Send accident notification only for detected plates in the database
                    send_accident_notification(details['father_phone'], text)

                plates.append(plate_entry)

        response = {
            'plates': plates
        }
        print(response)

        return jsonify(response)
    
def send_accident_notification(phone_number, license_plate):
    message = f"An accident has occurred involving vehicle with license plate {license_plate}."
    
    # Here you would typically use an SMS service API
    # For demonstration, we'll just print the message
    print(f"Sending SMS to {phone_number}: {message}")
    
    # If you want to use an actual SMS service, you could use something like Twilio:
    # from twilio.rest import Client
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=message,
        from_=twilio_number,
        to=phone_number
    )
    
# @app.route('/upload_image', methods=['POST'])
# def handle_upload():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part in the request'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file:
#         img = file.read()
#         print(img)
#         vision_model = GenerativeModel("gemini-pro-vision")
#         # image = Image.load_from_file(img)
#         output = vision_model.generate_content(["describe about the accident in image?", img])
#         print(output)
#         text = output.candidates[0].content.parts[0].text
#         return jsonify(text)

import io
@app.route('/upload_image', methods=['POST'])
def handle_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Read the file into memory
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)

        # Process the image with Vertex AI
        vision_model = GenerativeModel("gemini-pro-vision")
        image = Image.from_bytes(in_memory_file.getvalue())
        output = vision_model.generate_content(["describe about the accident in image?", image])
        text = output.candidates[0].content.parts[0].text

        return jsonify({'text': text})
    
if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8000, debug=True)




# model2 = YOLO('license_model.pt')  # Replace with your YOLOv8 model path
# reader = easyocr.Reader(['en'])

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file:
#         img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
#         results = model2(img)
#         plates = []

#         for result in results:
#             for bbox in result.boxes:
#                 x1, y1, x2, y2 = map(int, bbox.xyxy[0])
#                 plate_img = img[y1:y2, x1:x2]
#                 result = reader.readtext(plate_img)
#                 text = result[0][1] if result else 'No text detected'

#                 plates.append({
#                     'text': text,
#                     'image': cv2.imencode('.png', plate_img)[1].tobytes()
#                 })

#         response = {
#             'plates': plates
#         }

#         return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=8000, debug=True)

# from flask import Flask, Response, render_template, send_from_directory, jsonify, request
# from flask_cors import CORS
# import cv2
# from ultralytics import YOLO
# import torch
# import os
# from flask_socketio import SocketIO, emit
# from twilio.rest import Client
# from geopy.distance import geodesic

# app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")

# CORS(app)  # Enable CORS for all routes

# # Twilio credentials
# account_sid = 'ACf265a7fc0a5c92850d9a664e796343e5'
# auth_token = '9adb31b646cec076031f28ed816e5fc0'
# twilio_number = '+12513253346'
# client = Client(account_sid, auth_token)

# # Check if CUDA is available
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load the YOLO model
# model = YOLO('best.pt').to(device)  # Ensure the model is loaded onto the GPU if available

# # Open the video file
# video_path = 'video3.mp4'
# cap = cv2.VideoCapture(video_path)

# # Frame resizing factor
# resize_factor = 0.5

# # Frame skipping factor
# frame_skip = 2
# frame_count = 0

# # Create a directory to save snapshots
# snapshot_dir = 'snapshots'
# os.makedirs(snapshot_dir, exist_ok=True)

# # List to store detected accident snapshots
# accident_snapshots = []

# # CCTV device location
# cctv_location = {'latitude': 12.9716, 'longitude': 77.5946}  # Example location

# # List of nearby hospitals with their locations and phone numbers
# hospitals = [
#     {'name': 'Hospital A', 'location': {'latitude': 12.9732, 'longitude': 77.6017}, 'phone': '+918121405305'},
#     {'name': 'Hospital B', 'location': {'latitude': 12.9751, 'longitude': 77.6092}, 'phone': '+918121405405'},
#     # Add more hospitals as needed
# ]

# def find_nearest_hospital(cctv_location):
#     nearest_hospital = None
#     min_distance = float('inf')
    
#     for hospital in hospitals:
#         distance = geodesic(
#             (cctv_location['latitude'], cctv_location['longitude']),
#             (hospital['location']['latitude'], hospital['location']['longitude'])
#         ).kilometers
#         if distance < min_distance:
#             min_distance = distance
#             nearest_hospital = hospital
    
#     return nearest_hospital

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/snapshots/<filename>')
# def snapshots(filename):
#     return send_from_directory(snapshot_dir, filename)

# @app.route('/get_accidents')
# def get_accidents():
#     return jsonify(accident_snapshots)

# # Function to send SMS
# def send_sms_to_hospital(message, hospital_number):
#     client.messages.create(
#         body=message,
#         from_=twilio_number,
#         to=hospital_number
#     )

# def generate():
#     global frame_count
#     snapshot_taken = False
#     notification_sent = False

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         # Resize the frame for faster processing
#         resized_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

#         # Perform object detection
#         results = model(resized_frame)

#         snapshot_taken = False

#         # Process results
#         for result in results:
#             boxes = result.boxes.xyxy  # Extract bounding box coordinates
#             confidences = result.boxes.conf  # Extract confidences
#             class_ids = result.boxes.cls  # Extract class IDs

#             for box, confidence, class_id in zip(boxes, confidences, class_ids):
#                 x1, y1, x2, y2 = map(int, box / resize_factor)  # Scale coordinates back to original frame size
#                 label = model.names[int(class_id)]
#                 confidence = float(confidence)

#                 # Draw the bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

#                 if not snapshot_taken:
#                     # snapshot_filename = f'snapshot_{frame_count}.jpg'
#                     snapshot_filename = os.path.join(snapshot_dir, f'snapshot_{frame_count}.jpg')
#                     # snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
#                     cv2.imwrite(snapshot_filename, frame)
#                     # accident_snapshots.append({
#                     #     'filename': snapshot_filename,
#                     #     'location': cctv_location
#                     # })
#                     accident_snapshots.append(snapshot_filename)
#                     print(f'Snapshot saved as: {accident_snapshots}')
#                     snapshot_taken = True

#                     if not notification_sent:
#                         nearest_hospital = find_nearest_hospital(cctv_location)
#                         message = f'Accident detected! Location: {cctv_location["latitude"]}, {cctv_location["longitude"]}'
#                         socketio.emit('accident_detected', {'message': message, 'location': cctv_location})
#                         send_sms_to_hospital(message, nearest_hospital['phone'])
#                         notification_sent = True

#         # Encode the frame as JPEG
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         if not ret:
#             continue

#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     socketio.run(app, host='0.0.0.0', port=8000, debug=True)

