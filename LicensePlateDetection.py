import cv2
import pytesseract
from ultralytics import YOLO
import torch

import cv2
import numpy as np
from scipy.signal import convolve2d


# tesseract_cmd =r'C:/Program Files/Tesseract-OCR/tesseract.exe' # Change this path if using Windows (e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe')


# pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# Load the YOLOv8 model
import easyocr
reader = easyocr.Reader(['en'])


model = YOLO('license_model.pt')  # Replace with your YOLOv8 model path

def detect_and_ocr(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    # Loop through detections and perform OCR
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            conf = bbox.conf[0]
            cls = int(bbox.cls[0])
            label = result.names[cls]
            
            plate_img = img[y1:y2, x1:x2]
            cv2.imwrite('licenseplate.png',plate_img)
            # text = pytesseract.image_to_string(plate_img)
            result = reader.readtext(plate_img)


            print(f'Detected license plate: {result[0][1]}')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, result[0][1], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the result image
    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



if __name__ == '__main__':
    # Test the function
    detect_and_ocr('car.png')  # Replace with your image path

    
