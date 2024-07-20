# CodeIgnite
# Accident Detection and Response System

## Introduction

Road accidents are a global concern, causing millions of injuries and fatalities each year. The World Health Organization reports that approximately 1.3 million people die annually due to road traffic crashes. In this context, the application of Computer Vision and Artificial Intelligence has become crucial in enhancing road safety and emergency response systems.

Our project leverages these technologies to create a comprehensive accident detection and response system, aiming to reduce response times and potentially save lives.

## Project Overview

This system utilizes advanced computer vision techniques and AI to detect accidents, identify vehicles, and generate detailed reports. Here's a breakdown of its key features:

### 1. Real-time Accident Detection and Notification

- Uses YOLO (You Only Look Once) for real-time accident detection in video streams.
- Captures an immediate snapshot when an accident is detected.
- Sends SMS alerts to nearby police stations and hospitals using Twilio.
- Includes accident image, GPS coordinates, and a brief "Accident Detection" message in the alert.
- Calculates the nearest emergency services using Haversine distance.

### 2. License Plate Detection

- Employs YOLO for accurate license plate detection in uploaded images.
- Utilizes EasyOCR to extract text from the detected license plates.

### 3. Advanced Accident Scene Analysis

- Leverages the Gemini Vision Pro model, a Large Language Model (LLM), to generate detailed descriptions of accident scenes from snapshots.
- Provides comprehensive information to emergency responders, enabling them to prepare adequately before arriving at the scene.

This feature represents a significant advancement in accident response systems. By providing detailed, AI-generated descriptions of accident scenes, emergency services can:

- Better assess the severity of the situation
- Allocate resources more effectively
- Prepare specific medical equipment or personnel based on the description
- Potentially save more lives through faster and more informed responses

## Technology Stack

- Frontend: React
- Backend: Flask, Python
- SMS Service: Twilio
- Geolocation: Geopy
- OCR: EasyOCR
- Computer Vision: Ultralytics (YOLO), Torch, NumPy
- LLM: Google's Gemini Vision Pro model
- Additional Libraries: SciPy, Flask-SocketIO, Flask-CORS

## Setup and Installation

### Prerequisites

1. Create a Twilio account for SMS messaging. You'll need:
   - `account_sid`
   - `auth_token`
   - `twilio_number`

### Backend Setup

1. Open a terminal and navigate to the backend folder.
2. Run the Flask application:
   ```
   python app.py
   ```

### Frontend Setup

1. Open another terminal and navigate to the frontend folder.
2. Install the required packages:
   ```
   npm i
   ```
3. Start the React application:
   ```
   npm start
   ```

## Usage

The web application consists of three main tabs:

1. **Accident Detection**: Monitors real-time video feed, detects accidents, and sends alerts.
2. **License Plate Detection**: Allows users to upload images for license plate detection and text extraction.
3. **Report Generation**: Enables users to upload crash site images for detailed AI-generated descriptions.

## Conclusion

This Accident Detection and Response System represents a significant step forward in leveraging AI and computer vision for public safety. By providing real-time accident detection, rapid notification, and detailed scene analysis, it has the potential to significantly reduce emergency response times and improve outcomes for accident victims.
