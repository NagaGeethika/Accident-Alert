🚨 Road Accident Alert System

An AI-powered system that detects road accidents in real time from video feeds and automatically alerts emergency services with the exact GPS location.
Its goal is to reduce emergency response time and help save lives through faster alerts.

PROJECT WORKING VIDEO LINK : https://drive.google.com/file/d/1n7aZtyR44B3EE5vHNglinlcnEQftdeME/view?usp=drivesdk
CALL RECORDING AUDIO LINK  : https://drive.google.com/file/d/14ZdcRmPgPQ94WMaVArTkmlYsZGofmVL4/view?usp=drivesdk

⚙️ How It Works (6 Simple Steps)

Video Input: Captures live or recorded video using OpenCV.

AI Detection: Analyzes each frame with a trained VGG16 deep learning model to detect accidents.

Alert Trigger: Saves a snapshot when an accident is detected.

GUI Confirmation: A Tkinter pop-up asks the operator to confirm the emergency.

Location Services: Uses Geopy to convert GPS coordinates into a readable address.

Emergency Call: The Twilio API makes an automated voice call with the location details.

🧩 Tech Stack
🔹 AI & Machine Learning

TensorFlow & Keras – Used to build and train the VGG16 model with Transfer Learning.

Scikit-learn – For metrics like Precision, Recall, and F1-Score, and to handle class imbalance.

🔹 Computer Vision

OpenCV – For real-time video capture, frame analysis, and live detection display.

🔹 Cloud Communication & Location

Twilio API – Makes automated voice calls to emergency responders.

TwiML – Generates dynamic voice messages with location details.

Geopy – Converts GPS coordinates into human-readable addresses.

🔹 GUI & Security

Tkinter – Creates the confirmation pop-up window.(Due to limited free trails in Twilio)

Dotenv (.env) – Securely stores API keys and credentials.

✅ This project combines deep learning, computer vision, and cloud APIs to create a real-time alert system that can make roads safer through intelligent automation.
