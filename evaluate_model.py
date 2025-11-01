import cv2
import numpy as np
from detection import AccidentDetectionModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_JSON_FILE = "model.json"
MODEL_WEIGHTS_FILE = "model_weights.keras"
VIDEO_PATH = "test_video.mp4" # IMPORTANT: Use a video the model has NOT seen before
CONFIDENCE_THRESHOLD = 85  
ACCIDENT_FRAMES = [(120,168)] # Example: Accident happens between frame 550 and 650


def is_frame_an_accident(frame_number, accident_ranges):
    """Checks if a frame number falls within any of the true accident ranges."""
    for start, end in accident_ranges:
        if start <= frame_number <= end:
            return True
    return False

def main():
    # Load the model
    model = AccidentDetectionModel(MODEL_JSON_FILE, MODEL_WEIGHTS_FILE)
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    y_true = []  # List to store the true labels (1 for Accident, 0 for No Accident)
    y_pred = []  # List to store the model's predictions

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("Finished processing video.")
            break

        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        # Get model prediction
        pred_class, prob = model.predict_accident(roi[np.newaxis, :, :])
        prob_percent = prob[0][0] * 100

        # --- LOGIC TO DETERMINE PREDICTION ---
        model_says_accident = (pred_class == "Accident" and prob_percent > CONFIDENCE_THRESHOLD)

        # --- LOGIC TO DETERMINE GROUND TRUTH ---
        truth_is_accident = is_frame_an_accident(frame_count, ACCIDENT_FRAMES)

        # Append results to our lists
        y_true.append(1 if truth_is_accident else 0)
        y_pred.append(1 if model_says_accident else 0)

        frame_count += 1

    video.release()

    # --- CALCULATE AND PRINT METRICS ---
    if not y_true:
        print("No frames were processed. Cannot calculate metrics.")
        return
        
    print("\n--- Model Evaluation Report ---")
    # Use scikit-learn's classification_report for a detailed summary
    # target_names tells the report what to call class 0 and class 1
    # report = classification_report(y_true, y_pred, target_names=["No Accident", "Accident"])
    # This is the fix:
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=["No Accident", "Accident"], 
        labels=[0, 1]  # <--- ADD THIS PART
    )
    print(report)

    # Generate and display a Confusion Matrix for a visual report
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Optional: Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Accident", "Accident"], yticklabels=["No Accident", "Accident"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()