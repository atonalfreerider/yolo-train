import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("model.pt")

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or specify a different index if you have multiple cameras

# Loop through the webcam frames
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to capture frame from webcam.")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
