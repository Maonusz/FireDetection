import cv2
from ultralytics import YOLO
import cvzone

# Load the ONNX model with the correct path
model = YOLO('/home/pi/FireDetection/best.onnx')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly 'ret' is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize frame to reduce load
    frame = cv2.resize(frame, (320, 240))

    # Run inference on the captured frame
    results = model(frame)

    # Process and display the results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
