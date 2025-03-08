import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Configuration values
video_path = "C:\\Users\\zhieu\\OneDrive\\Documents\\btl_xla\\yolov9\\Data_ext\\video.mp4"
conf_threshold = 0.5
tracking_class = 0

# Initialize DeepSort tracker
tracker = DeepSort(max_age=5)

# Initialize device
device = torch.device("cpu")

# Load YOLO model
try:
    model = YOLO("weights/yolov9c.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load class names
try:
    with open("Data_ext/classes.name", "r", encoding='utf-8') as f:
        class_name = f.read().strip().split("\n")
except FileNotFoundError:
    print("Classes file not found!")
    class_name = ["object"]  # Default class if file not found

# Generate random colors for classes
colors = np.random.randint(0, 255, size=(len(class_name), 3))

# Initialize video capture with alternative method
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")
except Exception as e:
    print(f"Video capture error: {e}")
    exit()

while True:
    # Read frame with error handling
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame")
        break

    # Ensure frame is not empty
    if frame is None:
        print("Empty frame")
        break

    # Detect objects
    try:
        results = model(frame)
    except Exception as e:
        print(f"Detection error: {e}")
        continue

    detect = []
    for result in results[0].boxes:
        # Extract detection information
        bbox = result.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, bbox[:4])
        confidence = result.conf.item()
        class_id = int(result.cls.item())

        # Apply filtering based on confidence and tracking class
        if tracking_class is not None and class_id != tracking_class:
            continue
        
        if confidence < conf_threshold:
            continue

        # Prepare detection for DeepSort
        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update tracks
    tracks = tracker.update_tracks(detect, frame=frame)

    # Visualize tracks
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_tlbr()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.get_det_class()
            
            color = colors[class_id]
            B, G, R = map(int, color)

            label = f"{class_name[class_id]}:{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("Object Tracking", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()