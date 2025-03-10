import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from ultralytics.nn.tasks import DetectionModel
import time
import threading
import os

# Add DetectionModel to safe globals
torch.serialization.add_safe_globals([DetectionModel])

# Config value
video_path = "C:\\Users\\zhieu\\OneDrive\\Documents\\btl_xla\\yolov9\\Data_ext\\CarsMoving.mp4"
conf_threshold = 0.5
tracking_class = None  # None để theo dõi tất cả các lớp đối tượng

# Khởi tạo deepsort tracker
tracker = DeepSort(max_age=5)

# Khởi tạo YOLOv9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights="weights/yolov9c.pt", device=device, fuse=True)
model = AutoShape(model)

# Load class name
with open("yolov9/Data_ext/classes.name") as f:
    class_name = f.read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(class_name), 3))

# Hàm tính toán IOU (Intersection Over Union)
def calculate_iou(box1, box2):
    """
    Tính toán IOU giữa hai hộp giới hạn
    box1, box2: [x1, y1, x2, y2]
    """
    # Tọa độ của giao điểm
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Tính diện tích giao điểm
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Tính diện tích của hai hộp
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Tính IOU
    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

# Lớp xử lý video để cải thiện hiệu suất
class VideoProcessor:
    def __init__(self, video_path, model, tracker, conf_threshold, tracking_class=None, frame_skip=2):
        self.cap = cv2.VideoCapture(video_path)
        self.model = model
        self.tracker = tracker
        self.conf_threshold = conf_threshold
        self.tracking_class = tracking_class
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.tracks = []
        self.latest_frame = None
        self.processed_frame = None
        self.running = True
        self.last_time = time.time()  # Khởi tạo last_time ngay từ đầu
        
        # Lấy các thông số video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Khởi động thread xử lý
        self.processing_thread = threading.Thread(target=self.process_video)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_video(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
                
            self.latest_frame = frame.copy()
            self.frame_count += 1
            
            # Bỏ qua một số frame để tăng tốc
            if self.frame_count % self.frame_skip != 0:
                continue
                
            # Resize frame để xử lý nhanh hơn
            frame_resized = cv2.resize(frame, (720, 480))
            
            # Phát hiện đối tượng với YOLOv9
            with torch.no_grad():  # Tăng hiệu suất bằng cách tắt tính toán gradient
                results = self.model(frame_resized)
            
            detect = []
            detections_with_bbox = []  # Lưu trữ các bbox để tính IOU sau này
            
            for detect_object in results.pred[0]:
                label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)
                
                if self.tracking_class is None:
                    if confidence < self.conf_threshold:
                        continue
                else:
                    if class_id != self.tracking_class or confidence < self.conf_threshold:
                        continue
                        
                detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
                detections_with_bbox.append([x1, y1, x2, y2, confidence, class_id])
            
            # Cập nhật tracking với DeepSORT
            self.tracks = self.tracker.update_tracks(detect, frame=frame_resized)
            
            # Vẽ các track lên frame
            for track in self.tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    ltrb = track.to_tlbr()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    color = colors[class_id]
                    B, G, R = map(int, color)
                    
                    # Tính IOU với các detection
                    max_iou = 0
                    for det_bbox in detections_with_bbox:
                        iou = calculate_iou([x1, y1, x2, y2], det_bbox[:4])
                        max_iou = max(max_iou, iou)
                    
                    # Kết hợp tên lớp, ID và IOU vào cùng một label
                    combined_label = "{}:{} {:.2f}".format(class_name[class_id], track_id, max_iou)
                    
                    # Vẽ hộp giới hạn
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (B, G, R), 2)
                    
                    # Vẽ nền cho label
                    text_size = cv2.getTextSize(combined_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame_resized, (x1 - 1, y1 - 20), (x1 + text_size[0], y1), (B, G, R), -1)
                    
                    # Vẽ label
                    cv2.putText(frame_resized, combined_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Tính và hiển thị FPS
            current_time = time.time()
            time_diff = current_time - self.last_time
            if time_diff > 0:  # Tránh chia cho 0
                fps = 1.0 / time_diff
            else:
                fps = 0
            self.last_time = current_time
            
            cv2.putText(frame_resized, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Lưu frame đã xử lý
            self.processed_frame = frame_resized
    
    def get_processed_frame(self):
        return self.processed_frame
    
    def release(self):
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join()
        self.cap.release()
        print("Đã giải phóng tài nguyên")

# Khởi tạo và chạy video processor
processor = VideoProcessor(
    video_path=video_path,
    model=model,
    tracker=tracker,
    conf_threshold=conf_threshold,
    tracking_class=tracking_class,
    frame_skip=3  # Tăng giá trị này để tăng tốc độ xử lý
)

# Hiển thị video đã xử lý
last_frame_time = time.time()  # Khởi tạo biến last_frame_time trước khi sử dụng
while processor.running:
    frame = processor.get_processed_frame()
    if frame is not None:
        # Hiển thị frame (FPS đã được thêm trong lớp VideoProcessor)
        cv2.imshow("Object Tracking", frame)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Thêm một khoảng thời gian ngắn để giảm tải CPU
    time.sleep(0.01)

# Giải phóng tài nguyên
processor.release()
cv2.destroyAllWindows()

print("Chương trình đã kết thúc")