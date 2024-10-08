import sys
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from ultralytics import YOLO
import supervision as sv

class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección en Tiempo Real")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.resize(800, 600)

        layout = QVBoxLayout()
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.cap = cv2.VideoCapture("C:\\Users\\camil\\Downloads\\hiv00001.mp4")
        self.model = YOLO("yolov10s.pt")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.tracker = sv.ByteTrack()  # Initialize the ByteTrack tracker from Supervision
        self.box_annotator = sv.BoxAnnotator()  # Initialize the box annotator
        self.unique_id_counter = 0  # Initialize a counter for unique IDs

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame)[0]  # Perform object detection
            detections = sv.Detections.from_ultralytics(results)  # Convert detections
            tracked_detections = self.tracker.update_with_detections(detections)  # Update tracker with detections

            # Annotate the frame with the detections
            annotated_frame = self.box_annotator.annotate(frame.copy(), detections=tracked_detections)

            # Draw bounding boxes and assign unique IDs to each detection
            for detection in tracked_detections:
                bbox = detection[0]  # Bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)

                # Assign a unique ID
                self.unique_id_counter += 1
                unique_id = self.unique_id_counter

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID: {unique_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Convert the image to RGB format for display
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))
        else:
            self.cap.release()
            self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
