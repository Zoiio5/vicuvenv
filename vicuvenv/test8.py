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
        self.label_annotator = sv.LabelAnnotator()  # Initialize the label annotator
        self.trace_annotator = sv.TraceAnnotator()  # Initialize the trace annotator

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.process_frame(frame)

            # Convert the image to RGB format for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))
        else:
            self.cap.release()
            self.timer.stop()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # Create labels with unique tracker IDs and class names
        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        # Annotate the frame with bounding boxes, labels, and traces
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return self.trace_annotator.annotate(annotated_frame, detections=detections)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
