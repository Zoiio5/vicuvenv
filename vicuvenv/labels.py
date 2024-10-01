import os
import sys

import cv2
import numpy as np
import openpyxl
import pandas as pd
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPen, QPainter, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QMessageBox,
                             QLabel, QVBoxLayout, QWidget, QToolBar, QSlider, QHBoxLayout, QPushButton)
from ultralytics import YOLO


class VideoProcessor:
    def __init__(self):
        self.model = None
        self.load_model()
        self.excel_path = None
        self.workbook = None
        self.worksheet = None
        self.roi_points = []
        self.labels_folder = os.path.join(os.path.expanduser("~"), "Documentos", "YOLO_LABELS")
        os.makedirs(self.labels_folder, exist_ok=True)

    def load_model(self):
        try:
            self.model = YOLO('yolov10x.pt')
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None

    def process_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print(f"No se pudo abrir el video: {file_path}")
            return False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def read_frame(self, frame_number=None):
        if frame_number is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()

    def detect_vehicles(self, frame):
        if self.model is None:
            raise ValueError("El modelo no está cargado.")
        results = self.model(frame)

        # Convert results to pandas DataFrame
        detections = results[0].boxes.data.cpu().numpy()
        df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

        # Map class IDs to class names if needed (YOLOv8 might use class IDs)
        df['name'] = df['class'].apply(lambda x: self.model.names[int(x)])

        # Filter detections based on ROI if ROI points are defined
        if self.roi_points:
            roi_polygon = np.array(
                [(point.x() * frame.shape[1], point.y() * frame.shape[0]) for point in self.roi_points])
            df = df[df.apply(lambda row: self.is_within_roi(row, roi_polygon), axis=1)]

        return df

    def is_within_roi(self, row, roi_polygon):
        bbox_center = ((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2)
        return cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0

    def initialize_excel(self, file_path):
        self.excel_path = file_path
        self.workbook = openpyxl.Workbook()
        self.worksheet = self.workbook.active
        self.worksheet.append(["Frame", "xmin", "ymin", "xmax", "ymax", "confidence", "name"])
        self.workbook.save(self.excel_path)

    def update_excel(self, frame_number, detections):
        if self.worksheet is None:
            return
        for _, row in detections.iterrows():
            self.worksheet.append(
                [frame_number, row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'], row['name']])
        self.workbook.save(self.excel_path)

    def save_yolo_labels(self, frame_number, detections, frame):
        if detections.empty:
            return  # No guardar etiquetas ni fotos si no hay detecciones

        label_file_path = os.path.join(self.labels_folder, f"frame_{frame_number}.txt")
        image_file_path = os.path.join(self.labels_folder, f"frame_{frame_number}.jpg")

        with open(label_file_path, 'w') as f:
            for _, row in detections.iterrows():
                class_id = int(row['class'])
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                x_center = (xmin + xmax) / 2 / frame.shape[1]
                y_center = (ymin + ymax) / 2 / frame.shape[0]
                width = (xmax - xmin) / frame.shape[1]
                height = (ymax - ymin) / frame.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        cv2.imwrite(image_file_path, frame)

    def process_all_frames(self):
        for frame_number in range(self.total_frames):
            frame = self.read_frame(frame_number)
            if frame is not None:
                detections = self.detect_vehicles(frame)
                self.update_excel(frame_number, detections)
                self.save_yolo_labels(frame_number, detections, frame)
                print(f"Procesado cuadro {frame_number}/{self.total_frames}")
            else:
                print(f"No se pudo leer el cuadro {frame_number}")
        self.release()


class DrawingLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.last_point = QPointF()
        self.current_point = QPointF()
        self.pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            self.current_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing:
            painter = QPainter(self)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, self.current_point)
            self.last_point = self.current_point


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Detección de Vehículos'
        self.processor = VideoProcessor()
        self.initUI()
        self.video_files = []
        self.current_video_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.playback_speed = 1.0
        self.detections = []


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(500, 400)

        toolbar = QToolBar(self)
        self.addToolBar(toolbar)

        self.load_video_action = QAction('Cargar Videos', self)
        self.load_video_action.triggered.connect(self.load_videos)
        toolbar.addAction(self.load_video_action)

        self.save_detections_action = QAction('Guardar Detecciones', self)
        self.save_detections_action.triggered.connect(self.save_detections)
        toolbar.addAction(self.save_detections_action)

        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        self.drawing_label = DrawingLabel()
        self.drawing_label.setAlignment(Qt.AlignCenter)
        self.drawing_label.setStyleSheet("background-color: white;")
        main_layout.addWidget(self.drawing_label)

        self.play_button = QPushButton('>', self)
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)

        self.video_slider = QSlider(Qt.Horizontal, self)
        self.video_slider.setRange(0, 0)
        self.video_slider.sliderMoved.connect(self.slider_moved)
        control_layout.addWidget(self.video_slider)

        self.time_label = QLabel("00:00", self)
        control_layout.addWidget(self.time_label)

        main_layout.addLayout(control_layout)
        self.show()

    def process_next_video(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            if self.processor.process_video(video_file):
                self.total_frames = self.processor.total_frames
                excel_file_name = f"detetections_{self.current_video_index + 1}.xlsx"
                self.excel_path = os.path.join(self.excel_folder, excel_file_name)
                self.processor.initialize_excel(self.excel_path)
                self.frame_number = 0
                self.timer.start(int(1000 / 30))  # Assuming 30 FPS
            else:
                QMessageBox.warning(self, "Advertencia", f"No se pudo cargar el video: {video_file}")
                self.current_video_index += 1
                self.process_next_video()
        else:
            QMessageBox.information(self, "Información", "Procesamiento de todos los videos completado.")

    def next_frame(self):
        if self.frame_number < self.total_frames:
            frame = self.processor.read_frame(self.frame_number)
            if frame is not None:
                detections = self.processor.detect_vehicles(frame)
                self.processor.update_excel(self.frame_number, detections)
                self.processor.save_yolo_labels(self.frame_number, detections, frame)
                self.frame_number += 1
            else:
                print(f"No se pudo leer el cuadro {self.frame_number}")
        else:
            self.current_video_index += 1
            self.process_next_video()

    def process_next_video(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            if self.processor.process_video(video_file):
                self.total_frames = self.processor.total_frames
                excel_file_name = f"detetections_{self.current_video_index + 1}.xlsx"
                self.excel_path = os.path.join(self.excel_folder, excel_file_name)
                self.processor.initialize_excel(self.excel_path)
                self.frame_number = 0
                self.timer.start(0)  # Start processing immediately
            else:
                QMessageBox.warning(self, "Advertencia", f"No se pudo cargar el video: {video_file}")
                self.current_video_index += 1
                self.process_next_video()
        else:
            QMessageBox.information(self, "Información", "Procesamiento de todos los videos completado.")

    def load_videos(self):
        if self.processor.model is None:
            QMessageBox.warning(self, "Advertencia", "El modelo no está cargado correctamente.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(self, "Selecciona uno o más videos", "",
                                                     "Video files (*.mp4; *.avi)")
        if file_paths:
            self.video_files = file_paths
            self.current_video_index = 0
            self.create_excel_folder()
            self.process_next_video()

    def create_excel_folder(self):
        self.excel_folder = os.path.join(os.path.expanduser("~"), "Documentos", "Resultado excels")
        os.makedirs(self.excel_folder, exist_ok=True)

    def toggle_playback(self):
        if self.drawing_label.drawing:
            QMessageBox.warning(self, "Modo de Dibujo Activo",
                                "No se puede reproducir el video mientras está en modo de dibujo.")
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('>')
        else:
            self.timer.start(int(1000 / (30 * self.playback_speed)))
            self.play_button.setText('II')
    def load_videos(self):
        if self.processor.model is None:
            QMessageBox.warning(self, "Advertencia", "El modelo no está cargado correctamente.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(self, "Selecciona uno o más videos", "",
                                                     "Video files (*.mp4; *.avi)")
        if file_paths:
            self.video_files = file_paths
            self.current_video_index = 0
            self.create_excel_folder()
            self.process_next_video()

    def create_excel_folder(self):
        self.excel_folder = os.path.join(os.path.expanduser("~"), "Documentos", "Resultado excels")
        os.makedirs(self.excel_folder, exist_ok=True)







    def slider_moved(self, position):
        self.frame_number = position
        self.processor.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        frame = self.processor.read_frame()
        if frame is not None:
            detections = self.processor.detect_vehicles(frame)
            self.original_frame = frame.copy()
            self.display_frame(frame, detections)
            self.update_time_label()

    def update_time_label(self):
        current_time = int(self.frame_number / 30)  # Assuming 30 FPS
        total_time = int(self.total_frames / 30)  # Assuming 30 FPS
        current_time_str = f"{current_time // 60:02d}:{current_time % 60:02d}"
        total_time_str = f"{total_time // 60:02d}:{total_time % 60:02d}"
        self.time_label.setText(f"{current_time_str} / {total_time_str}")



    def save_detections(self):
        if not self.detections:
            QMessageBox.warning(self, "Advertencia", "No hay detecciones para guardar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Detecciones", "", "Excel Files (*.xlsx)")
        if file_path:
            all_detections = pd.concat(self.detections, ignore_index=True)
            all_detections.to_excel(file_path, index=False)
            QMessageBox.information(self, "Información", "Detecciones guardadas correctamente.")
        else:
            QMessageBox.warning(self, "Advertencia", "No se especificó un archivo para guardar.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
