import os
import sys

import cv2
import numpy as np
import openpyxl
import pandas as pd
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPen, QPainter, QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QMessageBox,
                             QLabel, QVBoxLayout, QWidget, QToolBar, QSlider, QHBoxLayout, QPushButton, QInputDialog)
from ultralytics import YOLO
import supervision as sv  # Importar la biblioteca Supervision


class VideoProcessor:
    def __init__(self):
        self.model = None
        self.load_model()
        self.excel_path = None
        self.workbook = None
        self.worksheet = None
        self.roi_points = []
        self.fps = 30  # Valor por defecto

        # Inicializar herramientas de supervisión
        self.tracker = sv.ByteTrack()  # Inicializar el rastreador ByteTrack
        self.box_annotator = sv.BoxAnnotator()  # Inicializar el anotador de cajas
        self.label_annotator = sv.LabelAnnotator()  # Inicializar el anotador de etiquetas
        self.trace_annotator = sv.TraceAnnotator()  # Inicializar el anotador de trazas

    def load_model(self):
        try:
            self.model = YOLO('yolov8n.pt')
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None

    def process_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            return None
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Obtener la FPS del video
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
        return results

    def is_within_roi(self, row, roi_polygon):
        bbox_center = ((row['xmin'] + row['xmax']) / 2, (row['ymin'] + row['ymax']) / 2)
        return cv2.pointPolygonTest(roi_polygon, bbox_center, False) >= 0

    def initialize_excel(self, file_path):
        self.excel_path = file_path
        self.workbook = openpyxl.Workbook()
        self.worksheet = self.workbook.active
        self.worksheet.append(["Frame", "Confidence", "Name", "Time"])
        self.workbook.save(self.excel_path)

    def initialize_excel(self, file_path):
        self.excel_path = file_path
        self.workbook = openpyxl.Workbook()
        self.worksheet = self.workbook.active
        self.worksheet.append(["Name", "Confidence", "Time"])
        self.workbook.save(self.excel_path)

    def update_excel(self, frame_number, detections):
        if self.worksheet is None:
            return
        time_in_seconds = frame_number / self.fps  # Calculate the time in seconds
        hours, remainder = divmod(time_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

        for _, row in detections.iterrows():
            # Only append the 'name', 'confidence', and 'time' to the worksheet
            self.worksheet.append([row['name'], row['confidence'], time_str])
        self.workbook.save(self.excel_path)

    def annotate_frame(self, frame, detections):
        # Convert detections to Supervision format
        sv_detections = sv.Detections.from_pandas(detections)
        sv_detections = self.tracker.update_with_detections(sv_detections)

        # Create labels with unique tracker IDs and class names
        labels = [
            f"#{tracker_id} {self.model.names[class_id]}"
            for class_id, tracker_id in zip(sv_detections.class_id, sv_detections.tracker_id)
        ]

        # Annotate the frame with bounding boxes, labels, and traces
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=sv_detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=sv_detections, labels=labels)
        return self.trace_annotator.annotate(annotated_frame, detections=sv_detections)


class DrawingLabel(QLabel):
    def __init__(self, image=None, processor=None):
        super().__init__()
        self.image = image
        self.processor = processor
        if self.image:
            self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)
        self.drawing_line = False
        self.drawing_roi = False
        self.start_point = None
        self.end_point = None
        self.lines = []
        self.points = []
        self.dragging_point = None
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)

        for line in self.lines:
            start_point = QPointF(line['start'].x() * self.width(), line['start'].y() * self.height())
            end_point = QPointF(line['end'].x() * self.width(), line['end'].y() * self.height())
            painter.drawLine(start_point, end_point)

        if self.drawing_line and self.start_point and self.end_point:
            painter.drawLine(self.start_point, self.end_point)

        pen.setColor(QColor('blue'))
        painter.setPen(pen)
        if self.points:
            for i in range(len(self.points) - 1):
                start_point = QPointF(self.points[i].x() * self.width(), self.points[i].y() * self.height())
                end_point = QPointF(self.points[i + 1].x() * self.width(), self.points[i + 1].y() * self.height())
                painter.drawLine(start_point, end_point)
            start_point = QPointF(self.points[-1].x() * self.width(), self.points[-1].y() * self.height())
            end_point = QPointF(self.points[0].x() * self.width(), self.points[0].y() * self.height())
            painter.drawLine(start_point, end_point)

            vertex_pen = QPen(QColor('orange'), 6)
            painter.setPen(vertex_pen)
            for point in self.points:
                painter.drawPoint(QPointF(point.x() * self.width(), point.y() * self.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing_line:
                self.start_point = event.pos()
                self.end_point = self.start_point
                self.update()
            elif self.drawing_roi:
                for i, point in enumerate(self.points):
                    if (event.pos() - QPointF(point.x() * self.width(), point.y() * self.height())).manhattanLength() < 30:
                        self.dragging_point = i
                        break
                else:
                    relative_point = QPointF(event.pos().x() / self.width(), event.pos().y() / self.height())
                    self.points.append(relative_point)
                    self.update()
                    if self.processor:
                        self.processor.roi_points = self.points

    def mouseMoveEvent(self, event):
        if self.drawing_line and self.start_point:
            self.end_point = event.pos()
            self.update()
        elif self.drawing_roi and self.dragging_point is not None:
            self.points[self.dragging_point] = QPointF(event.pos().x() / self.width(), event.pos().y() / self.height())
            self.update()
            if self.processor:
                self.processor.roi_points = self.points

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing_line:
                self.end_point = event.pos()
                relative_start = QPointF(self.start_point.x() / self.width(), self.start_point.y() / self.height())
                relative_end = QPointF(self.end_point.x() / self.width(), self.end_point.y() / self.height())
                line_name, ok = QInputDialog.getText(self, 'Nombre de la línea', 'Ingrese el nombre de la línea:')
                if ok and line_name:
                    self.lines.append({'start': relative_start, 'end': relative_end, 'name': line_name})
                self.start_point = None
                self.end_point = None
                self.update()
            elif self.drawing_roi:
                self.dragging_point = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self.drawing_line and self.lines:
                self.lines.pop()
            elif self.drawing_roi and self.points:
                self.points.pop()
                if self.processor:
                    self.processor.roi_points = self.points
            self.update()

    def toggle_drawing_line(self):
        self.drawing_line = not self.drawing_line
        self.setCursor(QCursor(Qt.CrossCursor if self.drawing_line else Qt.ArrowCursor))
        self.update()
        if self.drawing_line:
            self.setFocus()

    def toggle_drawing_roi(self):
        self.drawing_roi = not self.drawing_roi
        self.setCursor(QCursor(Qt.CrossCursor if self.drawing_roi else Qt.ArrowCursor))
        self.update()
        if self.drawing_roi:
            self.setFocus()

class App(QMainWindow):
    def __init__(self):
            super().__init__()
            self.title = 'Detección de Vehículos'
            self.processor = VideoProcessor()
            self.frame = None
            self.original_frame = None
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.next_frame)
            self.detections = []
            self.frame_number = 0
            self.initUI()
            self.video_files = []
            self.current_video_index = 0
            self.playback_speed = 1.0

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

        self.draw_line_action = QAction('Dibujar Línea', self)
        self.draw_line_action.setCheckable(True)
        self.draw_line_action.triggered.connect(self.toggle_drawing_line)
        toolbar.addAction(self.draw_line_action)

        self.draw_roi_action = QAction('Dibujar ROI', self)
        self.draw_roi_action.setCheckable(True)
        self.draw_roi_action.triggered.connect(self.toggle_drawing_roi)
        toolbar.addAction(self.draw_roi_action)

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

    def toggle_drawing_line(self):
        if self.drawing_label.drawing_roi:
            self.drawing_label.toggle_drawing_roi()
            self.draw_roi_action.setChecked(False)
        self.drawing_label.toggle_drawing_line()
        self.draw_line_action.setChecked(self.drawing_label.drawing_line)
        self.play_button.setEnabled(not self.drawing_label.drawing_line)

    def toggle_drawing_roi(self):
        if self.drawing_label.drawing_line:
            self.drawing_label.toggle_drawing_line()
            self.draw_line_action.setChecked(False)
        self.drawing_label.toggle_drawing_roi()
        self.draw_roi_action.setChecked(self.drawing_label.drawing_roi)
        self.play_button.setEnabled(not self.drawing_label.drawing_roi)

    def resizeEvent(self, event):
        if self.original_frame is not None:
            self.display_frame(self.original_frame.copy(), [])

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

    def process_next_video(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            if self.processor.process_video(video_file):
                self.total_frames = self.processor.total_frames
                self.fps = self.processor.fps  # Obtener la FPS detectada
                self.video_slider.setRange(0, self.total_frames)
                excel_file_name = f"detetections_{self.current_video_index + 1}.xlsx"
                self.excel_path = os.path.join(self.excel_folder, excel_file_name)
                self.processor.initialize_excel(self.excel_path)
                self.frame_number = 0
                self.detections = []
                self.next_frame()
            else:
                QMessageBox.warning(self, "Advertencia", f"No se pudo cargar el video: {video_file}")
                self.current_video_index += 1
                self.process_next_video()
        else:
            QMessageBox.information(self, "Información", "Procesamiento de todos los videos completado.")

    def next_frame(self):
        frame = self.processor.read_frame()
        if frame is not None:
            self.frame_number += 1
            self.video_slider.setValue(self.frame_number)
            results = self.processor.detect_vehicles(frame)
            detections = results[0].boxes.data.cpu().numpy()
            df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])
            df['name'] = df['class'].apply(lambda x: self.processor.model.names[int(x)])
            self.detections.append(df)
            self.processor.update_excel(self.frame_number, df)
            self.original_frame = frame.copy()
            self.display_frame(frame, results)
            self.update_time_label()
        else:
            self.timer.stop()
            self.processor.release()
            self.current_video_index += 1
            self.process_next_video()

    def display_frame(self, frame, results):
        if results:
            # Convert results to Supervision format
            sv_detections = sv.Detections.from_ultralytics(results[0])
            sv_detections = self.processor.tracker.update_with_detections(sv_detections)

            # Create labels with unique tracker IDs and class names
            labels = [
                f"#{tracker_id} {self.processor.model.names[class_id]}"
                for class_id, tracker_id in zip(sv_detections.class_id, sv_detections.tracker_id)
            ]

            # Annotate the frame with bounding boxes, labels, and traces
            annotated_frame = self.processor.box_annotator.annotate(frame.copy(), detections=sv_detections)
            annotated_frame = self.processor.label_annotator.annotate(annotated_frame, detections=sv_detections,
                                                                      labels=labels)
            annotated_frame = self.processor.trace_annotator.annotate(annotated_frame, detections=sv_detections)

            frame = annotated_frame

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)

        screen_size = self.size()  # Obtener el tamaño actual de la ventana
        self.drawing_label.setPixmap(pixmap.scaled(screen_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
        current_time = int(self.frame_number / self.fps)  # Use the actual FPS
        total_time = int(self.total_frames / self.fps)  # Use the actual FPS
        current_time_str = f"{current_time // 60:02d}:{current_time % 60:02d}"
        total_time_str = f"{total_time // 60:02d}:{total_time % 60:02d}"
        self.time_label.setText(f"{current_time_str} / {total_time_str}")

    def toggle_playback(self):
        if self.drawing_label.drawing_line or self.drawing_label.drawing_roi:
            QMessageBox.warning(self, "Modo de Dibujo Activo",
                                "No se puede reproducir el video mientras está en modo de dibujo.")
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('>')
        else:
            self.timer.start(int(1000 / (self.processor.fps * self.playback_speed)))  # Adjust timer using FPS
            self.play_button.setText('II')

    def save_detections(self):
        if not self.detections:
            QMessageBox.warning(self, "Advertencia", "No hay detecciones para guardar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Detecciones", "", "Excel Files (*.xlsx)")
        if file_path:
            # Create a new DataFrame to store all detections
            all_detections = pd.DataFrame(columns=['Name', 'Confidence', 'Time'])

            # Iterate over each frame's detections
            for frame_number, detections in enumerate(self.detections, start=1):
                if detections.empty:
                    continue  # Skip empty DataFrames

                # Calculate the time in seconds
                time_in_seconds = frame_number / self.fps
                hours, remainder = divmod(time_in_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

                # For each detection, keep only the name, confidence, and add the time
                for _, row in detections.iterrows():
                    df = pd.DataFrame({
                        'Name': [row['name']],
                        'Confidence': [row['confidence']],
                        'Time': [time_str]
                    })
                    all_detections = pd.concat([all_detections, df], ignore_index=True)

            # Write all detections to the Excel file
            all_detections.to_excel(file_path, index=False)

            QMessageBox.information(self, "Información", "Detecciones guardadas correctamente.")
        else:
            QMessageBox.warning(self, "Advertencia", "No se especificó un archivo para guardar.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
