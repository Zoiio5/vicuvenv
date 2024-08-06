from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QMessageBox,
                             QLabel, QVBoxLayout, QWidget, QToolBar, QMenu)
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
import sys
import os
import cv2
import torch
import pandas as pd
import openpyxl
import numpy as np

class VideoProcessor:
    def __init__(self):
        self.model = None
        self.load_model()
        self.excel_path = None
        self.workbook = None
        self.worksheet = None

    def load_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.model = self.model.cuda() if torch.cuda.is_available() else self.model
            print("Modelo cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None

    def process_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            return None
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True

    def read_frame(self):
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
        return results.pandas().xyxy[0]

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
            self.worksheet.append([frame_number, row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence'], row['name']])
        self.workbook.save(self.excel_path)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Detección de Vehículos con YOLOv5'
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



    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        # Establecer el tamaño mínimo y máximo de la ventana
        self.setMinimumSize(500, 400)
        self.showMaximized()

        # Barra de herramientas
        main_layout = QVBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        toolbar = QToolBar(self)
        self.addToolBar(toolbar)

        load_video_action = QAction('Cargar Videos', self)
        load_video_action.triggered.connect(self.load_videos)
        toolbar.addAction(load_video_action)

        play_video_action = QAction('Reproducir Video', self)
        play_video_action.triggered.connect(self.show_playback_menu)
        toolbar.addAction(play_video_action)

        draw_roi_action = QAction('Dibujar ROI', self)
        draw_roi_action.triggered.connect(self.toggle_drawing)
        toolbar.addAction(draw_roi_action)

        save_detections_action = QAction('Guardar Detecciones', self)
        save_detections_action.triggered.connect(self.save_detections)
        toolbar.addAction(save_detections_action)

        self.image_label = ImageLabel(self)
        self.setCentralWidget(self.image_label)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")
        self.image_label.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Habilitar eventos del mouse para QLabel

        self.show()

    def resizeEvent(self, event):
        if self.original_frame is not None:
            self.display_frame(self.original_frame.copy(), pd.DataFrame())

    def load_videos(self):
        if self.processor.model is None:
            QMessageBox.warning(self, "Advertencia", "El modelo no está cargado correctamente.")
            return

        file_paths, _ = QFileDialog.getOpenFileNames(self, "Selecciona uno o más videos", "", "Video files (*.mp4; *.avi)")
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
            detections = self.processor.detect_vehicles(frame)
            self.detections.append(detections)
            self.processor.update_excel(self.frame_number, detections)
            self.original_frame = frame.copy()
            self.display_frame(frame, detections)
        else:
            self.timer.stop()
            self.processor.release()
            self.current_video_index += 1
            self.process_next_video()

    def display_frame(self, frame, detections):
        if not detections.empty and 'confidence' in detections.columns:
            filtered_detections = detections[detections['confidence'] > 0.45]
            for _, row in filtered_detections.iterrows():
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = row['name']
                confidence = row['confidence']

                # Dibujar caja delimitadora
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Dibujar etiqueta y confianza
                cv2.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (36, 255, 12), 2)

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)

        screen_size = self.size()  # Obtener el tamaño actual de la ventana
        self.image_label.setPixmap(pixmap.scaled(screen_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_playback_menu(self):
        menu = QMenu(self)
        speeds = [("Velocidad 0.25x", 0.25), ("Velocidad 0.5x", 0.5), ("Velocidad 1x", 1), ("Velocidad 2x", 2),
                  ("Velocidad 4x", 4)]
        for label, s in speeds:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, s=s: self.play_video(s))
            menu.addAction(action)

        cursor = QCursor()
        menu.exec_(cursor.pos())

    def play_video(self, speed):
        self.timer.start(int(1000 / (30 * speed)))

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
