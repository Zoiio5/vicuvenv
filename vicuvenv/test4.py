from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction, QFileDialog, QMessageBox,
                             QLabel, QVBoxLayout, QWidget, QToolBar, QMenu, QSlider, QHBoxLayout, QPushButton,QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPen, QPainter, QColor
from PyQt5.QtCore import Qt, QTimer, QPointF
import sys
import os
import cv2
import torch
import pandas as pd
import openpyxl
from ultralytics import YOLO



class VideoProcessor:
    def __init__(self):
        self.model = None
        self.load_model()
        self.excel_path = None
        self.workbook = None
        self.worksheet = None


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
            return None
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

        return df

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
class ImageLabel(QLabel):
    def __init__(self, image=None):
        super().__init__()
        self.image = image
        if self.image:
            self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.lines = []
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

        if self.drawing and self.start_point and self.end_point:
            painter.drawLine(self.start_point, self.end_point)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and self.start_point:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            relative_start = QPointF(self.start_point.x() / self.width(), self.start_point.y() / self.height())
            relative_end = QPointF(self.end_point.x() / self.width(), self.end_point.y() / self.height())
            line_name, ok = QInputDialog.getText(self, 'Nombre de la línea', 'Ingrese el nombre de la línea:')
            if ok and line_name:
                self.lines.append({'start': relative_start, 'end': relative_end, 'name': line_name})
            self.start_point = None
            self.end_point = None
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self.lines:
                self.lines.pop()
                self.update()
                print(self.lines)
    def toggle_drawing(self):
        self.drawing = not self.drawing
        self.setCursor(QCursor(Qt.CrossCursor if self.drawing else Qt.ArrowCursor))
        self.update()
        if self.drawing:
            self.setFocus()  # Ensure the widget has focus when drawing

class RoiMaker(QLabel):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)
        self.drawing = False
        self.points = []
        self.dragging_point = None

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(QColor('blue'), 3)  # Change line color to blue
        painter.setPen(pen)

        if self.points:
            for i in range(len(self.points) - 1):
                start_point = QPointF(self.points[i].x() * self.width(), self.points[i].y() * self.height())
                end_point = QPointF(self.points[i + 1].x() * self.width(), self.points[i + 1].y() * self.height())
                painter.drawLine(start_point, end_point)
            # Draw line from last point to first point to close the polygon
            start_point = QPointF(self.points[-1].x() * self.width(), self.points[-1].y() * self.height())
            end_point = QPointF(self.points[0].x() * self.width(), self.points[0].y() * self.height())
            painter.drawLine(start_point, end_point)

            # Draw vertices with orange color
            vertex_pen = QPen(QColor('orange'), 6)
            painter.setPen(vertex_pen)
            for point in self.points:
                painter.drawPoint(QPointF(point.x() * self.width(), point.y() * self.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            for i, point in enumerate(self.points):
                if (event.pos() - QPointF(point.x() * self.width(), point.y() * self.height())).manhattanLength() < 30:
                    self.dragging_point = i
                    break
            else:
                if self.drawing:
                    relative_point = QPointF(event.pos().x() / self.width(), event.pos().y() / self.height())
                    self.points.append(relative_point)
                    self.update()

    def mouseMoveEvent(self, event):
        if self.dragging_point is not None:
            self.points[self.dragging_point] = QPointF(event.pos().x() / self.width(), event.pos().y() / self.height())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_point = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self.points:
                self.points.pop()
                self.update()

    def toggle_drawing(self):
        self.drawing = not self.drawing
        self.update()




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
        self.dibujar_linea = False
        self.dibujar_roi = False

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(500, 400)

        toolbar = QToolBar(self)
        self.addToolBar(toolbar)

        load_video_action = QAction('Cargar Videos', self)
        load_video_action.triggered.connect(self.load_videos)
        toolbar.addAction(load_video_action)

        save_detections_action = QAction('Guardar Detecciones', self)
        save_detections_action.triggered.connect(self.save_detections)
        toolbar.addAction(save_detections_action)

        draw_action = QAction('Dibujar en Frame Actual', self)
        draw_action.triggered.connect(self.toggle_drawing)
        toolbar.addAction(draw_action)

        draw_roi = QAction('Dibujar Roi', self)
        draw_roi.triggered.connect(self.toggle_drawing_roi)
        toolbar.addAction(draw_roi)

        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(main_layout)

        self.image_label = ImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")
        main_layout.addWidget(self.image_label)

        # Provide a default image for RoiMaker
        default_image = QImage(800, 600, QImage.Format_RGB32)
        default_image.fill(Qt.white)
        self.roi_maker = RoiMaker(default_image)

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

    def toggle_drawing(self):
        self.dibujar_linea = not self.dibujar_linea
        self.image_label.toggle_drawing()

        if self.dibujar_linea:
            self.play_button.setEnabled(False)
            self.image_label.setFocus()  # Ensure ImageLabel gets focus

        else:
            self.play_button.setEnabled(True)

    def toggle_drawing_roi(self):
        self.dibujar_roi = not self.dibujar_roi
        self.roi_maker.toggle_drawing()

        if self.dibujar_roi:
            self.play_button.setEnabled(False)
            self.roi_maker.setFocus()  # Ensure ImageLabel gets focus
        else:
            self.play_button.setEnabled(True)


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
                self.total_frames = self.processor.total_frames
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
            detections = self.processor.detect_vehicles(frame)
            self.detections.append(detections)
            self.processor.update_excel(self.frame_number, detections)
            self.original_frame = frame.copy()
            self.display_frame(frame, detections)
            self.update_time_label()
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

    def toggle_playback(self):
        if self.drawing_mode:
            QMessageBox.warning(self, "Modo de Dibujo Activo", "No se puede reproducir el video mientras está en modo de dibujo.")
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('>')
        else:
            self.timer.start(int(1000 / (30 * self.playback_speed)))
            self.play_button.setText('II')

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
