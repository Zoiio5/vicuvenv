import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPointF


class ImageLabel(QLabel):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.lines = []

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
            self.drawing = False
            self.end_point = event.pos()
            relative_start = QPointF(self.start_point.x() / self.width(), self.start_point.y() / self.height())
            relative_end = QPointF(self.end_point.x() / self.width(), self.end_point.y() / self.height())
            line_name, ok = QInputDialog.getText(self, 'Nombre de la línea', 'Ingrese el nombre de la línea:')
            if ok and line_name:
                self.lines.append({'start': relative_start, 'end': relative_end, 'name': line_name})
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() == Qt.ControlModifier:
            if self.lines:
                self.lines.pop()
                self.update()

    def toggle_drawing(self):
        self.drawing = not self.drawing
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Pintar sobre el primer frame de un video")
        self.image_label = ImageLabel(image)

        self.button = QPushButton("Dibujar línea")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.image_label.toggle_drawing)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def keyPressEvent(self, event):
        self.image_label.keyPressEvent(event)


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return q_image
    else:
        return None


app = QApplication(sys.argv)
video_path = "C:\\Users\\camil\\vcenv\\3.mp4"
first_frame = get_first_frame(video_path)
if first_frame:
    window = MainWindow(first_frame)
    window.show()
else:
    print("No se pudo cargar el primer frame del video.")
sys.exit(app.exec_())
