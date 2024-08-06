import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF


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


class MainWindow(QMainWindow):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Pintar sobre el primer frame de un video")
        self.image_label = RoiMaker(image)

        self.button = QPushButton("Dibujar Polígono")
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
