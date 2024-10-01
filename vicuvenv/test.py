import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QApplication, QInputDialog, QPushButton


class Ventana(QWidget):
    def __init__(self):
        super().__init__()

        self.figuras = {}
        self.puntos_temp = []
        self.punto_seleccionado = None
        self.dibujar = True  # Añade un atributo para controlar el dibujo

        # Añade un botón para activar/desactivar el dibujo
        self.boton = QPushButton('Activar/Desactivar dibujo', self)
        self.boton.clicked.connect(self.toggle_dibujo)

    def toggle_dibujo(self):
        self.dibujar = not self.dibujar

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.dibujar:  # Solo dibuja si self.dibujar es True
            for nombre, figura in self.figuras.items():
                for i, punto in enumerate(figura['puntos']):
                    if abs(event.x() - punto.x()) < 20 and abs(event.y() - punto.y()) < 20:  # Aumenta el área de arrastre
                        self.punto_seleccionado = (nombre, i)
                        return
            self.puntos_temp.append(event.pos())
            self.update()
        elif event.button() == Qt.RightButton and self.figuras:
            self.figuras.popitem(last=False)  # Elimina el primer elemento del diccionario
            self.update()

    def mouseMoveEvent(self, event):
        if self.punto_seleccionado is not None:
            self.figuras[self.punto_seleccionado[0]]['puntos'][self.punto_seleccionado[1]] = event.pos()
            self.update()
        elif self.puntos_temp:
            self.puntos_temp[-1] = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and len(self.puntos_temp) == 2:
            self.update()
            nombre, ok = QInputDialog.getText(self, 'Nombre de la figura', 'Introduce el nombre de la figura:')
            if ok:
                self.figuras[nombre] = {'puntos': list(self.puntos_temp), 'dibujar': True}
            self.puntos_temp = []

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(Qt.green, 20)
        brush = QBrush(Qt.green)
        painter.setPen(pen)
        painter.setBrush(brush)

        for figura in self.figuras.values():
            if figura['dibujar'] and len(figura['puntos']) == 2:
                painter.drawLine(figura['puntos'][0], figura['puntos'][1])
                painter.setBrush(Qt.transparent)  # Configura el pincel para dibujar círculos transparentes
                painter.drawEllipse(figura['puntos'][0].x() - 5, figura['puntos'][0].y() - 5, 10, 10)
                painter.drawEllipse(figura['puntos'][1].x() - 5, figura['puntos'][1].y() - 5, 10, 10)
                painter.setBrush(brush)  # Restaura el pincel para dibujar puntos verdes

        for punto in self.puntos_temp:
            painter.drawPoint(punto)

        print("Figuras:")
        for nombre, figura in self.figuras.items():
            print(f"Nombre: {nombre}, Puntos: {figura['puntos']}")

app = QApplication(sys.argv)
ventana = Ventana()
ventana.show()
sys.exit(app.exec_())
