import sys
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color  # Importar Color
import hashlib  # Para generar un hash único

class VideoProcessor:
    def __init__(self, input_path: str, output_path: str):
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            print("Error al abrir el archivo de video")
            sys.exit()

        # Obtener propiedades del video
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Inicializar el objeto para escribir el video procesado
        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        # Inicializar modelo YOLO
        try:
            self.model = YOLO("yolov10m.pt")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            sys.exit()

        # Inicializar herramientas de supervisión
        self.tracker = sv.ByteTrack()  # Inicializar el rastreador ByteTrack

        # Inicializar un color común para cajas, etiquetas y trazas (por ejemplo, azul)
        common_color = Color(r=255, g=0, b=0)  # Usar la clase Color para definir el color

        # Inicializar anotadores con el mismo color
        self.box_annotator = sv.BoxAnnotator(color=common_color)
        self.label_annotator = sv.LabelAnnotator(color=common_color)
        self.trace_annotator = sv.TraceAnnotator(color=common_color)  # Configurar el color del trazo

        # Diccionario para traducir las clases al español
        self.class_translation = {
            "person": "persona",
            "car": "Vehiculo_liviano",
            "bicycle": "bicicleta",
            "motorbike": "motocicleta",
            "truck": "camion_+2/2_ejes",
            "bus": "bus",
            # Agregar más clases y sus traducciones según sea necesario
        }

    def process_video(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Procesar el frame
            processed_frame = self.process_frame(frame)

            # Escribir el frame procesado al archivo de salida
            self.out.write(processed_frame)

        # Liberar recursos
        self.cap.release()
        self.out.release()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # Crear etiquetas con IDs únicos de rastreador, hash y nombres de clases traducidos
        labels = [
            f"#{self.generate_hash(tracker_id)} {self.translate_class(results.names[class_id])}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        # Anotar el frame con cajas delimitadoras, etiquetas y trazas
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return self.trace_annotator.annotate(annotated_frame, detections=detections)

    def translate_class(self, class_name: str) -> str:
        # Traducir la clase al español si existe en el diccionario, si no devolver el nombre original
        return self.class_translation.get(class_name, class_name)

    def generate_hash(self, tracker_id: int) -> str:
        # Generar un hash único basado en el tracker_id
        hash_object = hashlib.sha256(str(tracker_id).encode())
        return hash_object.hexdigest()[:8]  # Usar solo los primeros 8 caracteres del hash

if __name__ == "__main__":
    input_video_path = "C:\\Users\\camil\\Downloads\\hiv00059.mp4"
    output_video_path = "resultado_procesado3.mp4"

    processor = VideoProcessor(input_video_path, output_video_path)
    processor.process_video()

    print("Procesamiento completado. El video completo ha sido guardado como 'resultado_procesado.mp4'.")

