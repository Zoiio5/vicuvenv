import sys

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


class VideoProcessor:
    def __init__(self, input_path: str, output_path: str, process_minutes: int = 5):
        # Inicializar captura de video
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            print("Error al abrir el archivo de video")
            sys.exit()

        # Obtener propiedades del video
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Calcular el número de cuadros a procesar
        self.max_frames = int(self.fps * 60 * process_minutes)

        # Inicializar el objeto para escribir el video procesado
        self.out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        # Inicializar modelo YOLO
        try:
            self.model = YOLO("yolov8x.pt")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            sys.exit()

        # Inicializar herramientas de supervisión
        self.tracker = sv.ByteTrack()  # Inicializar el rastreador ByteTrack
        self.box_annotator = sv.BoxAnnotator()  # Inicializar el anotador de cajas
        self.label_annotator = sv.LabelAnnotator()  # Inicializar el anotador de etiquetas
        self.trace_annotator = sv.TraceAnnotator()  # Inicializar el anotador de trazas

    def process_video(self):
        frame_count = 0
        while frame_count < self.max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Procesar el frame
            processed_frame = self.process_frame(frame)

            # Escribir el frame procesado al archivo de salida
            self.out.write(processed_frame)

            frame_count += 1

        # Liberar recursos
        self.cap.release()
        self.out.release()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # Crear etiquetas con IDs únicos de rastreador y nombres de clases
        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        # Anotar el frame con cajas delimitadoras, etiquetas y trazas
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        return self.trace_annotator.annotate(annotated_frame, detections=detections)

if __name__ == "__main__":
    input_video_path = "C:\\Users\\camil\\Downloads\\hiv00001.mp4"
    output_video_path = "resultado_procesado.mp4"

    processor = VideoProcessor(input_video_path, output_video_path, process_minutes=5)
    processor.process_video()

    print("Procesamiento completado. El video de 5 minutos ha sido guardado como 'resultado_procesado.mp4'.")
