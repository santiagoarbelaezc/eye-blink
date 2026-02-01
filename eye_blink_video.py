import cv2
import pygame
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO
# Importación corregida para ultralytics
try:
    from ultralytics.utils.plotting import Annotator
except ImportError:
    # Para versiones más antiguas
    try:
        from ultralytics.yolo.utils.plotting import Annotator
    except ImportError:
        print("Versión de ultralytics incompatible. Instala la última versión:")
        print("pip install --upgrade ultralytics")
        exit()

class EyeBlinkVideoPlayer:
    def __init__(self, video_path, model_path='yolov8n.pt'):
        """
        Inicializa el sistema de detección de parpadeo y reproductor de video
        
        Args:
            video_path (str): Ruta al video a reproducir
            model_path (str): Ruta al modelo YOLO para detección
        """
        # Inicializar Pygame para reproducción de video
        pygame.init()
        pygame.mixer.init()
        
        # Configuración de rutas
        self.video_path = Path(video_path)
        self.model_path = model_path
        
        # Inicializar cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("No se pudo abrir la cámara, intentando con índice 1...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir ninguna cámara")
        
        # Configurar resolución de cámara
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Cargar modelo YOLO para detección facial
        print("Cargando modelo YOLO...")
        try:
            self.face_model = YOLO(model_path)  # yolov8n.pt ya incluye detección facial
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            print("Descargando modelo yolov8n.pt...")
            self.face_model = YOLO('yolov8n.pt')
        
        # Variables para detección de ojos
        self.eyes_closed = False
        self.blink_start_time = None
        self.blink_threshold = 0.5  # segundos para considerar como parpadeo (aumentado para mejor detección)
        
        # Variables para seguimiento de estado
        self.playing_video = False
        self.video_finished = False
        
        # Configuración de Pygame para video
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Reproductor de Video por Parpadeo")
        
        # Fuente para texto
        self.font = pygame.font.Font(None, 36)
        
        # Historial de estados de ojos
        self.eye_state_history = []
        self.history_size = 5
        
        # Umbral para detección de ojos cerrados
        self.eye_close_threshold = 0.4
        
    def detect_eyes_state(self, frame):
        """
        Detecta el estado de los ojos en el frame usando detección facial
        
        Args:
            frame: Frame de la cámara
            
        Returns:
            tuple: (eyes_closed, annotated_frame)
        """
        # Realizar detección con YOLO
        results = self.face_model(frame, verbose=False)
        
        eyes_closed = False
        annotated_frame = frame.copy()
        
        for result in results:
            annotator = Annotator(annotated_frame)
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.cpu().numpy()
                
                for box in boxes:
                    # Obtener coordenadas de la caja del rostro
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    
                    # Solo procesar detecciones de persona (clase 0 en COCO)
                    if cls == 0 and conf > 0.5:
                        # Dibujar caja del rostro
                        annotator.box_label([x1, y1, x2, y2], f"Face {conf:.2f}", color=(0, 255, 0))
                        
                        # Calcular región de interés para los ojos (parte superior del rostro)
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        # Región de ojos (parte superior 40% del rostro)
                        eye_region_y1 = y1 + int(face_height * 0.25)
                        eye_region_y2 = y1 + int(face_height * 0.45)
                        eye_region_x1 = x1 + int(face_width * 0.1)
                        eye_region_x2 = x2 - int(face_width * 0.1)
                        
                        # Extraer región de ojos
                        if (eye_region_y2 > eye_region_y1 and 
                            eye_region_x2 > eye_region_x1 and
                            eye_region_y1 >= 0 and eye_region_x1 >= 0):
                            
                            eye_region = frame[eye_region_y1:eye_region_y2, 
                                              eye_region_x1:eye_region_x2]
                            
                            if eye_region.size > 0:
                                # Convertir a escala de grises
                                eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                                
                                # Aplicar desenfoque para reducir ruido
                                eye_gray = cv2.GaussianBlur(eye_gray, (7, 7), 0)
                                
                                # Aplicar umbral adaptativo
                                thresh = cv2.adaptiveThreshold(eye_gray, 255, 
                                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                              cv2.THRESH_BINARY_INV, 11, 2)
                                
                                # Calcular porcentaje de píxeles oscuros
                                dark_pixels = np.sum(thresh == 255)
                                total_pixels = eye_region.shape[0] * eye_region.shape[1]
                                
                                if total_pixels > 0:
                                    dark_ratio = dark_pixels / total_pixels
                                    
                                    # Determinar si los ojos están cerrados
                                    eyes_closed = dark_ratio > self.eye_close_threshold
                                    
                                    # Dibujar región de ojos
                                    eye_color = (0, 0, 255) if eyes_closed else (255, 255, 0)
                                    cv2.rectangle(annotated_frame, 
                                                (eye_region_x1, eye_region_y1),
                                                (eye_region_x2, eye_region_y2), 
                                                eye_color, 2)
                                    
                                    # Mostrar estado
                                    status_text = "CLOSED" if eyes_closed else "OPEN"
                                    cv2.putText(annotated_frame, f"Eyes: {status_text}",
                                              (eye_region_x1, eye_region_y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)
                                    
                                    # Mostrar ratio
                                    cv2.putText(annotated_frame, f"Ratio: {dark_ratio:.2f}",
                                              (eye_region_x1, eye_region_y2 + 20),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return eyes_closed, annotated_frame
    
    def play_video(self):
        """
        Reproduce el video usando OpenCV (más simple y confiable)
        """
        try:
            print(f"Reproduciendo video: {self.video_path}")
            self.playing_video = True
            
            # Cargar video con OpenCV
            video_cap = cv2.VideoCapture(str(self.video_path))
            
            if not video_cap.isOpened():
                print(f"No se pudo abrir el video: {self.video_path}")
                self.show_error("Error al abrir video")
                return
            
            # Obtener propiedades del video
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            frame_delay = int(1000 / fps) if fps > 0 else 33
            
            print(f"Video FPS: {fps}, Delay: {frame_delay}ms")
            
            # Crear ventana para el video
            cv2.namedWindow('Video Reproducido', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video Reproducido', 800, 600)
            
            # Reproducir video
            while self.playing_video:
                ret, frame = video_cap.read()
                
                if not ret:
                    print("Fin del video")
                    break
                
                # Mostrar frame
                cv2.imshow('Video Reproducido', frame)
                
                # Controlar velocidad y salida
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == 27 or key == ord('q') or key == ord(' '):  # ESC, q o espacio
                    self.playing_video = False
            
            # Liberar recursos del video
            video_cap.release()
            cv2.destroyWindow('Video Reproducido')
            
            self.video_finished = True
            print("Video finalizado")
            
        except Exception as e:
            print(f"Error reproduciendo video: {e}")
            self.show_error(f"Error: {str(e)}")
    
    def show_error(self, message):
        """
        Muestra un mensaje de error en la pantalla de Pygame
        """
        self.screen.fill((0, 0, 0))
        error_text = self.font.render(message, True, (255, 0, 0))
        text_rect = error_text.get_rect(center=(400, 300))
        self.screen.blit(error_text, text_rect)
        pygame.display.flip()
        pygame.time.wait(3000)
    
    def show_instructions(self):
        """
        Muestra instrucciones en la pantalla de Pygame
        """
        self.screen.fill((0, 0, 0))
        
        title = self.font.render("Sistema de Detección de Parpadeo", True, (255, 255, 255))
        instruction1 = pygame.font.Font(None, 28).render(
            "Cierra los ojos por 0.5 segundos para reproducir el video", 
            True, (0, 255, 0))
        instruction2 = pygame.font.Font(None, 28).render(
            "Presiona ESC para salir | +/- para ajustar sensibilidad", 
            True, (255, 255, 0))
        instruction3 = pygame.font.Font(None, 24).render(
            f"Umbral actual: {self.blink_threshold:.1f}s | Sensibilidad ojos: {self.eye_close_threshold:.2f}",
            True, (255, 200, 0))
        
        title_rect = title.get_rect(center=(400, 100))
        instr1_rect = instruction1.get_rect(center=(400, 200))
        instr2_rect = instruction2.get_rect(center=(400, 250))
        instr3_rect = instruction3.get_rect(center=(400, 300))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(instruction1, instr1_rect)
        self.screen.blit(instruction2, instr2_rect)
        self.screen.blit(instruction3, instr3_rect)
        
        pygame.display.flip()
    
    def run(self):
        """
        Ejecuta el sistema principal
        """
        print("\n" + "="*50)
        print("SISTEMA DE DETECCIÓN DE PARPADEO")
        print("="*50)
        print(f"Umbral de parpadeo: {self.blink_threshold} segundos")
        print(f"Sensibilidad ojos: {self.eye_close_threshold}")
        print("Instrucciones:")
        print("1. Cierra los ojos por 0.5 segundos para reproducir el video")
        print("2. Presiona ESC para salir")
        print("3. +/- para ajustar el umbral de tiempo")
        print("4. a/z para ajustar la sensibilidad de detección de ojos")
        print("="*50 + "\n")
        
        # Mostrar instrucciones iniciales
        self.show_instructions()
        pygame.time.wait(2000)
        
        # Contadores para estadísticas
        blink_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Capturar frame de la cámara
                ret, frame = self.cap.read()
                if not ret:
                    print("Error al capturar frame de la cámara")
                    break
                
                # Voltear horizontalmente para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Detectar estado de ojos
                eyes_closed, annotated_frame = self.detect_eyes_state(frame)
                
                # Actualizar historial
                self.eye_state_history.append(eyes_closed)
                if len(self.eye_state_history) > self.history_size:
                    self.eye_state_history.pop(0)
                
                # Verificar parpadeo prolongado
                if eyes_closed:
                    if self.blink_start_time is None:
                        self.blink_start_time = time.time()
                    else:
                        blink_duration = time.time() - self.blink_start_time
                        
                        # Mostrar temporizador en el frame
                        timer_text = f"Parpadeo: {blink_duration:.2f}s"
                        cv2.putText(annotated_frame, timer_text, 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 255), 2)
                        
                        # Barra de progreso visual
                        bar_width = 200
                        bar_height = 20
                        bar_x = 10
                        bar_y = 90
                        
                        # Dibujar fondo de la barra
                        cv2.rectangle(annotated_frame, 
                                    (bar_x, bar_y), 
                                    (bar_x + bar_width, bar_y + bar_height), 
                                    (100, 100, 100), -1)
                        
                        # Dibujar progreso
                        progress = min(blink_duration / self.blink_threshold, 1.0)
                        progress_width = int(bar_width * progress)
                        progress_color = (0, int(255 * progress), int(255 * (1 - progress)))
                        
                        cv2.rectangle(annotated_frame,
                                    (bar_x, bar_y),
                                    (bar_x + progress_width, bar_y + bar_height),
                                    progress_color, -1)
                        
                        # Borde de la barra
                        cv2.rectangle(annotated_frame,
                                    (bar_x, bar_y),
                                    (bar_x + bar_width, bar_y + bar_height),
                                    (255, 255, 255), 1)
                        
                        # Texto de porcentaje
                        percent_text = f"{progress*100:.0f}%"
                        cv2.putText(annotated_frame, percent_text,
                                  (bar_x + bar_width + 10, bar_y + bar_height - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Si se alcanza el umbral y no hay video en reproducción
                        if (blink_duration >= self.blink_threshold and 
                            not self.playing_video):
                            
                            blink_count += 1
                            print(f"\n¡Parpadeo #{blink_count} detectado! ({blink_duration:.2f}s)")
                            print("Iniciando reproducción de video...")
                            
                            # Mostrar mensaje en pantalla
                            cv2.putText(annotated_frame, "¡VIDEO INICIADO!",
                                      (200, 300), cv2.FONT_HERSHEY_SIMPLEX,
                                      1.5, (0, 255, 0), 3)
                            cv2.imshow('Eye Blink Detection', annotated_frame)
                            cv2.waitKey(500)  # Breve pausa para mostrar el mensaje
                            
                            # Reproducir video
                            self.play_video()
                            
                            # Reiniciar estados
                            self.blink_start_time = None
                            self.eye_state_history = []
                            
                            # Mostrar instrucciones después del video
                            if self.video_finished:
                                self.show_instructions()
                                self.video_finished = False
                                pygame.time.wait(1000)
                else:
                    self.blink_start_time = None
                
                # Mostrar estadísticas en el frame
                elapsed_time = time.time() - start_time
                status_text = "OJOS CERRADOS" if eyes_closed else "ojos abiertos"
                status_color = (0, 0, 255) if eyes_closed else (0, 255, 0)
                
                cv2.putText(annotated_frame, f"Estado: {status_text}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                cv2.putText(annotated_frame, f"Parpadeos: {blink_count}", 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Tiempo: {elapsed_time:.0f}s", 
                          (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(annotated_frame, f"Umbral: {self.blink_threshold:.1f}s", 
                          (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
                cv2.putText(annotated_frame, f"Sensibilidad: {self.eye_close_threshold:.2f}", 
                          (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 255), 1)
                
                # Instrucciones en pantalla
                cv2.putText(annotated_frame, "ESC: Salir  |  +/-: Umbral  |  a/z: Sensibilidad",
                          (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Mostrar frame anotado
                cv2.imshow('Eye Blink Detection', annotated_frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nSaliendo del programa...")
                    break
                elif key == ord('q'):
                    print("\nSaliendo del programa...")
                    break
                elif key == ord('+'):
                    self.blink_threshold += 0.1
                    print(f"Umbral aumentado a: {self.blink_threshold:.1f}s")
                elif key == ord('-') and self.blink_threshold > 0.1:
                    self.blink_threshold -= 0.1
                    print(f"Umbral disminuido a: {self.blink_threshold:.1f}s")
                elif key == ord('a'):  # Aumentar sensibilidad
                    self.eye_close_threshold = min(0.9, self.eye_close_threshold + 0.05)
                    print(f"Sensibilidad aumentada a: {self.eye_close_threshold:.2f}")
                elif key == ord('z') and self.eye_close_threshold > 0.1:  # Disminuir sensibilidad
                    self.eye_close_threshold = max(0.1, self.eye_close_threshold - 0.05)
                    print(f"Sensibilidad disminuida a: {self.eye_close_threshold:.2f}")
                elif key == ord(' '):  # Espacio para forzar reproducción
                    if not self.playing_video:
                        print("Reproducción forzada con ESPACIO")
                        self.play_video()
                
                # Procesar eventos de Pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
        
        except KeyboardInterrupt:
            print("\nInterrumpido por el usuario")
        except Exception as e:
            print(f"\nError durante la ejecución: {e}")
        finally:
            # Liberar recursos
            self.cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            
            # Mostrar estadísticas finales
            total_time = time.time() - start_time
            print("\n" + "="*50)
            print("ESTADÍSTICAS FINALES")
            print("="*50)
            print(f"Tiempo total de ejecución: {total_time:.0f} segundos")
            print(f"Parpadeos detectados: {blink_count}")
            if total_time > 0:
                print(f"Frecuencia: {blink_count/total_time*60:.1f} parpadeos/minuto")
            print("="*50)
            print("Programa finalizado correctamente")

def main():
    """
    Función principal
    """
    # Lista de posibles archivos de video
    video_files = [
        "video.mp4",
        "video.avi",
        "video.mkv",
        "sample_video.mp4",
        "test_video.mp4"
    ]
    
    video_path = None
    
    # Buscar archivo de video
    for video_file in video_files:
        if Path(video_file).exists():
            video_path = video_file
            print(f"Video encontrado: {video_path}")
            break
    
    # Si no se encuentra video, preguntar al usuario
    if video_path is None:
        print("\nNo se encontró ningún archivo de video.")
        print("Archivos buscados:", ", ".join(video_files))
        
        create_sample = input("\n¿Deseas crear un video de ejemplo? (s/n): ").lower()
        
        if create_sample == 's':
            create_sample_video()
            video_path = "sample_video.mp4"
        else:
            # Preguntar por ruta manual
            manual_path = input("Ingresa la ruta completa al archivo de video: ").strip()
            if Path(manual_path).exists():
                video_path = manual_path
            else:
                print(f"El archivo no existe: {manual_path}")
                print("Creando video de ejemplo por defecto...")
                create_sample_video()
                video_path = "sample_video.mp4"
    
    # Verificar si el modelo YOLO existe
    model_files = [
        "yolov8n.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolov8l.pt",
        "yolov8x.pt"
    ]
    
    model_exists = any(Path(model).exists() for model in model_files)
    
    if not model_exists:
        print("\nModelo YOLO no encontrado.")
        print("Se descargará automáticamente yolov8n.pt (≈6MB)...")
    
    # Inicializar y ejecutar el sistema
    try:
        player = EyeBlinkVideoPlayer(video_path)
        player.run()
    except Exception as e:
        print(f"\nError inicializando el sistema: {e}")
        print("Asegúrate de que:")
        print("1. La cámara esté conectada y funcionando")
        print("2. Tienes instaladas todas las dependencias")
        print("3. Tienes conexión a internet para descargar el modelo YOLO")
        print("\nDependencias necesarias:")
        print("pip install opencv-python pygame pillow ultralytics numpy")

def create_sample_video():
    """
    Crea un video de ejemplo más elaborado
    """
    print("\nCreando video de ejemplo...")
    
    # Configuración del video
    width, height = 800, 600
    fps = 30
    duration = 10  # segundos
    total_frames = fps * duration
    
    # Crear objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_video.mp4', fourcc, fps, (width, height))
    
    print(f"Generando {total_frames} frames...")
    
    for i in range(total_frames):
        # Crear fondo negro
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Título animado
        title = "¡VIDEO DE DEMOSTRACIÓN!"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        title_x = (width - title_size[0]) // 2
        title_y = 150
        
        # Efecto de color cambiante
        color_hue = (i * 2) % 180
        color = cv2.cvtColor(np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = tuple(map(int, color))
        
        cv2.putText(frame, title, (title_x, title_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Subtítulo
        subtitle = "Activado por detección de parpadeo"
        subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        subtitle_x = (width - subtitle_size[0]) // 2
        cv2.putText(frame, subtitle, (subtitle_x, title_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Contador de frames
        time_elapsed = i / fps
        counter_text = f"Tiempo: {time_elapsed:.1f}s / {duration}s"
        cv2.putText(frame, counter_text, (50, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)
        
        # Barra de progreso
        progress = i / total_frames
        bar_width = 600
        bar_height = 30
        bar_x = (width - bar_width) // 2
        bar_y = 450
        
        # Fondo de la barra
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progreso
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + progress_width, bar_y + bar_height), 
                     (0, int(255 * progress), int(255 * (1 - progress))), -1)
        
        # Borde
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Porcentaje
        percent_text = f"{progress * 100:.0f}%"
        cv2.putText(frame, percent_text, 
                   (bar_x + bar_width + 20, bar_y + bar_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Círculo animado
        circle_radius = 30
        circle_x = width // 2
        circle_y = 300
        angle = i * 0.1
        
        # Movimiento circular
        offset_x = int(100 * np.sin(angle))
        offset_y = int(50 * np.cos(angle * 0.7))
        
        cv2.circle(frame, (circle_x + offset_x, circle_y + offset_y), 
                  circle_radius, (0, 255, 255), -1)
        
        # Escribir frame
        out.write(frame)
        
        # Mostrar progreso
        if i % 30 == 0:
            print(f"Progreso: {progress*100:.0f}%")
    
    out.release()
    print(f"\nVideo creado exitosamente: sample_video.mp4")
    print(f"Dimensiones: {width}x{height}")
    print(f"Duración: {duration} segundos")
    print(f"Tamaño: {Path('sample_video.mp4').stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    # Verificar e instalar dependencias
    print("Verificando dependencias...")
    
    required_packages = {
        'opencv-python': 'cv2',
        'pygame': 'pygame',
        'pillow': 'PIL',
        'ultralytics': 'ultralytics',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package} instalado")
        except ImportError:
            print(f"✗ {package} no encontrado")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nPaquetes faltantes: {', '.join(missing_packages)}")
        install = input("¿Instalar automáticamente? (s/n): ").lower()
        
        if install == 's':
            import subprocess
            import sys
            
            for package in missing_packages:
                print(f"Instalando {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            
            print("\nReinicia el programa después de la instalación.")
            input("Presiona Enter para salir...")
            exit()
        else:
            print("\nInstala manualmente los paquetes faltantes:")
            print(f"pip install {' '.join(missing_packages)}")
            input("\nPresiona Enter para salir...")
            exit()
    
    # Ejecutar programa principal
    main()