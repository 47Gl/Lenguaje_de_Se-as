# Importación de librerías
import cv2  # Para el manejo de la cámara y video
import mediapipe as mp  # Para detectar las manos
import pandas as pd  # Para manejar datos en formato CSV
import os  # Para manejo de archivos y carpetas
from pathlib import Path  # Para manejo de rutas de archivos

# Configuración inicial
DATA_DIR = "data"  # Nombre de la carpeta donde se guardarán los datos
DATA_FILE = os.path.join(DATA_DIR, "landmarks.csv")  # Ruta completa del archivo CSV
SAMPLES_PER_LETTER = 100  # Número objetivo de muestras por letra
AUTO_SAVE_INTERVAL = 10  # Guardar automáticamente cada X muestras

# Crear la carpeta si no existe
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# Inicializar el detector de manos de MediaPipe
mp_hands = mp.solutions.hands  # Módulo para detección de manos
# Configuración del detector:
# static_image_mode=False -> Optimizado para video
# max_num_hands=1 -> Solo detecta una mano
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def cambiar_letra(letra_actual):
    """Función para cambiar la letra actual que se está recolectando"""
    print(f"\nLetra actual: {letra_actual}")
    # Pide al usuario una nueva letra
    nueva_letra = input("Ingresa nueva letra (A-Z) o Enter para mantener: ").upper()
    # Devuelve la nueva letra si es válida, de lo contrario mantiene la actual
    return nueva_letra if nueva_letra.isalpha() else letra_actual

def main():
    """Función principal del programa"""
    letra_actual = "A"  # Letra inicial para recolectar datos
    muestras = []  # Lista temporal para almacenar muestras
    total_muestras = 0  # Contador de muestras totales guardadas
   
    # Inicializar la cámara (el 0 indica la cámara predeterminada)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # Verificar si la cámara se abrió correctamente
        print("¡Error al abrir la cámara!")
        return

    # Mensajes de instrucción para el usuario
    print("=== MODO CAPTURA ===")
    print("Instrucciones:")
    print("1. Presiona 'S' para guardar muestra")
    print("2. Presiona 'C' para cambiar letra")
    print("3. Presiona ESC para salir")

    # Bucle principal del programa
    while True:
        # Capturar frame por frame
        ret, frame = cap.read()  # ret indica si fue exitoso, frame contiene la imagen
        if not ret:  # Si no se pudo capturar el frame, continuar al siguiente ciclo
            continue

        # Convertir el color de BGR (OpenCV) a RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar el frame para detectar manos
        resultados = hands.process(rgb_frame)
       
        # Preparar texto informativo para mostrar en pantalla
        texto_info = [
            f"Letra: {letra_actual}",
            f"Muestras: {len(muestras)}",
            f"Total: {total_muestras + len(muestras)}",
            "'S'-Guardar  'C'-Cambiar  ESC-Salir"
        ]
       
        # Dibujar el texto en el frame
        for i, texto in enumerate(texto_info):
            # Parámetros: imagen, texto, posición, fuente, tamaño, color, grosor
            cv2.putText(frame, texto, (10, 30 + i*30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                      (0, 255, 0) if i < 3 else (255, 255, 0), 2)

        # Si se detectaron manos, dibujar los landmarks (puntos clave)
        if resultados.multi_hand_landmarks:
            # Dibujar los puntos y conexiones de la mano
            mp.solutions.drawing_utils.draw_landmarks(
                frame,  # Imagen donde dibujar
                resultados.multi_hand_landmarks[0],  # Primera mano detectada
                mp_hands.HAND_CONNECTIONS)  # Dibujar conexiones entre puntos

        # Mostrar el frame resultante en una ventana
        cv2.imshow("Capturador de Gestos", frame)

        # Esperar por teclas presionadas (1ms de espera)
        tecla = cv2.waitKey(1)
        if tecla == 27:  # Tecla ESC (27 es su código ASCII)
            break  # Salir del bucle
        elif tecla == ord('s') and resultados.multi_hand_landmarks:
            # Si se presiona 'S' y hay una mano detectada
            landmarks = []  # Lista para almacenar coordenadas
            # Extraer coordenadas (x,y,z) de cada punto de la mano
            for punto in resultados.multi_hand_landmarks[0].landmark:
                landmarks.extend([punto.x, punto.y, punto.z])
            # Añadir las coordenadas junto con la letra actual
            muestras.append(landmarks + [letra_actual])
            print(f"Muestra {len(muestras)} guardada para {letra_actual}")
           
            # Guardar automáticamente cada cierto número de muestras
            if len(muestras) % AUTO_SAVE_INTERVAL == 0:
                # Crear DataFrame y guardar en CSV
                pd.DataFrame(muestras).to_csv(
                    DATA_FILE,  # Ruta del archivo
                    mode='a',  # Append (añadir al existente)
                    header=not os.path.exists(DATA_FILE),  # Encabezado solo si es nuevo archivo
                    index=False)  # No guardar índice
                total_muestras += len(muestras)
                muestras = []  # Reiniciar lista temporal
               
        elif tecla == ord('c'):  # Si se presiona 'C' para cambiar letra
            letra_actual = cambiar_letra(letra_actual)

    # Al salir, guardar cualquier muestra pendiente
    if muestras:
        pd.DataFrame(muestras).to_csv(DATA_FILE, mode='a',
                                    header=not os.path.exists(DATA_FILE),
                                    index=False)

    # Liberar recursos
    cap.release()  # Liberar la cámara
    cv2.destroyAllWindows()  # Cerrar todas las ventanas

if __name__ == "__main__":
    main()