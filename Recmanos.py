# Importación de librerías
import cv2  # Para procesamiento de video
import mediapipe as mp  # Para detección de manos
import joblib  # Para cargar modelo entrenado
import numpy as np  # Para operaciones numéricas

# Configuración inicial
MODELO = r"C:\Users\luisn\Downloads\Inteligencia\Lenguaje_de_señas\proyecto\model.pkl"  # Ruta al modelo entrenado

# Inicializar detectores
mp_manos = mp.solutions.hands  # Módulo de detección de manos
manos = mp_manos.Hands()  # Configuración predeterminada
modelo = joblib.load(MODELO)  # Cargar modelo entrenado

# Configuración visual
COLOR_TEXTO = (255, 255, 255)  # Blanco (BGR)
COLOR_FONDO = (50, 50, 50)     # Gris oscuro
GROSOR_TEXTO = 3  # Grosor del texto en píxeles

def reconocer_gestos():
    """Función principal para reconocimiento en tiempo real"""
    cap = cv2.VideoCapture(0)  # Inicializar cámara
   
    while True:
        # Capturar frame de la cámara
        ret, frame = cap.read()
        if not ret:  # Si falla la captura, continuar
            continue

        # Convertir a RGB (necesario para MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar frame para detección de manos
        resultados = manos.process(rgb_frame)
       
        # Si se detecta al menos una mano
        if resultados.multi_hand_landmarks:
            # Dibujar landmarks y conexiones en el frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame,  # Imagen donde dibujar
                resultados.multi_hand_landmarks[0],  # Primera mano detectada
                mp_manos.HAND_CONNECTIONS)  # Conexiones entre puntos
           
            # Extraer coordenadas de los landmarks
            landmarks = []
            for punto in resultados.multi_hand_landmarks[0].landmark:
                landmarks.extend([punto.x, punto.y, punto.z])
           
            # Hacer predicción con el modelo
            letra = modelo.predict([landmarks])[0]  # Letra predicha
            confianza = modelo.predict_proba([landmarks]).max()  # Probabilidad
           
            # Mostrar resultado en pantalla
            texto = f"{letra} ({confianza*100:.0f}%)"  # Ej: "A (95%)"
            cv2.putText(frame, texto, (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2,  # Tamaño grande
                       COLOR_TEXTO, GROSOR_TEXTO)
       
        else:  # Si no se detectan manos
            cv2.putText(frame, "Mostrar mano a la cámara",
                       (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                       1, COLOR_TEXTO, 2)
       
        # Mostrar el frame resultante
        cv2.imshow("Reconocedor de Gestos", frame)
       
        # Salir si se presiona ESC (código 27)
        if cv2.waitKey(1) == 27:
            break

    # Liberar recursos al terminar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    reconocer_gestos()