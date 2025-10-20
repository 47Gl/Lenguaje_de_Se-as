# Importación de librerías
import pandas as pd  # Para manejar datos en formato tabular
from sklearn.ensemble import RandomForestClassifier  # Algoritmo de ML
from sklearn.model_selection import train_test_split  # Para dividir datos
import joblib  # Para guardar/leer modelos entrenados

# Configuración de rutas y parámetros
DATOS_ENTRADA = "data/landmarks.csv"  # Datos de entrenamiento
MODELO_SALIDA = "model.pkl"  # Donde se guardará el modelo
PRUEBAS_PORCENTAJE = 0.2  # 20% de datos para prueba
SEMILLA = 42  # Para reproducibilidad

def entrenar_modelo():
    """Función principal para entrenar el modelo"""
    print("\n=== INICIANDO ENTRENAMIENTO ===")
   
    try:
        # 1. Cargar datos desde el archivo CSV
        datos = pd.read_csv(DATOS_ENTRADA)
        print(f"📊 Datos cargados: {len(datos)} muestras")
       
        # Verificar si hay suficientes datos
        if len(datos) < 100:
            print("⚠️ Advertencia: Pocas muestras (<100), el modelo puede no aprender bien")
    except Exception as e:
        print(f"❌ Error al cargar datos: {str(e)}")
        return

    # 2. Preparar datos para entrenamiento
    X = datos.iloc[:, :-1]  # Todas las columnas excepto la última (características)
    y = datos["label"]      # Última columna (etiquetas/clases)
   
    # 3. Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PRUEBAS_PORCENTAJE,  # Porcentaje para prueba
        random_state=SEMILLA)  # Semilla para reproducibilidad
    print(f"✂️ División: {len(X_train)} entrenamiento, {len(X_test)} prueba")

    # 4. Crear y entrenar modelo Random Forest
    print("🧠 Entrenando modelo...")
    modelo = RandomForestClassifier(
        n_estimators=50,  # Número de árboles en el bosque
        random_state=SEMILLA)  # Semilla para reproducibilidad
    modelo.fit(X_train, y_train)  # Entrenamiento real

    # 5. Evaluar el modelo con datos de prueba
    precision = modelo.score(X_test, y_test)
    print(f"✅ Precisión en prueba: {precision*100:.2f}%")

    # 6. Guardar modelo entrenado para uso futuro
    joblib.dump(modelo, MODELO_SALIDA)
    print(f"💾 Modelo guardado como '{MODELO_SALIDA}'")

if __name__ == "__main__":
    entrenar_modelo()
