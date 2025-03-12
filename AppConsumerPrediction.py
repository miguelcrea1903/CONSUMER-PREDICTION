import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
from kmeans import aplicar_kmeans  # Importar la función de K-Means
from tfidf import aplicar_tfidf  # Importar la función de TF-IDF
from svd import aplicar_svd  # Importar la función de SVD
from modelo_hibrido import aplicar_modelo_hibrido  # Importar la función del Modelo Híbrido

# Configuración de la API de DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # URL base
DEEPSEEK_API_KEY = "sk-25466ab742444f188f2847c4ebce9a61"  # Tu API Key

# Función para hacer preguntas a la API de DeepSeek
def hacer_pregunta_a_deepseek(pregunta, contexto):
    """
    Envía una pregunta relacionada con publicidad, segmentación o marketing a la API de DeepSeek.
    
    Parámetros:
        pregunta (str): Pregunta que deseas hacer.
        contexto (str): Contexto o datos relevantes para la pregunta.
    
    Retorna:
        respuesta (str): Respuesta de la API.
    """
    # Encabezados de la solicitud
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    # Cuerpo de la solicitud
    payload = {
        "model": "deepseek-chat",  # Usar el modelo de chat
        "messages": [
            {"role": "system", "content": "Eres un asistente útil especializado en publicidad, segmentación y marketing."},
            {"role": "user", "content": f"{pregunta}\n\nContexto:\n{contexto}"}
        ],
        "stream": False  # Deshabilitar transmisión para respuestas inmediatas
    }

    # Enviar la solicitud a la API
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)

    # Verificar la respuesta
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error al hacer la pregunta: {response.status_code}"

# Imagen fija
st.image("Recurso1.png", caption="By Miguel Angel Mejia",  use_container_width=True)

# Título de la aplicación
st.title("PREDICCIÓN DE COMPORTAMIENTO DE LOS CONSUMIDORES")

# Widget para cargar el archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Verificar si el usuario ha subido un archivo
if uploaded_file is not None:
    # Cargar el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(uploaded_file, delimiter=";", encoding="latin1")

    # Mostrar las primeras filas del DataFrame
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df.head(900))

    # Mostrar información general del DataFrame
    st.write("Información del DataFrame:")
    st.text(df.info())  # Usar st.text para evitar problemas con la visualización

    # Verificar valores faltantes
    st.write("Valores faltantes por columna:")
    st.write(df.isnull().sum())

    # Preprocesamiento de datos
    st.subheader("Preprocesamiento de datos")

    # Eliminar filas con valores faltantes (si es necesario)
    if st.checkbox("Eliminar filas con valores faltantes"):
        df = df.dropna()
        st.write("Se han eliminado filas con valores faltantes.")

    # Eliminar filas duplicadas (si es necesario)
    if st.checkbox("Eliminar filas duplicadas"):
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        final_rows = df.shape[0]
        st.write(f"Se han eliminado {initial_rows - final_rows} filas duplicadas.")

    # Convertir la columna de fecha a formato datetime (si existe)
    if "fecha_compra" in df.columns:
        try:
            df['fecha_compra'] = pd.to_datetime(df['fecha_compra'])
            st.write("Columna 'fecha_compra' convertida a formato datetime.")
        except Exception as e:
            st.error(f"No se pudo convertir 'fecha_compra' a formato datetime: {e}")

    # Mostrar el DataFrame procesado
    st.write("Vista previa de los datos procesados:")
    st.dataframe(df.head(900))

    # Visualización de datos: Gráfico de barras de utilidad por mes
    st.subheader("Visualización de datos")

    # Verificar si las columnas necesarias están presentes
    if "Mes" in df.columns and "Utilidad Producto" in df.columns:
        # Agrupar la utilidad por mes
        utilidad_por_mes = df.groupby("Mes")["Utilidad Producto"].sum()

        # Crear el gráfico de barras
        fig, ax = plt.subplots()
        utilidad_por_mes.plot(kind="bar", color="blue", ax=ax)
        ax.set_xlabel("Mes")
        ax.set_ylabel("Utilidad Total")
        ax.set_title("Utilidad Generada por Mes")

        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
    else:
        st.error("Las columnas 'Mes' y 'Utilidad Producto' son necesarias para generar el gráfico.")

    # Aplicar K-Means para segmentación de clientes
    st.subheader("Segmentación de Clientes (K-Means)")

    if st.button("Ejecutar K-Means"):
        # Verificar que las columnas necesarias estén presentes
        if all(col in df.columns for col in ['Edad', 'Ubicación', 'Categoría Producto']):
            # Aplicar K-Means
            df = aplicar_kmeans(df)

            # Mostrar los resultados
            st.write("Resultados de la segmentación de clientes:")
            st.dataframe(df[['Cedula Cliente', 'Nombre Cliente', 'Edad', 'Ubicación', 'Categoría Producto', 'Cluster']].head(200))

            # Visualización de los clústeres
            st.write("Distribución de clústeres:")
            cluster_counts = df['Cluster'].value_counts()
            st.bar_chart(cluster_counts)
        else:
            st.error("Las columnas 'Edad', 'Ubicación' y 'Categoría Producto' son necesarias para ejecutar K-Means.")

    # Aplicar TF-IDF para analizar categorías de productos
    st.subheader("Análisis de Categorías de Productos (TF-IDF)")

    if st.button("Ejecutar TF-IDF"):
        # Verificar que la columna necesaria esté presente
        if 'Categoría Producto' in df.columns:
            # Aplicar TF-IDF
            tfidf_resultados = aplicar_tfidf(df)

            # Mostrar los resultados
            st.write("Resultados de TF-IDF:")
            st.dataframe(tfidf_resultados.head(900))

            # Visualización de las categorías más importantes
            st.write("Categorías más importantes por cliente:")
            st.bar_chart(tfidf_resultados.sum().sort_values(ascending=False))
        else:
            st.error("La columna 'Categoría Producto' es necesaria para ejecutar TF-IDF.")

    # Aplicar SVD para recomendaciones de productos
    st.subheader("Recomendaciones de Productos (SVD)")

    if st.button("Ejecutar SVD"):
        # Verificar que las columnas necesarias estén presentes
        if all(col in df.columns for col in ['Cedula Cliente', 'Categoría Producto', 'Cantidad Comprada']):
            # Aplicar SVD
            recomendaciones = aplicar_svd(df)

            # Mostrar los resultados
            st.write("Recomendaciones de productos para cada cliente:")
            st.dataframe(recomendaciones.head(900))
        else:
            st.error("Las columnas 'Cedula Cliente', 'Categoría Producto' y 'Cantidad Comprada' son necesarias para ejecutar SVD.")

    # Aplicar el Modelo Híbrido
    st.subheader("Recomendaciones Híbridas")

    if st.button("Ejecutar Modelo Híbrido"):
        # Verificar que las columnas necesarias estén presentes
        if all(col in df.columns for col in ['Cedula Cliente', 'Categoría Producto', 'Cantidad Comprada', 'Edad', 'Ubicación']):
            # Aplicar el Modelo Híbrido
            recomendaciones_hibridas = aplicar_modelo_hibrido(df)

            # Mostrar los resultados
            st.write("Recomendaciones híbridas para cada cliente:")
            st.dataframe(recomendaciones_hibridas.head(900))

            # Permitir al usuario hacer preguntas relacionadas con publicidad, segmentación y marketing
            st.subheader("Haz preguntas sobre publicidad, segmentación y marketing")
            pregunta_usuario = st.text_input("Escribe tu pregunta:")

            if st.button("Enviar pregunta"):
                if pregunta_usuario:
                    # Usar los resultados del modelo híbrido como contexto
                    contexto = recomendaciones_hibridas.to_string()
                    respuesta = hacer_pregunta_a_deepseek(pregunta_usuario, contexto)
                    st.write(f"**Pregunta:** {pregunta_usuario}")
                    st.write(f"**Respuesta:** {respuesta}")
                else:
                    st.warning("Por favor, escribe una pregunta.")
        else:
            st.error("Las columnas 'Cedula Cliente', 'Categoría Producto', 'Cantidad Comprada', 'Edad' y 'Ubicación' son necesarias para ejecutar el Modelo Híbrido.")

    # Descargar el DataFrame procesado
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df)
    st.download_button(
        label="Descargar datos procesados como CSV",
        data=csv_data,
        file_name="datos_procesados.csv",
        mime="text/csv",
    )
else:
    st.write("Por favor, sube un archivo CSV para comenzar.")