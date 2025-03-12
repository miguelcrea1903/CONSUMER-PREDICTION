import requests
import json
import pandas as pd
from modelo_hibrido import aplicar_modelo_hibrido

# Función para enviar datos a la API
def enviar_datos_a_api(datos):
    """
    Envía los datos procesados por el modelo híbrido a la API.
    
    Parámetros:
        datos (pd.DataFrame): DataFrame con los resultados del modelo híbrido.
    
    Retorna:
        response (requests.Response): Respuesta de la API.
    """
    # Convertir el DataFrame a formato JSON
    datos_json = datos.to_json(orient="records")

    # URL de la API (endpoint para recibir datos)
    api_url = "https://tu-api.com/endpoint-datos"

    # Encabezados de la solicitud
    headers = {"Content-Type": "application/json"}

    # Enviar los datos a la API
    response = requests.post(api_url, data=datos_json, headers=headers)

    return response

# Función para hacer preguntas a la API
def hacer_pregunta_a_api(pregunta):
    """
    Envía una pregunta relacionada con publicidad, segmentación o marketing a la API.
    
    Parámetros:
        pregunta (str): Pregunta que deseas hacer.
    
    Retorna:
        respuesta (str): Respuesta de la API.
    """
    # URL de la API (endpoint para preguntas)
    api_url = "https://tu-api.com/endpoint-preguntas"

    # Encabezados de la solicitud
    headers = {"Content-Type": "application/json"}

    # Cuerpo de la solicitud (pregunta en formato JSON)
    payload = json.dumps({"pregunta": pregunta})

    # Enviar la pregunta a la API
    response = requests.post(api_url, data=payload, headers=headers)

    # Verificar la respuesta
    if response.status_code == 200:
        return response.json().get("respuesta", "No se recibió una respuesta válida.")
    else:
        return f"Error al hacer la pregunta: {response.status_code}"

# Función principal para integrar el modelo híbrido con la API
def integrar_modelo_con_api(df):
    """
    Integra los resultados del modelo híbrido con la API y permite hacer preguntas.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de los clientes.
    """
    # Aplicar el modelo híbrido
    resultados_hibridos = aplicar_modelo_hibrido(df)

    # Enviar los resultados a la API
    respuesta_api = enviar_datos_a_api(resultados_hibridos)

    if respuesta_api.status_code == 200:
        print("Datos enviados correctamente a la API.")
    else:
        print(f"Error al enviar los datos a la API: {respuesta_api.status_code}")

    # Ejemplo de preguntas relacionadas con publicidad, segmentación y marketing
    preguntas = [
        "¿Cuáles son los segmentos de clientes más rentables?",
        "¿Qué categorías de productos tienen mayor impacto en las ventas?",
        "¿Cómo puedo mejorar la segmentación de mercado?",
        "¿Qué estrategias de publicidad recomiendas para el clúster 2?"
    ]

    # Hacer preguntas a la API
    for pregunta in preguntas:
        respuesta = hacer_pregunta_a_api(pregunta)
        print(f"Pregunta: {pregunta}")
        print(f"Respuesta: {respuesta}")
        print("-" * 50)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar los datos desde un archivo CSV (simulado aquí)
    df = pd.read_csv("datos_clientes.csv", delimiter=";", encoding="latin1")

    # Integrar el modelo híbrido con la API
    integrar_modelo_con_api(df)