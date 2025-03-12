# CONSUMER-PREDICTION
Este proyecto implementa un modelo híbrido de predicción del comportamiento del consumidor utilizando técnicas de aprendizaje automático no supervisado . El objetivo principal es identificar segmentos de mercado ocultos y proporcionar recomendaciones personalizadas a través del análisis de datos de clientes y sus patrones de compra.

El sistema combina varios algoritmos:

K-Means para segmentación de clientes.
TF-IDF para analizar categorías de productos.
SVD (Descomposición de Valores Singulares) para generar recomendaciones basadas en similitudes entre usuarios.
Además, el modelo se integra con una API externa para realizar consultas avanzadas relacionadas con estrategias de marketing y segmentación. El proyecto incluye una interfaz interactiva desarrollada con Streamlit , que permite cargar datos, visualizar resultados y recomendaciones explorar.

🛠️ kmeans.py: Segmentación de Clientes
Este módulo aplica el algoritmo K-Means para dividir a los clientes en grupos (clústeres) según sus características.

Procesa datos como edad, ubicación y categoría de productos comprados.
Normaliza valores numéricos y convierte variables categóricas en representaciones numéricas mediante técnicas como StandardScaler y OneHotEncoder .
Crea un pipeline que combina preprocesamiento y el algoritmo K-Means para asignar un clúster a cada cliente.
Resultado: Una columna adicional en los datos que indica el grupo al que pertenece cada cliente.

🔍 tfidf.py: Análisis de Categorías de Productos
Este módulo utiliza la técnica TF-IDF para identificar la importancia relativa de las categorías de productos comprados por cada cliente.

Agrupa las categorías de productos comprados por cliente en un único texto.
Calcula la frecuencia de las palabras en los datos, evaluando su relevancia mediante el peso TF-IDF .
Convierta los resultados en una matriz legible para análisis.
Resultado: Identifica las categorías más representativas para cada cliente.
📉 svd.py: Recomendaciones Basadas en SVD
Aquí se implementa SVD (Descomposición de Valores Singulares) para generar recomendaciones de productos.


Crea una matriz de usuario-producto basada en las cantidades compradas.
Reduzca la dimensionalidad de los datos para capturar relaciones clave entre clientes y productos.
Calcula similitudes entre usuarios utilizando métricas como la similitud de coseno.
Identifique los productos recomendados para cada cliente según sus similitudes con otros usuarios.
Resultado: Un conjunto de recomendaciones personalizadas para cada cliente.

🧠 modelo_hibrido.py: Modelo Híbrido
Este módulo integra los resultados de K-Means , TF-IDF y SVD para generar una visión más completa.

Combina los clústeres generados por K-Means, las categorías relevantes de TF-IDF y las recomendaciones de SVD en un solo DataFrame.
Crea una lista final de recomendaciones personalizadas para cada cliente.
Resultado: Proporciona una vista enriquecida del comportamiento del consumidor y productos recomendados.
🌐 Integración API: Comunicación con APIs Externas
Este módulo interactúa con una API externa para enriquecer el análisis y generar insights adicionales.

Envía los datos del modelo híbrido a la API para almacenarlos o procesarlos.
Realiza preguntas relacionadas con estrategias de marketing, publicidad y segmentación, y obtiene respuestas detalladas.
Resultado: Amplía la capacidad de análisis al integrar recomendaciones con una API especializada.

📊 AppConsumerPrediction.py: Interfaz de usuario
Este es el módulo principal que conecta todos los componentes y proporciona una interfaz amigable mediante Streamlit .

Permite cargar un archivo CSV con datos de clientes.
Realiza preprocesamiento de datos y aplica los algoritmos (K-Means, TF-IDF, SVD y Modelo Híbrido).
Genera visualizaciones de los resultados, como gráficos de barras y clústeres.
Permite realizar preguntas personalizadas mediante la integración con la API externa.
Resultado: Una plataforma interactiva para explorar los datos y obtener recomendaciones en tiempo real.

🎯 Relación entre módulos
Los módulos de análisis (K-Means, TF-IDF, SVD) preparan los datos y generan insights clave.
El modelo híbrido combina estos resultados en una visiónificada.
La integración API amplía la funcionalidad mediante consultas avanzadas.
La interfaz de usuario en Streamlit centraliza todas las funciones y facilita el acceso para usuarios no técnicos.



