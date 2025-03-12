# modelo_hibrido.py
import pandas as pd
from kmeans import aplicar_kmeans
from tfidf import aplicar_tfidf
from svd import aplicar_svd

def aplicar_modelo_hibrido(df):
    """
    Aplica un modelo híbrido que combina K-Means, TF-IDF y SVD para generar recomendaciones de productos.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de los clientes.
    
    Retorna:
        recomendaciones_hibridas (pd.DataFrame): DataFrame con las recomendaciones híbridas para cada cliente.
    """
    # Aplicar K-Means para segmentación de clientes
    df = aplicar_kmeans(df)

    # Aplicar TF-IDF para analizar categorías de productos
    tfidf_resultados = aplicar_tfidf(df)

    # Aplicar SVD para recomendaciones de productos
    svd_resultados = aplicar_svd(df)

    # Combinar los resultados de K-Means, TF-IDF y SVD
    recomendaciones_hibridas = pd.concat([df[['Cedula Cliente', 'Cluster']], tfidf_resultados, svd_resultados], axis=1)

    # Generar recomendaciones finales basadas en el modelo híbrido
    recomendaciones_hibridas['Recomendacion_Final'] = recomendaciones_hibridas.apply(
        lambda row: row[['Rec1', 'Rec2', 'Rec3', 'Rec4', 'Rec5']].tolist(), axis=1
    )

    return recomendaciones_hibridas