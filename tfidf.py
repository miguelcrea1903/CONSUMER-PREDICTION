# tfidf.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def aplicar_tfidf(df):
    """
    Aplica el algoritmo TF-IDF para analizar la importancia de las categorías de productos comprados.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de los clientes.
    
    Retorna:
        tfidf_matrix (pd.DataFrame): Matriz TF-IDF que muestra la importancia de cada categoría para cada cliente.
    """
    # Agrupar las categorías de productos comprados por cliente
    categorias_por_cliente = df.groupby('Cedula Cliente')['Categoría Producto'].apply(lambda x: ' '.join(x))

    # Aplicar TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(categorias_por_cliente)

    # Convertir la matriz TF-IDF a un DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=categorias_por_cliente.index)

    return tfidf_df