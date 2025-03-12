# kmeans.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def aplicar_kmeans(df):
    """
    Aplica el algoritmo K-Means para segmentar clientes basados en edad, ubicación y categorías de productos comprados.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de los clientes.
    
    Retorna:
        df (pd.DataFrame): DataFrame original con una columna adicional 'Cluster' que indica el grupo al que pertenece cada cliente.
    """
    # Seleccionar características relevantes
    caracteristicas = df[['Edad', 'Ubicación', 'Categoría Producto']]

    # Preprocesamiento de datos
    # OneHotEncoder para variables categóricas (Ubicación y Categoría Producto)
    # StandardScaler para la variable numérica (Edad)
    preprocesador = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Ubicación', 'Categoría Producto']),
            ('num', StandardScaler(), ['Edad'])
        ])

    # Crear un pipeline con el preprocesador y K-Means
    pipeline = Pipeline([
        ('preprocesador', preprocesador),
        ('kmeans', KMeans(n_clusters=4, random_state=42))  # Ajusta el número de clústeres según sea necesario
    ])

    # Aplicar el pipeline a los datos
    df['Cluster'] = pipeline.fit_predict(caracteristicas)

    return df