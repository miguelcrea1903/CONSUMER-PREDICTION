# svd.py
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def aplicar_svd(df):
    """
    Aplica el algoritmo SVD para generar recomendaciones de productos basadas en el comportamiento de compra de los clientes.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de los clientes.
    
    Retorna:
        recomendaciones (pd.DataFrame): DataFrame con las recomendaciones de productos para cada cliente.
    """
    # Crear una matriz de usuario-producto
    matriz_usuario_producto = df.pivot_table(index='Cedula Cliente', columns='Categoría Producto', values='Cantidad Comprada', aggfunc='sum', fill_value=0)

    # Verificar el número de características (categorías de productos)
    n_caracteristicas = matriz_usuario_producto.shape[1]
    n_componentes = min(5, n_caracteristicas)  # Ajustar el número de componentes

    # Aplicar SVD
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    matriz_reducida = svd.fit_transform(matriz_usuario_producto)

    # Calcular la similitud de coseno entre los clientes
    similitud_clientes = cosine_similarity(matriz_reducida)

    # Convertir la matriz de similitud a un DataFrame
    similitud_df = pd.DataFrame(similitud_clientes, index=matriz_usuario_producto.index, columns=matriz_usuario_producto.index)

    # Generar recomendaciones basadas en la similitud de clientes
    recomendaciones = {}
    for cliente in matriz_usuario_producto.index:
        # Obtener los clientes más similares
        clientes_similares = similitud_df[cliente].sort_values(ascending=False).index[1:6]  # Excluir al propio cliente

        # Obtener los productos comprados por los clientes similares
        productos_recomendados = df[df['Cedula Cliente'].isin(clientes_similares)]['Categoría Producto'].value_counts().index[:5]

        # Si no hay suficientes productos recomendados, rellenar con None
        productos_recomendados = list(productos_recomendados)
        while len(productos_recomendados) < 5:
            productos_recomendados.append(None)  # Rellenar con None si no hay suficientes productos

        # Guardar las recomendaciones
        recomendaciones[cliente] = productos_recomendados

    # Convertir las recomendaciones a un DataFrame
    recomendaciones_df = pd.DataFrame.from_dict(recomendaciones, orient='index', columns=['Rec1', 'Rec2', 'Rec3', 'Rec4', 'Rec5'])

    return recomendaciones_df