# Importamos las librerias requeridas para la creacion de la API
import pandas as pd
from fastapi import FastAPI
import  uvicorn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Definición de la aplicación FastAPI
app = FastAPI()

# Cargamos los datos
df = pd.read_csv('datos.csv')

# Función para obtener la cantidad de películas producidas en un idioma específico
@app.get('/peliculas_idioma')
def peliculas_idioma(idioma: str):
    # Filtrar el DataFrame por el idioma especificado y obtiene la cantidad de peliculas
    peliculas_filtradas = df[df['original_language'] == idioma]
    cantidad_peliculas = len(peliculas_filtradas)

    return f"{cantidad_peliculas} películas fueron estrenadas en {idioma}"

# Función para obtener la duracion y el año de estreno de la pelicula consultada
@app.get('/peliculas_duracion')
def peliculas_duracion(pelicula: str):
    # Filtrar el DataFrame por el título
    pelicula_filtrada = df[df['title'] == pelicula]

    # Verificar si la película existe en el DataFrame
    if len(pelicula_filtrada) == 0:
        return f"No se encontró la película {pelicula}"

    # Obtener la duración y el año de la película
    duracion = pelicula_filtrada['runtime'].values[0]
    anio = pelicula_filtrada['release_year'].values[0]

    return f"Película: {pelicula}. Duración: {duracion} minutos. Estrenada en el Año: {anio}"

# Función para obtener la cantidad de peliculas y la ganancia total de una franquicia
@app.get('/franquicia')
def franquicia(franquicia: str):
    # Filtrar el DataFrame por la franquicia
    franquicia_filtrada = df[df['belongs_to_collection'] == franquicia]

    # Verificar si se encontraron películas de la franquicia
    if len(franquicia_filtrada) == 0:
        return f"No se encontraron películas de la franquicia {franquicia}"

    # Obtener la cantidad de películas, la ganancia total y el promedio de ganancia
    cantidad_peliculas = len(franquicia_filtrada)
    ganancia_total = franquicia_filtrada['revenue'].sum()
    ganancia_promedio = franquicia_filtrada['revenue'].mean()

    return f"La franquicia {franquicia} posee {cantidad_peliculas} películas, una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}"

# Función para obtener la cantidad de peliculas producidas en un pais
@app.get('/peliculas_pais')
def peliculas_pais(pais: str):
    # Filtrar el DataFrame por el país
    peliculas_filtradas = df[df['production_countries'].str.contains(pais, na=False)]

    # Obtener la cantidad de películas en el país
    cantidad_peliculas = len(peliculas_filtradas)

    return f"Se produjeron {cantidad_peliculas} películas en el país {pais}"

# Función para obtener la ganancia total y la cantidad de peliculas de una productora
@app.get('/productoras_exitosas')
def productoras_exitosas(productora: str):
    # Filtrar el DataFrame por la productora especificada
    productora_filtrada = df[df['production_companies'].str.contains(productora, na=False)]

    # Obtener la ganancia total y la cantidad de películas de la productora
    revenue_total = productora_filtrada['revenue'].sum()
    cantidad_peliculas = len(productora_filtrada)

    return f"La productora {productora} ha tenido un revenue de {revenue_total} y ha realizado {cantidad_peliculas} películas"

# Función para obtener el exito (a travez de la ganancia de inversion total) y las pelicualas (con sus titulos, fecha de lanzamiento, ganancia de inversion individual, costo y ganancias de las mismas) de un director
@app.get('/get_director')
def get_director(nombre_director: str):
    # Filtrar el DataFrame por el nombre del director especificado y por valores no nulos en la columna 'directores'
    director_filtrado = df[(df['directores'].str.contains(nombre_director, case=False)) & (pd.notnull(df['directores']))]

    # Calcular la suma de los retornos de las películas
    suma_retornos = director_filtrado['return'].sum()

    # Obtener la información de las películas del director
    peliculas = []
    for _, row in director_filtrado.iterrows():
        pelicula = {
            "titulo": row['title'],
            "fecha_lanzamiento": row['release_date'],
            "retorno_individual": row['return'],
            "costo": row['budget'],
            "ganancia": row['revenue']
        }
        peliculas.append(pelicula)

    return {
        "exito": suma_retornos,
        "peliculas": peliculas
    }


# Crea una muestra aleatoria con 5000 filas del dataset
muestra = df.head(5000)

# Crea la matriz de características TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(muestra['overview'].fillna(''))

# Calcula la similitud coseno entre todas las descripciones
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

# Función que calcula la similitud entre dos conjuntos de géneros cinematográficos
def calcular_similitud_generos(generos_referencia, generos):
    if pd.isnull(generos_referencia) or pd.isnull(generos):
        return 0.0

    generos_referencia = set(generos_referencia.split('|'))
    generos = set(generos.split('|'))
    intersection = len(generos_referencia.intersection(generos))
    union = len(generos_referencia.union(generos))
    return intersection / union

# Crea la función de recomendación
@app.get("/recomendacion")
def recomendacion(titulo: str):
    # Busca el id de la pelicula en la muestra
    idx = muestra[muestra['title'] == titulo].index[0]
    sim_cosine = cosine_similarity[idx]

    # Calcula la similitud de géneros para la película de referencia
    generos_referencia = muestra.loc[idx, 'genres']
    sim_generos = muestra['genres'].apply(lambda x: calcular_similitud_generos(generos_referencia, x))

    # Combina la similitud coseno y la similitud de géneros
    sim_total = sim_cosine + sim_generos

    # Obtiene los índices de las películas similares
    similar_indices = sim_total.argsort()[::-1][1:6]

    # Obtiene los títulos de las películas similares
    similar_movies = muestra['title'].iloc[similar_indices].tolist()

    return similar_movies


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)