import pandas as pd
from fastapi import FastAPI
import  uvicorn

app = FastAPI()

df = pd.read_csv('datos.csv')

# Función para obtener la cantidad de películas producidas en un idioma específico
@app.get('/peliculas_idioma')
def peliculas_idioma(idioma: str):
    # Filtrar el DataFrame por el idioma especificado
    peliculas_filtradas = df[df['original_language'] == idioma]

    # Obtener la cantidad de películas en el idioma
    cantidad_peliculas = len(peliculas_filtradas)

    return f"{cantidad_peliculas} películas fueron estrenadas en {idioma}"


@app.get('/peliculas_duracion')
def peliculas_duracion(pelicula: str):
    # Filtrar el DataFrame por el título de la película especificada
    pelicula_filtrada = df[df['title'] == pelicula]

    # Verificar si la película existe en el DataFrame
    if len(pelicula_filtrada) == 0:
        return f"No se encontró la película {pelicula}"

    # Obtener la duración y el año de la película
    duracion = pelicula_filtrada['runtime'].values[0]
    anio = pelicula_filtrada['release_year'].values[0]

    return f"Película: {pelicula}. Duración: {duracion} minutos. Estrenada en el Año: {anio}"


@app.get('/franquicia')
def franquicia(franquicia: str):
    # Filtrar el DataFrame por la franquicia especificada
    franquicia_filtrada = df[df['belongs_to_collection'] == franquicia]

    # Verificar si se encontraron películas de la franquicia
    if len(franquicia_filtrada) == 0:
        return f"No se encontraron películas de la franquicia {franquicia}"

    # Obtener la cantidad de películas, la ganancia total y el promedio de ganancia
    cantidad_peliculas = len(franquicia_filtrada)
    ganancia_total = franquicia_filtrada['revenue'].sum()
    ganancia_promedio = franquicia_filtrada['revenue'].mean()

    return f"La franquicia {franquicia} posee {cantidad_peliculas} películas, una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}"


@app.get('/peliculas_pais')
def peliculas_pais(pais: str):
    # Filtrar el DataFrame por el país especificado
    peliculas_filtradas = df[df['production_countries'].str.contains(pais, na=False)]

    # Obtener la cantidad de películas en el país
    cantidad_peliculas = len(peliculas_filtradas)

    return f"Se produjeron {cantidad_peliculas} películas en el país {pais}"


@app.get('/productoras_exitosas')
def productoras_exitosas(productora: str):
    # Filtrar el DataFrame por la productora especificada
    productora_filtrada = df[df['production_companies'].str.contains(productora, na=False)]

    # Obtener el revenue total y la cantidad de películas de la productora
    revenue_total = productora_filtrada['revenue'].sum()
    cantidad_peliculas = len(productora_filtrada)

    return f"La productora {productora} ha tenido un revenue de {revenue_total} y ha realizado {cantidad_peliculas} películas"


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)