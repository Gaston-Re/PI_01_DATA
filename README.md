<p align="center">
  <img src="https://github.com/Streakuwu/PI_01_DATA/assets/117231675/986f0146-43bc-4fd7-9f41-c06a3d22854d" alt="Sublime's custom image"/>
</p>
# <h2 align=center> **PROYECTO INDIVIDUAL Nº1 - Gaston Re** </h2>

<p align="center">
  <img src="https://github.com/Streakuwu/PI_01_DATA/assets/117231675/f9cb31c6-6697-4e30-b066-f6e776c28b96" alt="Sublime's custom image"/>
</p>
# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

### Proceso - Descripción del problema (Contexto y rol a desarrollar)
En este proyecto realizare un proceso de ETL(Extraction, Transformation and Loading), creacion de una API (Application Programming Interface), EDA (Exploratory Data Analysis) y terminaremos con un modelo ML (Machine learning) de recomendacion de peliculas.
##### Contexto
Tienes tu modelo de recomendación dando unas buenas métricas 😏, y ahora, cómo lo llevas al mundo real? 👀

El ciclo de vida de un proyecto de Machine Learning debe contemplar desde el tratamiento y recolección de los datos (Data Engineer stuff) hasta el entrenamiento y mantenimiento del modelo de ML según llegan nuevos datos.
##### Rol a desarrollar
Empezaste a trabajar como Data Scientist en una start-up que provee servicios de agregación de plataformas de streaming. El mundo es bello y vas a crear tu primer modelo de ML que soluciona un problema de negocio: un sistema de recomendación que aún no ha sido puesto en marcha!

Vas a sus datos y te das cuenta que la madurez de los mismos es poca (ok, es nula 😭): Datos anidados, sin transformar, no hay procesos automatizados para la actualización de nuevas películas o series, entre otras cosas…. haciendo tu trabajo imposible 😩.

Debes empezar desde 0, haciendo un trabajo rápido de Data Engineer y tener un MVP (Minimum Viable Product) para el cierre del proyecto! Tu cabeza va a explotar 🤯, pero al menos sabes cual es, conceptualmente, el camino que debes de seguir ❗. Así que te espantas los miedos y te pones manos a la obra 💪

### ETL (Extraction, Transformation and Loading)
Descripcion De Mis Datos:
**Característica**\Descripción
- **adult**: Indica si la película tiene califiación X, exclusiva para adultos.
- **belongs_to_collection**: Un diccionario que indica a que franquicia o serie de películas pertenece la película
- **budget**: El presupuesto de la película, en dólares
- **genres**: Un diccionario que indica todos los géneros asociados a la película
- **homepage**: La página web oficial de la película
- **id**: ID de la pelicula
- **imdb_id**: IMDB ID de la pelicula
- **original_language**: Idioma original en la que se grabo la pelicula
- **original_title**: Titulo original de la pelicula
- **overview**: Pequeño resumen de la película
- **popularity**: Puntaje de popularidad de la película, asignado por TMDB (TheMoviesDataBase)
- **poster_path**: URL del póster de la película
- **production_companies**: Lista con las compañias productoras asociadas a la película
- **production_countries**: Lista con los países donde se produjo la película
- **release_date**: Fecha de estreno de la película
- **revenue**: Recaudación de la pelicula, en dolares
- **runtime**: Duración de la película, en minutos
- **spoken_languages**: Lista con los idiomas que se hablan en la pelicula
- **status**: Estado de la pelicula actual (si fue anunciada, si ya se estreno, etc)
- **tagline**: Frase celebre asociadaa la pelicula
- **title**: Titulo de la pelicula
- **video**: Indica si hay o no un trailer en video disponible en TMDB
- **vote_average**: Puntaje promedio de reseñas de la pelicula
- **vote_count**: Numeros de votos recibidos por la pelicula, en TMDB

**Transformaciones**: Para este MVP no necesitas perfección, ¡necesitas rapidez! ⏩ Vas a hacer estas, y solo estas, transformaciones a los datos:

1. Eliminar las columnas que no serán utilizadas, video,imdb_id,adult,original_title,poster_path y homepage.

2. Los valores nulos de los campos revenue, budget deben ser rellenados por el número 0.

3. Crear la columna con el retorno de inversión, llamada return con los campos revenue y budget, dividiendo estas dos últimas revenue / budget, cuando no hay datos disponibles para calcularlo, deberá tomar el valor 0.

4. Los valores nulos del campo release date deben eliminarse. De haber fechas, deberán tener el formato AAAA-mm-dd, además deberán crear la columna release_year donde extraerán el año de la fecha de estreno.

5. Algunos campos, como belongs_to_collection, production_companies y otros (ver diccionario de datos) están anidados, esto es o bien tienen un diccionario o una lista como valores en cada fila, ¡deberán desanidarlos para poder y unirlos al dataset de nuevo hacer alguna de las consultas de la API! O bien buscar la manera de acceder a esos datos sin desanidarlos.

*`encontraremos este proceso en el archivo Datos-ETL.ipynb`*
### API (Application Programming Interface)
**Desarrollo**: Propones disponibilizar los datos de la empresa usando el framework FastAPI. Las consultas que propones son las siguientes:

Deben crear 6 funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).
- **Consulta 1**
````
# Se ingresa un idioma (como están escritos en el dataset, no hay que traducirlos!). Debe devolver la cantidad de películas producidas en ese idioma.
@app.get('/peliculas_idioma')
def peliculas_idioma(idioma: str):
	# Lógica para obtener la cantidad de películas en el idioma especificada
    return f"{cantidad_peliculas} películas fueron estrenadas en {idioma}"
````
- **Consulta 2**
````
# Se ingresa una pelicula. Debe devolver la duracion y el año.
@app.get('/peliculas_duracion')
def peliculas_duracion(pelicula: str):
	# Lógica para obtener la duración y el año de la película especificada
    return f"Película: {pelicula}. Duración: {duracion} minutos. Estrenada en el Año: {anio}"
````
- **Consulta 3**
````
# Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio
@app.get('/franquicia')
def franquicia(franquicia: str):
	 # Lógica para obtener la cantidad de películas, ganancia total y promedio de la franquicia especificada
    return f"La franquicia {franquicia} posee {cantidad_peliculas} películas, una ganancia total de {ganancia_total} y una ganancia promedio de {ganancia_promedio}"
````
- **Consulta 4**
````
# Se ingresa un país (como están escritos en el dataset, no hay que traducirlos), retornando la cantidad de peliculas producidas en el mismo.
@app.get('/peliculas_pais')
def peliculas_pais(pais: str):
	# Lógica para obtener la cantidad de películas producidas en el país especificado
    return f"Se produjeron {cantidad_peliculas} películas en el país {pais}"
````
- **Consulta 5**
````
# Se ingresa la productora, entregandote el revunue total y la cantidad de peliculas que realizo.
@app.get('/productoras_exitosas')
def productoras_exitosas(productora: str):
	# Lógica para obtener el revenue total y la cantidad de películas de la productora especificada
    return f"La productora {productora} ha tenido un revenue de {revenue_total} y ha realizado {cantidad_peliculas} películas"
````
- **Consulta 6**
````
# Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma, en formato lista.
@app.get('/get_director')
def get_director(nombre_director: str):
	# Lógica para obtener el éxito del director y la información de cada película
    return {
        "exito": suma_retornos,
        "peliculas": peliculas
    }
````
*`encontraremos este proceso en el archivo main.py`*
<br/>
###Deployment
Encontraremos la API desplegada con Render :tw-1f449: [Deploy](https://deploy-render-api.onrender.com/docs)
<br/>
### EDA (Exploratory Data Analysis-EDA)
Ya los datos están limpios, ahora es tiempo de investigar las relaciones que hay entre las variables de los datasets, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente 👀 ), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior. Las nubes de palabras dan una buena idea de cuáles palabras son más frecuentes en los títulos, ¡podría ayudar al sistema de recomendación! En esta ocasión vamos a pedirte que no uses librerías para hacer EDA automático ya que queremos que pongas en practica los conceptos y tareas involucrados en el mismo. 

En esta parte relize distintos tipos de analisis de datos los cuales son:
- Analisis General De La Informacion
- Mapa De Calor
- GráFico De DispersióN
- AnáLisis De Agrupamiento
- Histograma
- Grafico De Barra
- Grafico De Linea
- Diagrama De Caja
- Nubes De Palabras

*`encontraremos este proceso en el archivo EDA.ipynb`*

### ML (Machine learning)
##### Sistema de recomendación:
Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un sistema de recomendación de películas. El EDA debería incluir gráficas interesantes para extraer datos, como por ejemplo una nube de palabras con las palabras más frecuentes en los títulos de las películas. Éste consiste en recomendar películas a los usuarios basándose en películas similares, por lo que se debe encontrar la similitud de puntuación entre esa película y el resto de películas, se ordenarán según el score de similaridad y devolverá una lista de Python con 5 valores, cada uno siendo el string del nombre de las películas con mayor puntaje, en orden descendente. Debe ser deployado como una función adicional de la API anterior y debe llamarse:
````
# Se ingresa el nombre de una película y te recomienda las similares en una lista de 5 valores.
@app.get("/recomendacion")
def recomendacion(titulo: str):
	# Lógica para obtener 5 peliculas recomendadas con mayor similitudes a la pelicula especificada
    return similar_movies
````
*`encontraremos este proceso en el archivo main.py`*

*Con esto finalizamos el Proyecto individual Nº1 de la carrera Data Science Gracias por su atención*
<br/>
