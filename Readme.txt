Proyecto MVP de recomendacion de juegos de Steam deployado en FastAPI.

Cada carpeta sigue los pasos en que se trabajo el Proyecto.

En la carpeta 1 se realizo la carga de los datos y transformacion para eliminar filas con valores nulos NaNs y otros en donde no son permitidos en json.
Luego se transform en parquet para mejorar el tamano de los archivos. Especialmente para ML donde las funciones como explode de usuarios items genera tamanos de mas de 1GB. Adicionalmente se agrego la columna de sentimientos para cada review.

En carpeta 2 se realizo el trabajo de entrenamiento usando similtud de cosenos. Como explode generaba problemas se opto por recomendar juegos que sean mas de 20$ en precio y para usuarios que tengan mas de 200 juegos en su libreria.

En carpeta 3 se hizo todos los pasos para deployar en render para esto use una estructura adecuada de archivos separacion de carpeta de datos y de la app, agregacion de requirements, optimizacion de codigo para cachear los archivos y mejorar la velocidad de respuesta.

Video de youtube de demostracion de Proyecto: https://youtu.be/DzayDc2FP2s
Repositorio de desarollo de Proyecto: https://github.com/ArtiomDiakov/ProjectMLOps.git
Repositorio de deployment en Render: https://github.com/ArtiomDiakov/API_RECOMMENDER.git

Nota: No se agregaron archivos JSON por problemas de subida a repositorio por sus tamanos exorbitantes.
