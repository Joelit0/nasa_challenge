# Sistema de Clasificación de Exoplanetas

Aplicación web de clasificación de exoplanetas desarrollada con Streamlit. Utiliza machine learning para identificar exoplanetas basándose en datos astronómicos de las misiones Kepler y TESS de NASA.

## Descripcion

Este proyecto implementa un modelo de aprendizaje automático para clasificar objetos astronómicos como exoplanetas confirmados, candidatos o falsos positivos. El sistema fue entrenado con datos validados del Archivo de Exoplanetas de NASA.

[Demo](media/demo.mp4)

## Caracteristicas

- Clasificacion automatica de exoplanetas mediante modelo XGBoost
- Visualizaciones interactivas con Plotly
- Filtrado y analisis de resultados
- Descarga de resultados en formato CSV
- Interfaz de usuario intuitiva con Streamlit
- Contenedorizacion completa con Docker

## Requisitos

- Python 3.11 o superior
- Docker y Docker Compose (opcional, para ejecutar con contenedores)
- Homebrew (para macOS, instalacion de OpenMP)

## Instalacion

### Opcion 1: Entorno Virtual (Desarrollo Local)

1. Clonar el repositorio:
```bash
git clone git@github.com:Joelit0/nasa_challenge.git
cd nasa_challenge
```

2. Crear entorno virtual:
```bash
python3 -m venv venv
```

3. Activar entorno virtual:
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. Instalar dependencias:
```bash
pip install -r requirements.txt
```

5. Instalar OpenMP (solo macOS):
```bash
brew install libomp
```

### Opcion 2: Docker (Recomendado para Produccion)

1. Asegurarse de tener Docker y Docker Compose instalados

2. Construir y ejecutar contenedor:
```bash
docker-compose up --build
```

## Uso

### Ejecutar con Entorno Virtual

1. Activar entorno virtual:
```bash
source venv/bin/activate
```

2. Ejecutar aplicacion:
```bash
streamlit run app.py
```

3. Abrir navegador en:
```
http://localhost:8501
```

### Ejecutar con Docker

1. Construir contenedor:
```bash
docker-compose up --build
```

2. Abrir navegador en:
```
http://localhost:8501
```

3. Detener aplicacion:
```bash
docker-compose down
```

## Estructura del Proyecto

```
nasa_challenge/
├── app.py                      # Aplicacion Streamlit principal
├── requirements.txt            # Dependencias Python
├── Dockerfile                  # Configuracion Docker
├── docker-compose.yml          # Configuracion Docker Compose
├── .gitignore                  # Archivos ignorados por Git
├── .dockerignore              # Archivos ignorados por Docker
├── README.md                   # Este archivo
├── exoplanet_pipeline.pkl    # Modelo entrenado (pipeline)
├── feature_names.pkl          # Nombres de caracteristicas del modelo
└── tests/                      # Tests unitarios
    └── test_data_generator.py
```

## Componentes Principales

### Modelo Machine Learning

- Algoritmo: XGBoost Classifier
- Clases: False Positive, Candidate, Confirmed Exoplanet
- Precision: Trainig accuracy del 76.5%
- Caracteristicas: 8 features astronomicas

### Caracteristicas Requeridas

Las siguientes caracteristicas son necesarias en el archivo CSV:

1. planet_radius - Radio del planeta en radios terrestres
2. transit_depth - Profundidad del transito en ppm
3. transit_duration - Duracion del transito en horas
4. orbital_period - Periodo orbital en dias
5. stellar_temp - Temperatura de la estrella en Kelvin
6. stellar_radius - Radio de la estrella en radios solares
7. stellar_logg - Logaritmo de la gravedad superficial estelar
8. eq_temperature - Temperatura de equilibrio del planeta en Kelvin

## Formato de Datos de Entrada

El CSV de entrada debe incluir las 8 caracteristicas mencionadas. Ejemplo de estructura:

```csv
planet_radius,transit_depth,transit_duration,orbital_period,stellar_temp,stellar_radius,stellar_logg,eq_temperature
2.5,850,3.2,245.6,5800,1.2,4.3,280
...
```

## Uso de la Aplicacion

1. Ir a la pestaña "Data Upload & Prediction"
2. Seleccionar archivo CSV con datos de exoplanetas
3. Hacer clic en "Run Classification"
4. Visualizar resultados y filtros
5. Descargar resultados como CSV

## Pestañas de la Aplicacion

- Data Upload & Prediction: Carga de datos y clasificacion
- Feature Documentation: Documentacion de caracteristicas
- Model Information: Informacion del modelo y su rendimiento
- About: Informacion general del sistema

## Tecnologias Utilizadas

- Python 3.11
- Streamlit 1.50.0
- XGBoost 3.1.1
- Scikit-learn 1.7.2
- Pandas 2.3.3
- NumPy 2.3.4
- Plotly 6.3.1
- Matplotlib 3.10.7
- Docker y Docker Compose

## Dependencias del Sistema

### macOS

- OpenMP runtime (libomp): Requerido para XGBoost
```bash
brew install libomp
```

### Linux (Docker)

- OpenMP incluye en el contenedor Docker
- No requiere instalacion adicional

## Comandos Utiles

### Docker

Verificar estado de contenedores:
```bash
docker ps
```

Ver logs:
```bash
docker-compose logs -f
```

Detener servicios:
```bash
docker-compose down
```

Reconstruir sin cache:
```bash
docker-compose up --build --no-cache
```

### Desarrollo

Activar entorno virtual:
```bash
source venv/bin/activate
```

Instalar nuevas dependencias:
```bash
pip install <paquete>
pip freeze > requirements.txt
```

## Solucion de Problemas

### Error: Ports are not available

Si el puerto 8501 esta en uso, detener procesos de Streamlit:
```bash
pkill -f streamlit
```

### Error: XGBoost library could not be loaded (macOS)

Instalar OpenMP:
```bash
brew install libomp
```

### Error: Missing features en CSV

Verificar que el archivo CSV contenga todas las caracteristicas requeridas. La aplicacion mostrara las columnas disponibles y las requeridas.

## Licencia

Este proyecto fue desarrollado como parte del NASA Challenge. Ver documentacion oficial para detalles de licencia.

## Fuentes de Datos

- KOI (Kepler Objects of Interest): Datos del telescopio espacial Kepler
- TOI (TESS Objects of Interest): Datos del satelite TESS
- NASA Exoplanet Archive: Archivo de exoplanetas confirmados

## Contacto

Para preguntas o problemas, consultar la documentacion del proyecto o contactar al equipo de desarrollo.

