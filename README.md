<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a> 


## ***POLITECNICO MALVINAS ARGENTINAS***

### Materia:   *APRENDIZAJE AUTOMATICO*

### Alumno:   *NICORA, Adrian Julio*

### Profesor:   *MIRABETE, Martin*
------
 
 
# **Análisis Predictivo de Enfermedad Cardiovascular Mediante Técnicas de Clasificación**


------

## **Objetivo:**

El objetivo de este trabajo es desarrollar un modelo de aprendizaje automático capaz de predecir la presencia o ausencia de ***enfermedades cardiovasculares (ECV)*** en individuos, utilizando datos de salud y estilo de vida, con el fin de contribuir a la identificación temprana de personas en riesgo y mejorar las estrategias de prevención en la provincia.

 

## **Contexto:**

Según el IPIEC, las ECV son la segunda causa de muerte en la provincia de Tierra del Fuego para mayores de 18 años, lo que indica una carga significativa para la salud pública y un impacto profundo en la calidad de vida de las personas. Esta alta tasa de mortalidad sugiere que existen desafíos en la prevención y detección temprana, lo que hace de este estudio una herramienta importante para los profesionales de la salud.

El problema se trata de una ***clasificación***, ya que el objetivo es definir si el paciente puede tener o no una ECV en base a los datos de salud del mismo. Dado que es un problema de clasificación binaria, se podrían aplicar los siguientes modelos:

·         Regresión Logística

·         Árboles de Decisión

·         K-Nearest Neighbors (KNN)

·         Support Vector Machine (SVM)

·         Random Forest

Durante el desarrollo se evaluarán los distintos modelos y se seleccionará el o los más óptimos.

## **Estructura del Proyecto**
El proyecto sigue una estructura organizada en carpetas, diseñada para almacenar y procesar los datos de manera eficiente. A continuación se describe la organización principal del repositorio:

- data/: Carpeta principal de datos del proyecto.
    - raw/: Contiene los datos originales.
    - processed/: Contiene los datos preprocesados y limpios de los datasets.

- notebooks/: Contiene notebooks de Jupyter con los análisis y modelos desarrollados.

- docs: Contiene los archivos pdf entregables del proyecto de Machine Learning junto al enlace del video explicativo.


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         enfermedades_ecv and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── enfermedades_ecv   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes enfermedades_ecv a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
