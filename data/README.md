# **Carpeta data**
Esta carpeta contiene todos los archivos de datos utilizados y generados durante el desarrollo del proyecto. El objetivo de esta carpeta es organizar los datos necesarios para los análisis y modelos de predicción.

## **Estructura de la Carpeta**
La carpeta data está organizada en subdirectorios para facilitar el acceso y organización de los datos:

- raw/: Contiene los datos en su formato original, tal como fueron obtenidos.

    - Cardiovascular_Disease_Dataset.csv: Archivo CSV con registros de salud de pacientes.

- processed/: Contiene los datos preprocesados y limpios para su posterior analisis y modelado.

    - cardiovascular_disease_dataset.csv: Archivo CSV limpio para el posterior uso en los modelos de machine learning.

## **Acceso al Dataset**
El dataset fue descargado de Doppala, Bhanu Prakash; Bhattacharyya, Debnath (2021), “Cardiovascular_Disease_Dataset”, Mendeley Data, V1, doi: 10.17632/dzz48mvjht.1

**URL de referencia:** https://data.mendeley.com/datasets/dzz48mvjht/1
 
## **Descripcion a los archivos del dataset**
El dataset consta de 1000 registros y 14 columnas, y esta compuesto por los siguientes datos:

|  | Columna | Descripción | Unidades | Tipo de Dato |
| ----- | :---- | :---- | :---- | :---- |
| 1 | patientid | Numero de identificación del paciente | Number | Numérico |
| 2 | age | Edad | In Years | Numérico |
| 3 | gender | Género | 1,0(0= female, 1 \= male) | Binario |
| 4 | chestpain | Nivel de dolor de pecho | 0,1,2,3 (Value 0: typical angina Value 1: atypical angina Value 2: non-anginal pain Value 3: asymptomatic) | Nominal |
| 5 | restingBP | Presión sanguínea en reposo | 94-200 (in mm HG) | Numérico |
| 6 | serumcholestrol | Colesterol en sangre | 126-564 ( in mg/dl) | Numérico |
| 7 | fastingbloodsugar | Glucosa en sangre en ayunas | 0,1 \> 120 mg/dl (0 \= false , 1 \= true) | Binario |
| 8 | restingrelectro | Resultado del ECG en reposo | 0,1,2 (Value 0: normal, Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of \> 0.05 mV), Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria) | Nominal |
| 9 | maxheartrate | Máximas pulsaciones del corazón | 71-202 | Numérico |
| 10 | exerciseangia | Angina de pecho inducida por ejercicio | 0,1 (0 \= no, 1 \= yes) | Binario |
| 11 | oldpeak | Depresión del segmento ST inducida por el ejercicio | 0-6.2 | Numérico |
| 12 | slope | Pendiente del segmento ST | 1,2,3 (1-upsloping, 2-flat, 3- downsloping) | Nominal |
| 13 | noofmajorvessels | Número de vasos mayores | 0,1,2,3 | Numérico |
| 14 | target | Clasificación (tiene o no enfermedad cardiovascular) | 0,1 (0= Absence of Heart Disease, 1= Presence of Heart Disease) | Binario |

-----