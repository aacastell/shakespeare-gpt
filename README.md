# Decisiones de Arquitectura - shakespeare-gpt - MicroProyecto2

## Objetivo

Entrenar y comparar:

1. Un modelo de lenguaje estilo GPT entrenado desde cero.
2. Un modelo de lenguaje GPT-2 con ajuste fino.

SObre un corpus de trabajos de William Shakespeare.

## Restricciones 

- Tamaño de vocabulario fijo en 50,527 (tokenizador de GPT-2)
- Mismo conjunto de datos y tokenizacion para ambos modelos
- Uso Exclusivo de PyTorch
- Generacion de texto implementada desde cero

## Principios de Diseño

- **Comparabilidad:** Ambos modelos deben operar bajo el mismo esquema de tokenizacion y las mismas particiones del dataset para asegurar una comparacion justa.

- **Claridad Conceptual:** El proyecto prioriza la claridad conceptual y la interpretabilidad sobre maximizar el tamaño o desempeño de el modelo.

- **Reproducibilidad:** Todos los pasos de preprocesamiento, particion de datos y entrenamiento estan completamente definidos.

## Tokenizacion 

- **Tokenizador:** Tokenizador GPT-2 basado de en Byte Pair Encoding (BPE) ('GPT2Tokenizar' de Hugging Face)
- **Tamaño del Vocabulario:** Fijado en **50257 tokens**

### Justificacion

- El enunciado exige explicitamente que ambos modelos utilicen el mismo tamaño de vocabulario que GPT-2.
- Usar el tokenizador oficial de GPT-2 garantiza compatibilidad directa con el model pre-entrenado.
- No es posible utilizar otro tokenizador o un tokenizador personalizado, ya que esto romperia la equivalencia lexica, y requeririra pre entranar el modelo GPT-2, so solo ajustarlo finamente. 

---

## Datos

### Particion de Datos

- El corpus se divide de manera **secuencial** en:
    - Entrenamiento: 90-95%
    - Validacion: 5-10%

### Justificación

- Una particion secuencial evita filtraciones de informacion futura hacia el conjunto de validacion.
- Esta estrategia conserva la estructura temporal natural del texto literario.

## Muestras

- El corpus tokenizado se segmenta en secuencias de longitud fija.
- Para cada secuencia, la entrada corresponde a los tokens 't1 ... tn-1' y el objetivo a los tokens "t2 ... tn".

---

## Modelo 1: GPT Entrenado desde Cero

### Tipo de Arquitectura
- Transformer *decoder-only* (estilo GPT)

### Componentes Principales
- Capa de embeddings te tokens
- embeddings posicionales aprendidos
- Pila de bloques Transformer, cada uno compuesto por:
    - Atencion causal (*causal self-attention*)
    - Red feed-forward
    - Conexiones residuales
    - Normalizacion por capas
- Capa lineal final que proyecta al tamaño del vocabulario (50 257)

### Justificacion
- Esta escala es suficiente para demostrar el aprendizaje de estructura lingüistica basica y, al mismo tiempo, es computacionalmente apropiada.
- El objetivo de este modelo es comparativo, no competir con GPT-2.

---

## Modelo 2: Fine-Tuning de GPT-2 

### Base
- GPT-2 Pequeño (124M de parametros) con pesos pre-entrenados

### Estrategia de Fine-Tuning
- Fine-Tuning Completo sin congelar capas
- Uso del mismo dataset tokenizado y las mismas particiones que el modelo entrenado desde cero

### Restricciones de Entrenamiento

- Longitud de el contexto y *batch size* son limitadas por la memoria de el procesador utilizado. 
- Un numero conservador de epocas para mitigar sobre ajuste.

### Justificacion

- Ajusta todas las capas permite mejor adaptacion al estilo lingüistico  
-

---

## Funcion de Perdida y Metricas de Evaluacion

- **Perdida durante entrenamiento:** Perdida de entropia cruzada (*cross-entropy*) sobre la prediccion del siguiente token
- **Metrica de Evaluacion:** Valor minimo de la perdida en el conjunto de validacion alcanzado durante el entrenamiento.

Se utiliza exactamante la misma funcion de perdida y procedimiento de evaluacion para ambos modelos.

---

## Generacion de Texto

### Metodo de Generacion

- Generacion autoregresiva token a token implementada desde cero
- No se emplean funciones de alto nivel como 'model.generate()'

### Estrategia de Muestreo

- Muestreo con temperatura
- Restriccion mediante top-k

### COndiciones de Comparacion
- Mismo "prompt" inicial para ambos modelos
- Misma longitud de texto generado
- Mismo metodo de muestreo

### Justificacion
- Utilizar una unica funcion de generacion garantiza una comparacion cualitativa justa.

---

## Consideraciones sobre el sobreajuste

La capacidad de generalizacion de los modelos se evalua mediante:
- Comparacion entre las curvas de perdida de entranamiento y validacion
- Limitacion del numero de epocas de fine-tuning
- Uso de un conjunto de validacion no visto durante el entrenamiento

---

## Decisiones Congeladas

Las siguientes decisiones se consideran finales para esta version del proyecto:

- Uso del tokenizador GPT-2 con vocabulario de 50 257 tokens
- Arquitecturas Transformer *decoder-only*
- Particion Secuencial del dataset
- Uso de una unica funcion de generacion compartida
- Fine-Tuning completo del modelo GPT-2 pequeño

## Autoria del Proyecto
Esta implementacion es realizada por el equipo de trabajo 4, de el curso de Modelos Avanzados Para el Procesamiento de Lenguaje Natural, de la Universidad de los Andes, Bogota, Abril 2026.

