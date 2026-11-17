# Asistente de Solicitudes – Área de Administración de Datos

Asistente inteligente que orienta a los usuarios sobre las solicitudes que levantan al Área de Administración de Datos mediante Google Forms y que se registran como *work items* en Azure DevOps Boards.

| Etapa | Contenido |
| --- | --- |
| Avance 1 | Chatbot basado en reglas (Python + NLTK) |
| **Avance 2 (actual)** | Clasificador de intenciones con PLN (TF-IDF + SVM), clasificador de capturas de pantalla con CNN (EfficientNet-B0, transferencia de aprendizaje) y asistente multimodal texto + imagen |
| Siguiente | Integración con la API de Azure DevOps y despliegue web (Databricks App) |

## Problema

Al enviar el formulario, el usuario no recibe el ID de su solicitud ni puede consultar su avance, los comentarios del analista o la resolución. Esto genera consultas manuales repetitivas y solicitudes mal llenadas.

## Funcionalidades

- Consulta de **estatus**, **comentarios** y **resolución** de una solicitud por ID.
- **Recuperación de IDs** a partir del correo corporativo.
- Explicación de los **estados** del work item (New, Active, Blocked, Resolved, Closed).
- Guía de **llenado del formulario** y **campos obligatorios**.
- Orientación para solicitudes de **ETL**, **sábanas de datos (CSV/XLSX)** e **incidencias**.
- Información de **prioridades**, **tiempos de atención** y **escalamiento** a una persona.
- Manejo de contexto: recuerda el último ID consultado y completa consultas pendientes cuando el usuario envía solo el número.
- Tolerancia a errores de escritura (distancia de edición ≤ 1) y respuesta de *fallback* con escalamiento.

## Estructura del repositorio

```
.
├── chatbot_reglas.py            # v1: chatbot por reglas (gestor de diálogo y respuestas reutilizados en la v2)
├── pruebas_conversaciones.py    # v1: 10 conversaciones de prueba (26 turnos)
├── preprocesamiento.py          # v2: normalización, tokenización, stopwords y stemming (NLTK)
├── dataset_intenciones.py       # v2: dataset sintético de intenciones + prueba retadora escrita a mano
├── clasificador_intenciones.py  # v2: BoW/TF-IDF + Naive Bayes/SVM, evaluación y ablación
├── dataset_imagenes.py          # v2: generador de capturas de pantalla sintéticas (5 clases)
├── clasificador_imagenes.py     # v2: EfficientNet-B0 / ResNet50 con transferencia de aprendizaje
├── base_conocimiento.py         # v2: 300 tickets ficticios adicionales
├── asistente_multimodal.py      # v2: asistente que integra texto + imagen
├── pruebas_multimodales.py      # v2: 10 conversaciones multimodales de prueba
├── ejecutar_avance2             # Notebook de Databricks que ejecuta todo el proceso
├── requirements.txt
├── README.md
└── salidas/                     # Se genera al ejecutar (modelos, métricas, matrices, imágenes)
```

## Requisitos

- **Python 3.10, 3.11 o 3.12** (probado con 3.12.3). Verifica con `python --version`.
- **pip** actualizado.
- **Conexión a internet en la primera ejecución** para descargar: las librerías, los recursos de NLTK (`stopwords`, `punkt_tab`) y los pesos preentrenados de EfficientNet-B0 (unos 20 MB).
- **CPU es suficiente**; no se requiere GPU. Espacio libre aproximado: 1.5 GB (librerías) + 100 MB (salidas).

Versiones probadas: torch 2.15.0+cpu, torchvision 0.29.2+cpu, NLTK 3.10.4, scikit-learn 1.7.2, Pillow 12.0.0, matplotlib 3.10.6.

## Instalación en otro equipo

### 1. Obtener el código

```bash
git clone https://github.com/jraa22/asistente_solicitudes.git
cd asistente_solicitudes
```

Todos los comandos siguientes se ejecutan **desde la carpeta del repositorio**, porque los módulos se importan entre sí.

### 2. Crear y activar un entorno virtual

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea el script, ejecuta antes `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Instalar dependencias

Versión **CPU** de PyTorch (recomendada, más ligera):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

En macOS omite `--extra-index-url` (la versión estándar ya es para CPU/Apple Silicon). Si tienes GPU NVIDIA y quieres usarla, instala torch siguiendo https://pytorch.org/get-started/locally/; el código funciona igual en CPU.

### 4. Descargar los recursos de NLTK

`preprocesamiento.py` los descarga automáticamente la primera vez. Si el equipo no tiene internet o hay un proxy, descárgalos manualmente:

```bash
python -m nltk.downloader stopwords punkt_tab
```

Sin estos recursos el código sigue funcionando con un tokenizador y una lista de stopwords de respaldo, pero los resultados pueden variar ligeramente.

### 5. (Opcional) Cambiar la carpeta de salidas

Por defecto todo se guarda en `./salidas`. Para usar otra carpeta define la variable `ASISTENTE_SALIDAS`:

```bash
export ASISTENTE_SALIDAS=/ruta/a/salidas        # Linux / macOS
```

```powershell
$env:ASISTENTE_SALIDAS = "C:\ruta\a\salidas"    # Windows PowerShell
```

### 6. Verificar la instalación

```bash
python -c "import nltk, sklearn, torch, torchvision, PIL; print('torch', torch.__version__, '| nltk', nltk.__version__)"
python preprocesamiento.py
```

La segunda instrucción debe imprimir cada paso del preprocesamiento (normalizado, tokens, sin stopwords y stems).

## Ejecución (en este orden)

| Paso | Comando | Qué hace | Tiempo aprox. (CPU) |
| --- | --- | --- | --- |
| 1 | `python pruebas_conversaciones.py` | Pruebas del chatbot de reglas v1 (26/26 esperado) | < 5 s |
| 2 | `python clasificador_intenciones.py` | Genera el dataset textual, compara NB/SVM, guarda `modelo_intenciones.joblib` y ejecuta la ablación | ~1 min |
| 3 | `python dataset_imagenes.py` | Genera 950 capturas en `salidas/imagenes/{train,val,test,ood}` | ~30 s |
| 4 | `python clasificador_imagenes.py` | Entrena EfficientNet-B0 (5 épocas cabeza + 3 de ajuste fino) y guarda `modelo_imagenes.pt` | 7-10 min |
| 5 | `python pruebas_multimodales.py` | 10 conversaciones texto + imagen (14/14 y 8/8 esperado) | ~10 s |
| 6 | `python asistente_multimodal.py` | Asistente interactivo | - |

Los pasos 5 y 6 requieren que existan los modelos de los pasos 2 y 4. Si se omite el paso 3, el paso 4 genera las imágenes automáticamente.

Para entrenar con ResNet50 en lugar de EfficientNet-B0:

```bash
python clasificador_imagenes.py --arquitectura resnet50 --epocas-f1 5 --epocas-f2 3
```

### Archivos generados en `salidas/`

| Archivo | Contenido |
| --- | --- |
| `modelo_intenciones.joblib` | Clasificador de intenciones calibrado |
| `metricas_intenciones.json`, `prueba_retadora_resultados.csv` | Métricas y predicciones de texto |
| `matriz_confusion_intenciones.png` / `.csv` | Matriz de confusión de intenciones |
| `modelo_imagenes.pt` | Pesos de la CNN y nombres de clases |
| `metricas_imagenes.json` | Historia de entrenamiento y métricas de test / ood |
| `matriz_confusion_imagenes_test.png`, `matriz_confusion_imagenes_ood.png` | Matrices de confusión visuales |
| `imagenes/` | Dataset de capturas sintéticas |

La carpeta `salidas/` no se sube a GitHub (está en `.gitignore`); se regenera con los comandos anteriores.

### Ejecución en Databricks

Abre el notebook `ejecutar_avance2` en la carpeta del proyecto y ejecuta todas las celdas en orden. La primera celda instala las dependencias con `%pip` y la segunda reinicia Python; repítelas en cada sesión nueva. Los archivos `.py` no tienen panel de dependencias, por eso se ejecutan desde el notebook.

## Uso

### Chatbot de reglas v1

Modo interactivo (escribe `salir` para terminar):

```bash
python chatbot_reglas.py
```

Ejemplo:

```
Tu: Como va la 48213?
Bot: Solicitud #48213 - ETL de tabla ventas_diarias hacia el Lakehouse
  Tipo: ETL / Ingesta
  Estado: Active (En atencion: un analista ya trabaja en ella.)
  Asignada a: Equipo de Ingesta
  Alta: 2026-10-02 | Ultima actualizacion: 2026-10-21
Tu: Y que comentarios tiene?
Bot: Comentarios de la solicitud #48213:
  2026-10-05 - Analista: Se valido el acceso a la base origen.
  2026-10-21 - Analista: Pipeline en pruebas en ambiente QA.
```

### Asistente multimodal v2

```bash
python asistente_multimodal.py
```

Para adjuntar una imagen escribe el mensaje seguido de `img:` y la ruta (sin espacios en la ruta):

```
Tu: El proceso de cartera fallo en la noche img:salidas/imagenes/test/mensaje_error/img_00140.png
Bot: Registra la incidencia con el tipo 'Incidencia' e incluye: ...
Revise tu imagen: parece un mensaje de error o traza de excepcion (confianza 99%). Te sugiero registrarla
como 'Incidencia tecnologica' con prioridad Alta. ...
```

También puedes enviar solo la imagen (`img:ruta.png`) o capturas propias en PNG/JPG.

## Resultados de referencia

Si tu ejecución es correcta, deberías obtener valores iguales o muy cercanos (pueden variar levemente según el sistema operativo, las fuentes instaladas o la versión de las librerías):

| Componente | Métrica | Valor |
| --- | --- | --- |
| Reglas v1 | Pruebas funcionales | 26/26 |
| Intenciones (SVM TF-IDF palabras + caracteres) | CV aleatoria F1-macro / CV por plantilla F1-macro | 0.993 / 0.497 |
| Intenciones | Prueba retadora (66 frases): ML vs reglas | 0.833 vs 0.515 |
| Imágenes (EfficientNet-B0) | Test: accuracy / F1-macro | 1.000 / 1.000 |
| Imágenes | OOD (cambio de distribución): accuracy / F1-macro | 0.847 / 0.850 |
| Multimodal | Pruebas: texto / imagen | 14/14 / 8/8 |

## Solución de problemas

| Error | Causa y solución |
| --- | --- |
| `ModuleNotFoundError: No module named 'torch'` (o `nltk`, `sklearn`) | El entorno virtual no está activo o faltan dependencias: activa `.venv` y repite el paso 3. En Databricks, ejecuta primero las celdas 1 y 2 del notebook. |
| `ModuleNotFoundError: No module named 'chatbot_reglas'` (u otro módulo del proyecto) | Ejecuta los comandos desde la carpeta del repositorio (`cd asistente_solicitudes`). |
| `LookupError: Resource punkt_tab not found` | Ejecuta `python -m nltk.downloader stopwords punkt_tab`. |
| `FileNotFoundError: ... modelo_intenciones.joblib` o `modelo_imagenes.pt` | Falta entrenar: ejecuta los pasos 2 y 4. |
| Error SSL / sin conexión al descargar `efficientnet_b0_rwightman-7f5810bc.pth` | Descarga el archivo en otro equipo desde `https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth` y cópialo a `~/.cache/torch/hub/checkpoints/` (Windows: `C:\Users\<USUARIO>\.cache\torch\hub\checkpoints\`). |
| Advertencias o errores al cargar `modelo_intenciones.joblib` en otra versión de scikit-learn | Los modelos guardados dependen de la versión; vuelve a ejecutar el paso 2 en el nuevo equipo. |
| El entrenamiento de imágenes es muy lento | Es normal en CPU (7-10 min). Puedes reducir épocas: `python clasificador_imagenes.py --epocas-f1 3 --epocas-f2 1`. |
| Las imágenes generadas se ven con letra pequeña o distinta | El generador busca DejaVuSans o Arial; si no existen usa la fuente por defecto de Pillow. Instala DejaVu (`sudo apt install fonts-dejavu`) para resultados idénticos. |
| Caracteres raros con acentos en la consola de Windows | Ejecuta `chcp 65001` antes o usa Windows Terminal. |

## Cómo funciona

### Chatbot de reglas (v1)

1. **Preprocesamiento (NLTK):** normalización (minúsculas, sin acentos), `RegexpTokenizer`, eliminación de stopwords y `SnowballStemmer("spanish")`.
2. **Extracción de entidades:** ID de solicitud y correo mediante expresiones regulares.
3. **Clasificación por reglas:** cada intención suma puntos (frase exacta = 2, palabra clave = 1, coincidencia aproximada con `edit_distance` = 0.5). Las intenciones sociales tienen peso 0.9 para no ganar sobre las de negocio. Umbral mínimo: 0.5.
4. **Gestor de diálogo:** memoria del último ID, intención pendiente y *fallback* con escalamiento al segundo fallo.
5. **Respuesta:** plantillas alimentadas por la base de conocimiento.

### Asistente multimodal (v2)

1. **Texto:** normalización, `word_tokenize`, stopwords en español (conservando negaciones e interrogativos) y `SnowballStemmer`. Representación TF-IDF de palabras (1-2 gramas) + n-gramas de caracteres (2-5) y clasificación con SVM lineal calibrado. Si la confianza es menor a 0.30 se usa el *fallback*.
2. **Imagen:** EfficientNet-B0 preentrenada en ImageNet; fase 1 entrena solo la cabeza y fase 2 ajusta los últimos 3 bloques. Clases: `mensaje_error`, `tablero_graficas`, `tabla_datos`, `editor_sql`, `formulario`.
3. **Fusión:** la clase visual se traduce en tipo de solicitud, prioridad sugerida e información a incluir; se detecta incoherencia entre texto e imagen, se asocia la captura al ID en conversación y se arma un borrador de solicitud. Si la confianza visual es menor a 60 % se pide confirmación.

### Agregar una intención (v1)

Añade un objeto `Intencion` a la lista `INTENCIONES` en `chatbot_reglas.py`:

```python
Intencion("accesos", ["acceso", "permiso", "permisos"], ["dar acceso"],
          ["Para solicitar accesos indica esquema/tabla, tipo de permiso y data owner que autoriza."])
```

Para la v2, agrega plantillas de frases en `PLANTILLAS` de `dataset_intenciones.py` y vuelve a ejecutar el paso 2.

## Limitaciones conocidas

- Los datos de entrenamiento de texto e imágenes son **sintéticos**; el desempeño con mensajes y capturas reales puede ser menor.
- El clasificador de intenciones generaliza de forma limitada a estructuras de frase no vistas (F1 0.497 por plantilla) y detecta mal los mensajes fuera de dominio.
- La CNN confunde capturas oscuras o de baja calidad con mensajes de error (OOD accuracy 0.847).
- La base de conocimiento de work items es **simulada** (los IDs 48213, 48250, 48301, 48120, 48333 y 49000-49299 son ficticios).
- Sin autenticación de usuario; interfaz solo de consola.

## Próximas etapas

- Reentrenar con mensajes y capturas reales anonimizadas; evaluar *embeddings* / LLM.
- Integración con la API REST de Azure DevOps y confirmación del ID al enviar el formulario.
- Despliegue en la Databricks App existente con autenticación corporativa.

## Autor

Jesus Roberto Ahumada Avendaño – Maestria en Inteligencia Artificial – 2026
