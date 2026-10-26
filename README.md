# Asistente de Solicitudes – Área de Administración de Datos

Chatbot basado en reglas (Python + NLTK) que orienta a los usuarios sobre las solicitudes que levantan al Área de Administración de Datos mediante Google Forms y que se registran como *work items* en Azure DevOps Boards.

> **Etapa 1 (avance actual):** chatbot funcional basado en reglas con base de conocimiento simulada. Las siguientes etapas incorporarán PLN, clasificación de imágenes e integración web (Databricks App).

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
├── chatbot_reglas.py          # Chatbot: preprocesamiento, base de conocimiento, intenciones y motor de diálogo
├── pruebas_conversaciones.py  # 10 conversaciones de prueba con intención esperada por turno
├── requirements.txt           # Dependencias
└── README.md
```

## Requisitos

- Python 3.10 o superior
- NLTK 3.8 o superior (no requiere descargar corpus)

```bash
pip install -r requirements.txt   # o: pip install nltk
```

`requirements.txt`:

```
nltk>=3.8
```

## Uso

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

Pruebas funcionales:

```bash
python pruebas_conversaciones.py
```

Resultado de la última ejecución (Python 3.12.3, NLTK 3.10.4): **26/26 turnos correctos (100 %)**.

## Cómo funciona

1. **Preprocesamiento (NLTK):** normalización (minúsculas, sin acentos), `RegexpTokenizer`, eliminación de stopwords y `SnowballStemmer("spanish")`.
2. **Extracción de entidades:** ID de solicitud y correo mediante expresiones regulares.
3. **Clasificación por reglas:** cada intención suma puntos (frase exacta = 2, palabra clave = 1, coincidencia aproximada con `edit_distance` = 0.5). Las intenciones sociales tienen peso 0.9 para no ganar sobre las de negocio. Umbral mínimo: 0.5.
4. **Gestor de diálogo:** memoria del último ID, intención pendiente y *fallback* con escalamiento al segundo fallo.
5. **Respuesta:** plantillas alimentadas por la base de conocimiento.

### Agregar una intención

Añade un objeto `Intencion` a la lista `INTENCIONES` en `chatbot_reglas.py`:

```python
Intencion("accesos", ["acceso", "permiso", "permisos"], ["dar acceso"],
          ["Para solicitar accesos indica esquema/tabla, tipo de permiso y data owner que autoriza."])
```

## Limitaciones conocidas

- Depende de palabras clave: no reconoce sinónimos ni paráfrasis.
- No maneja negación ni múltiples intenciones en un mismo mensaje.
- Los empates se resuelven por el orden de las intenciones.
- La base de conocimiento de work items es **simulada** (los IDs 48213, 48250, 48301, 48120 y 48333 son ficticios).
- Sin autenticación de usuario; interfaz solo de consola.

## Próximas etapas

- Clasificador de intenciones con PLN (TF-IDF / embeddings / LLM).
- Integración con la API REST de Azure DevOps y confirmación del ID al enviar el formulario.
- Clasificación de imágenes para capturas de incidencias.
- Despliegue en la Databricks App existente con autenticación corporativa.

## Autor

Jesus Roberto Ahumada Avendaño – Maestria en Inteligencia Artificial – 2026
