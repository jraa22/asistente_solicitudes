"""
Construccion del conjunto de datos de intenciones conversacionales - Avance 2.

El dataset se genera de forma sintetica y reproducible a partir de:
  * plantillas por intencion (distintas formas de expresar una misma necesidad),
  * ranuras con sinonimos y entidades (tipo de solicitud, IDs, tablas, origenes, estados, correos),
  * aumentacion: prefijos/sufijos coloquiales, perdida de acentos y errores de dedo.
Cada ejemplo conserva el indice de su plantilla ('grupo') para evaluar con
validacion por grupos (frases cuya estructura no se vio en entrenamiento).

Tambien incluye PRUEBA_RETADORA: frases escritas a mano, con redaccion distinta
a las plantillas, para medir la generalizacion real del modelo.

Uso:  python dataset_intenciones.py   -> salidas/intenciones.csv
"""
from __future__ import annotations

import itertools
import os
import random
import string
import unicodedata
from pathlib import Path

import pandas as pd

SALIDAS = Path(os.environ.get("ASISTENTE_SALIDAS", Path(__file__).resolve().parent / "salidas"))

RANURAS: dict[str, list[str]] = {
    "sol": ["solicitud", "ticket", "requerimiento", "petición", "caso", "work item", "folio"],
    "sols": ["solicitudes", "tickets", "requerimientos", "peticiones", "casos"],
    "ref": ["", " NID", " #NID", " número NID", " con id NID", " NID"],
    "estado": ["New", "Active", "Blocked", "Resolved", "Closed", "en progreso", "en validación",
               "Done", "bloqueado"],
    "tabla": ["ventas", "inventarios", "clientes", "cartera", "proveedores", "cobranza", "nómina",
              "créditos", "pedidos", "sucursales"],
    "origen": ["SQL Server", "Oracle", "SAP", "PostgreSQL", "un SFTP", "archivos de Excel",
               "Salesforce", "Cosmos DB", "una API"],
    "email": ["maria.lopez@empresa.com", "juan.perez@empresa.com", "ana.ruiz@empresa.com",
              "luis.garcia@empresa.com", "carla.mendez@empresa.com"],
}

PLANTILLAS: dict[str, list[str]] = {
    "estatus": [
        "¿cómo va mi {sol}{ref}?", "¿cuál es el estatus de mi {sol}{ref}?",
        "¿en qué estado está mi {sol}{ref}?", "¿ya atendieron mi {sol}{ref}?",
        "quiero saber el avance de mi {sol}{ref}", "me das seguimiento a mi {sol}{ref}",
        "¿ya está lista mi {sol}{ref}?", "¿alguien ya está revisando mi {sol}{ref}?",
        "¿por qué sigue detenida mi {sol}{ref}?", "¿en qué va lo que pedí{ref}?",
        "status{ref}", "¿sigue pendiente mi {sol}{ref}?", "¿ya la asignaron a alguien{ref}?",
        "llevo días esperando, ¿qué pasó con mi {sol}{ref}?",
    ],
    "comentarios": [
        "¿qué comentarios tiene mi {sol}{ref}?", "¿qué me respondió el analista{ref}?",
        "¿hay notas en mi {sol}{ref}?", "¿qué observaciones dejaron en mi {sol}{ref}?",
        "quiero ver los comentarios de mi {sol}{ref}", "¿el analista dejó algún mensaje{ref}?",
        "¿qué dijeron sobre mi {sol}{ref}?", "¿me contestaron algo en mi {sol}{ref}?",
        "no quiero el estatus, quiero los comentarios de mi {sol}{ref}",
        "muéstrame el historial de comentarios{ref}", "¿qué escribió el equipo en mi {sol}{ref}?",
    ],
    "resolucion": [
        "¿cómo se resolvió mi {sol}{ref}?", "¿cuál fue la solución de mi {sol}{ref}?",
        "¿dónde dejaron mi entregable{ref}?", "ya cerraron mi {sol}{ref}, ¿qué hicieron?",
        "quiero ver la resolución de mi {sol}{ref}", "¿dónde está el archivo que me entregaron{ref}?",
        "¿cómo quedó mi {sol}{ref}?", "¿cuál fue la respuesta final de mi {sol}{ref}?",
        "¿qué resolvieron en mi {sol}{ref}?", "¿dónde descargo el resultado de mi {sol}{ref}?",
    ],
    "estados_significado": [
        "¿qué significa el estado {estado}?", "¿qué quiere decir {estado}?",
        "mi {sol} dice {estado}, ¿qué es eso?", "¿qué estados puede tener una {sol}?",
        "¿qué es {estado} en devops?", "explícame los estados de los {sols}",
        "no entiendo el estado {estado}", "¿cuál es la diferencia entre resolved y closed?",
    ],
    "obtener_id": [
        "no me dieron el número de mi {sol}", "envié el formulario pero no me mostraron el folio",
        "¿cuál es el id de mi {sol}?", "perdí el número de mi {sol}",
        "¿cómo sé cuál es mi número de {sol}?", "no tengo el folio de lo que solicité",
        "¿dónde veo el id de mi {sol}?", "olvidé el identificador de mi {sol}",
        "después de enviar el formulario no apareció ningún número",
        "necesito el número de {sol} que me generaron",
    ],
    "mis_solicitudes": [
        "mi correo es {email}", "busca mis {sols} con {email}", "¿qué {sols} tengo registradas?",
        "lista mis {sols}", "muéstrame todas mis {sols}", "{email}",
        "soy {email}, ¿qué {sols} tengo?", "¿cuántas {sols} tengo abiertas?",
        "quiero ver mis pendientes",
    ],
    "tipos_solicitud": [
        "¿qué tipos de {sol} existen?", "¿qué puedo solicitar al área de datos?",
        "¿qué servicios ofrece administración de datos?", "¿qué puedo pedir en el formulario?",
        "¿cuáles son las categorías de {sol}?", "¿en qué me puede ayudar el área de datos?",
        "¿qué tipo de {sol} elijo?", "no sé qué tipo de {sol} seleccionar",
        "catálogo de servicios de datos",
    ],
    "llenado_formulario": [
        "¿cómo lleno el formulario?", "¿cómo levanto una {sol}?", "¿dónde registro una {sol} nueva?",
        "¿cómo creo un {sol}?", "¿cuál es la liga del formulario?",
        "¿dónde está el formulario para pedir datos?", "¿cómo hago una {sol} al área de datos?",
        "pásame el link para levantar un {sol}", "quiero dar de alta una {sol}",
        "¿qué pasos sigo para registrar mi {sol}?",
    ],
    "campos_obligatorios": [
        "¿qué campos son obligatorios?", "¿qué información necesito para llenar el formulario?",
        "¿qué datos me piden en la {sol}?", "¿es obligatorio poner la justificación?",
        "¿qué debo poner en la descripción?", "¿qué requisitos tiene el formulario?",
        "¿puedo dejar campos vacíos?", "¿qué tengo que escribir en el formulario?",
        "me falta información para enviar el formulario, ¿qué piden?",
    ],
    "solicitud_etl": [
        "necesito que integren una tabla nueva de {tabla} al lakehouse", "quiero un etl desde {origen}",
        "¿pueden cargar la tabla de {tabla} a databricks?", "necesito ingestar datos de {origen}",
        "¿cómo pido que suban una tabla de {tabla} al lakehouse?",
        "quiero automatizar la carga diaria de {tabla}", "necesito un pipeline para {tabla}",
        "migrar la tabla de {tabla} de {origen} a la nube", "¿qué necesito para solicitar un etl?",
        "quiero agregar una nueva fuente de datos de {origen}",
    ],
    "solicitud_sabana": [
        "necesito una sábana de datos de {tabla}", "¿me pueden pasar la información de {tabla} en excel?",
        "quiero una extracción a detalle de {tabla}", "necesito un csv con {tabla}",
        "ocupo un archivo con el detalle de {tabla}", "¿me mandan un excel con los datos de {tabla}?",
        "necesito descargar la base de {tabla}", "¿me pasan un archivo plano de {tabla}?",
        "quiero un reporte detallado de {tabla} en xlsx", "¿cuántas filas caben en un excel?",
    ],
    "reportar_incidencia": [
        "el tablero de {tabla} no tiene información de hoy", "los datos de {tabla} están incorrectos",
        "hay un error en la tabla de {tabla}", "no se actualizó el reporte de {tabla}",
        "el proceso de {tabla} falló", "la carga de {tabla} está caída",
        "me sale un error al consultar {tabla}", "las cifras de {tabla} no cuadran",
        "faltan registros en {tabla}", "quiero reportar una falla en {tabla}",
        "el dashboard de {tabla} no carga",
    ],
    "solicitud_accesos": [
        "necesito acceso a la tabla de {tabla}", "no tengo permisos para ver {tabla}",
        "¿cómo pido permisos al esquema de {tabla}?", "¿me pueden dar acceso a databricks?",
        "quiero permiso de lectura en {tabla}", "me sale acceso denegado en {tabla}",
        "¿cómo solicito acceso al catálogo de {tabla}?",
        "necesito que den de alta a mi compañero en el grupo de {tabla}",
        "no puedo ver el esquema de {tabla}",
    ],
    "tiempos_atencion": [
        "¿cuánto tarda una {sol}?", "¿en cuántos días me entregan?", "¿cuánto tiempo tarda un etl?",
        "¿cuál es el sla de una incidencia?", "¿cuándo estaría lista una sábana?",
        "¿cuánto se demoran en atender?", "¿cuál es el tiempo de respuesta del área?",
        "¿en cuánto tiempo atienden un {sol}?", "¿cuánto debo esperar?",
    ],
    "prioridad": [
        "¿qué prioridad le pongo?", "es urgente", "¿cómo marco mi {sol} como crítica?",
        "¿cuál es la diferencia entre prioridad alta y media?", "¿pueden darle prioridad a mi {sol}{ref}?",
        "lo necesito para hoy", "esto es crítico para el cierre", "¿cómo subo la prioridad de mi {sol}?",
    ],
    "escalar": [
        "quiero hablar con una persona", "¿a quién le escribo si nadie me responde?",
        "necesito hablar con un analista", "¿cómo contacto al área de datos?", "pásame con un humano",
        "quiero poner una queja", "nadie atiende mi {sol}, ¿con quién me comunico?",
        "¿cuál es el correo del área?", "¿hay un teléfono o canal de teams?", "quiero escalar mi caso",
    ],
    "ayuda": [
        "¿qué puedes hacer?", "ayuda", "menú", "¿en qué me ayudas?", "¿qué opciones tengo?",
        "¿cómo funcionas?", "¿para qué sirves?", "¿qué preguntas te puedo hacer?",
    ],
    "saludo": ["hola", "buenos días", "buenas tardes", "¿qué tal?", "hey", "saludos", "hola, buen día",
               "buenas noches", "hola asistente", "qué onda"],
    "agradecimiento": ["gracias", "muchas gracias", "te agradezco", "excelente, gracias", "perfecto",
                       "muy amable", "gracias por la ayuda", "mil gracias", "listo, gracias"],
    "despedida": ["adiós", "hasta luego", "nos vemos", "bye", "salir", "eso es todo", "hasta pronto",
                  "me retiro", "chao"],
    "enviar_captura": [
        "te envío una captura del error", "adjunto una imagen de la pantalla", "te paso un pantallazo",
        "mira esta captura", "¿qué ves en esta imagen?", "¿puedes revisar esta foto del tablero?",
        "aquí está la screenshot", "analiza esta imagen", "te mando una foto de lo que me aparece",
    ],
    "fuera_dominio": [
        "¿cuál es el clima de mañana?", "¿quién ganó el partido?", "recomiéndame una película",
        "cuéntame un chiste", "¿cuál es la capital de francia?", "¿cómo hago un pastel?",
        "¿qué hora es?", "¿a cuánto está el dólar?", "xyz", "asdf", "me gusta el futbol",
        "¿dónde venden pizza?", "¿quién es el presidente?", "jajaja",
    ],
}
SOCIALES = {"saludo", "agradecimiento", "despedida", "fuera_dominio", "ayuda"}
PREFIJOS = ["", "", "", "hola, ", "oye, ", "disculpa, ", "buen día, ", "una pregunta: ", "oigan, "]
SUFIJOS = ["", "", "", " por favor", " gracias", "??", " porfa"]

# Frases escritas a mano con redaccion diferente a las plantillas (no se usan para entrenar).
PRUEBA_RETADORA: list[tuple[str, str]] = [
    ("¿Ya alguien tomó mi ticket 48301?", "estatus"),
    ("Llevo una semana esperando, ¿hay avance con la 48213?", "estatus"),
    ("¿Mi petición sigue en cola o ya la están trabajando?", "estatus"),
    ("¿Qué notas dejó el analista en el folio 48213?", "comentarios"),
    ("Quiero leer lo que escribieron en mi caso", "comentarios"),
    ("¿Hubo alguna respuesta del equipo en mi ticket?", "comentarios"),
    ("¿Cuál fue el resultado final de la 48250?", "resolucion"),
    ("¿Dónde quedó el archivo que me prepararon?", "resolucion"),
    ("Me dijeron que ya está resuelto, ¿qué hicieron exactamente?", "resolucion"),
    ("¿Qué quiere decir que mi ticket esté en Blocked?", "estados_significado"),
    ("No me queda claro qué es Resolved", "estados_significado"),
    ("¿En qué se diferencia Active de New?", "estados_significado"),
    ("Llené el form y nunca vi un número de seguimiento", "obtener_id"),
    ("¿Dónde consigo el folio de mi petición?", "obtener_id"),
    ("Se me olvidó el número que me asignaron", "obtener_id"),
    ("Revisa qué tengo pendiente, soy ana.ruiz@empresa.com", "mis_solicitudes"),
    ("¿Cuántos tickets he levantado?", "mis_solicitudes"),
    ("Enséñame todos mis casos", "mis_solicitudes"),
    ("¿Qué cosas le puedo pedir a su área?", "tipos_solicitud"),
    ("¿Qué categorías hay en el formulario?", "tipos_solicitud"),
    ("¿Qué servicios dan ustedes?", "tipos_solicitud"),
    ("¿Por dónde mando una nueva petición?", "llenado_formulario"),
    ("Quiero registrar un requerimiento, ¿cómo le hago?", "llenado_formulario"),
    ("Mándame la liga del form", "llenado_formulario"),
    ("¿Qué datos debo incluir sí o sí en el formulario?", "campos_obligatorios"),
    ("¿Es necesario poner la fecha requerida?", "campos_obligatorios"),
    ("¿Qué escribo en el campo de justificación?", "campos_obligatorios"),
    ("Queremos tener la tabla de pedidos de Oracle en el lakehouse", "solicitud_etl"),
    ("¿Me ayudan a programar una carga incremental de clientes?", "solicitud_etl"),
    ("Hay que traer los datos de SAP a Databricks", "solicitud_etl"),
    ("Me urge un Excel con todas las ventas de octubre a detalle", "solicitud_sabana"),
    ("¿Me pueden exportar los clientes activos a CSV?", "solicitud_sabana"),
    ("Necesito la base completa de cobranza en un archivo", "solicitud_sabana"),
    ("El tablero de ventas no muestra los datos de ayer", "reportar_incidencia"),
    ("Los totales de inventario salen duplicados", "reportar_incidencia"),
    ("La consulta de cartera me marca timeout", "reportar_incidencia"),
    ("No me deja entrar al esquema de finanzas", "solicitud_accesos"),
    ("Ocupo permisos de lectura para mi equipo", "solicitud_accesos"),
    ("¿Cómo obtengo acceso al catálogo de ventas?", "solicitud_accesos"),
    ("¿Para cuándo podría estar un ETL nuevo?", "tiempos_atencion"),
    ("¿Cuántos días hábiles tardan con una incidencia?", "tiempos_atencion"),
    ("¿Qué tan rápido atienden?", "tiempos_atencion"),
    ("Esto es para el cierre de mes, ¿lo pueden adelantar?", "prioridad"),
    ("¿Cuándo debo marcar algo como crítico?", "prioridad"),
    ("Necesito que lo atiendan ya, es urgentísimo", "prioridad"),
    ("¿Con quién hablo si mi caso no avanza?", "escalar"),
    ("Prefiero que me atienda alguien del equipo", "escalar"),
    ("Dame el contacto de un responsable", "escalar"),
    ("¿Qué cosas sabes hacer?", "ayuda"),
    ("No sé qué preguntarte", "ayuda"),
    ("Muéstrame las opciones", "ayuda"),
    ("Buen día", "saludo"),
    ("Qué onda, asistente", "saludo"),
    ("Holi", "saludo"),
    ("Mil gracias por la info", "agradecimiento"),
    ("Muy amable, gracias", "agradecimiento"),
    ("Te pasaste, gracias", "agradecimiento"),
    ("Eso sería todo, hasta luego", "despedida"),
    ("Ya me voy, bye", "despedida"),
    ("Nos vemos después", "despedida"),
    ("Te comparto un screenshot del problema", "enviar_captura"),
    ("Adjunto foto de la pantalla", "enviar_captura"),
    ("Revisa la imagen que te mando", "enviar_captura"),
    ("¿Quién ganó el clásico?", "fuera_dominio"),
    ("Recomiéndame un restaurante", "fuera_dominio"),
    ("¿Cuánto está el euro hoy?", "fuera_dominio"),
]


def _sin_acentos(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _error_de_dedo(t: str, rnd: random.Random) -> str:
    palabras = t.split()
    largas = [i for i, p in enumerate(palabras) if len(p) > 4 and p.isalpha()]
    if not largas:
        return t
    i = rnd.choice(largas)
    p = palabras[i]
    j = rnd.randrange(1, len(p) - 1)
    palabras[i] = p[:j] + p[j + 1] + p[j] + p[j + 2:] if rnd.random() < 0.5 else p[:j] + p[j + 1:]
    return " ".join(palabras)


def _aumentar(t: str, intencion: str, rnd: random.Random) -> str:
    if intencion not in SOCIALES:
        t = rnd.choice(PREFIJOS) + t + rnd.choice(SUFIJOS)
    if rnd.random() < 0.35:
        t = _sin_acentos(t)
    if rnd.random() < 0.20:
        t = _error_de_dedo(t, rnd)
    if rnd.random() < 0.30:
        t = t.replace("¿", "").replace("?", "")
    if rnd.random() < 0.25:
        t = t.capitalize()
    return t


def _expandir(plantilla: str, rnd: random.Random, max_comb: int) -> list[str]:
    campos = [f for _, f, _, _ in string.Formatter().parse(plantilla) if f]
    if not campos:
        return [plantilla]
    combos = list(itertools.product(*(RANURAS[c] for c in campos)))
    rnd.shuffle(combos)
    salida = []
    for combo in combos[:max_comb]:
        t = plantilla.format(**dict(zip(campos, combo)))
        salida.append(t.replace("NID", str(rnd.randint(40000, 49999))))
    return salida


def generar_dataset(por_intencion: int = 90, semilla: int = 7) -> pd.DataFrame:
    """DataFrame con columnas: texto, intencion, grupo (plantilla de origen)."""
    rnd = random.Random(semilla)
    filas = []
    for intencion, plantillas in PLANTILLAS.items():
        vistos: set[str] = set()
        intentos = 0
        while len(vistos) < por_intencion and intentos < por_intencion * 20:
            intentos += 1
            idx = intentos % len(plantillas)
            base = rnd.choice(_expandir(plantillas[idx], rnd, max_comb=15))
            texto = _aumentar(base, intencion, rnd)
            if texto.lower() in vistos:
                continue
            vistos.add(texto.lower())
            filas.append({"texto": texto, "intencion": intencion, "grupo": f"{intencion}:{idx}"})
    return pd.DataFrame(filas)


def prueba_retadora() -> pd.DataFrame:
    return pd.DataFrame(PRUEBA_RETADORA, columns=["texto", "intencion"])


if __name__ == "__main__":
    df = generar_dataset()
    SALIDAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDAS / "intenciones.csv", index=False, encoding="utf-8")
    print(df.intencion.value_counts().to_string())
    print(f"Total: {len(df)} ejemplos, {df.intencion.nunique()} intenciones, {df.grupo.nunique()} plantillas")
