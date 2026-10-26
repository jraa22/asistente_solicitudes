"""
Asistente de Solicitudes del Area de Administracion de Datos (v1 - basado en reglas)
=================================================================================
Chatbot basado en reglas para orientar a los usuarios sobre:
  * Seguimiento de solicitudes registradas en Azure DevOps Boards (estatus,
    comentarios, resolucion, recuperacion del ID).
  * Llenado del formulario de Google Forms y tipos de solicitud (ETL, sabanas
    de datos CSV/XLSX, incidencias, etc.).

Requisitos: Python 3.10+ y NLTK (pip install nltk).
No requiere descargar corpus de NLTK: usa RegexpTokenizer, SnowballStemmer
(espanol) y edit_distance.

Uso interactivo:  python chatbot_reglas.py
"""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field

from nltk.metrics.distance import edit_distance
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

# ---------------------------------------------------------------------------
# 1. Preprocesamiento (NLTK)
# ---------------------------------------------------------------------------
TOKENIZER = RegexpTokenizer(r"[a-z0-9@._-]+")
STEMMER = SnowballStemmer("spanish")
STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o",
    "a", "en", "por", "para", "con", "mi", "mis", "me", "que", "se", "lo",
    "al", "del", "es", "su", "sus", "le", "les", "tu", "te", "cual", "esta",
    "este", "esa", "ese", "hay", "muy", "ya", "si", "pero", "son", "tiene",
    "quiero", "necesito", "solicitud", "solicitudes", "favor",
}
RE_ID = re.compile(r"(?<![\w@.])#?(\d{4,6})(?![\w@])")
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
UMBRAL = 0.5          # puntaje minimo para aceptar una intencion
PESO_SOCIAL = 0.9     # saludos/gracias pierden contra intenciones de negocio


def normalizar(texto: str) -> str:
    """Minusculas y eliminacion de acentos/diacriticos."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def preprocesar(texto: str) -> list[str]:
    """Tokeniza, filtra stopwords/numeros y aplica stemming."""
    tokens = TOKENIZER.tokenize(normalizar(texto))
    return [STEMMER.stem(t) for t in tokens if t not in STOPWORDS and not t.isdigit()]


# ---------------------------------------------------------------------------
# 2. Base de conocimiento
# ---------------------------------------------------------------------------
# 2.1 Simulacion de work items de Azure DevOps Boards (en la etapa de
#     integracion se sustituira por la API REST de Azure DevOps).
WORK_ITEMS: dict[int, dict] = {
    48213: {"titulo": "ETL de tabla ventas_diarias hacia el Lakehouse", "tipo": "ETL / Ingesta",
            "estado": "Active", "solicitante": "maria.lopez@empresa.com",
            "asignado": "Equipo de Ingesta", "alta": "2026-10-02", "actualizado": "2026-10-21",
            "comentarios": ["2026-10-05 - Analista: Se valido el acceso a la base origen.",
                            "2026-10-21 - Analista: Pipeline en pruebas en ambiente QA."],
            "resolucion": None},
    48250: {"titulo": "Sabana de clientes activos 2026 (XLSX)", "tipo": "Sabana de datos",
            "estado": "Resolved", "solicitante": "maria.lopez@empresa.com",
            "asignado": "Equipo de Explotacion", "alta": "2026-10-08", "actualizado": "2026-10-15",
            "comentarios": ["2026-10-10 - Analista: Se confirmo el filtro por region."],
            "resolucion": "Archivo entregado en la carpeta compartida /sabanas/48250 (1.2 M registros)."},
    48301: {"titulo": "Incidencia: tablero de inventarios sin datos del dia", "tipo": "Incidencia",
            "estado": "New", "solicitante": "juan.perez@empresa.com",
            "asignado": "Sin asignar", "alta": "2026-10-23", "actualizado": "2026-10-23",
            "comentarios": [], "resolucion": None},
    48120: {"titulo": "Alta de permisos de lectura al esquema finanzas", "tipo": "Accesos",
            "estado": "Closed", "solicitante": "juan.perez@empresa.com",
            "asignado": "Equipo de Gobierno de Datos", "alta": "2026-09-20", "actualizado": "2026-09-24",
            "comentarios": ["2026-09-22 - Analista: Autorizado por el data owner."],
            "resolucion": "Permiso SELECT otorgado al grupo finanzas_lectura."},
    48333: {"titulo": "ETL de catalogo de proveedores", "tipo": "ETL / Ingesta",
            "estado": "Blocked", "solicitante": "ana.ruiz@empresa.com",
            "asignado": "Equipo de Ingesta", "alta": "2026-10-12", "actualizado": "2026-10-20",
            "comentarios": ["2026-10-20 - Analista: En espera de credenciales de la base origen por parte del usuario."],
            "resolucion": None},
}

ESTADOS = {
    "New": "Nueva: registrada y pendiente de asignacion a un analista.",
    "Active": "En atencion: un analista ya trabaja en ella.",
    "Blocked": "En espera: se requiere informacion o accion del solicitante o de un tercero.",
    "Resolved": "Resuelta: el entregable esta listo; pendiente de tu validacion.",
    "Closed": "Cerrada: atendida y validada; no requiere mas acciones.",
}

FORM_URL = "https://forms.gle/FORMULARIO-ADMIN-DATOS"   # liga ilustrativa
CORREO_AREA = "administracion.datos@empresa.com"


# 2.2 Intenciones conversacionales
@dataclass
class Intencion:
    nombre: str
    palabras: list[str]
    frases: list[str] = field(default_factory=list)
    respuestas: list[str] = field(default_factory=list)
    requiere_id: bool = False
    social: bool = False
    stems: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.stems = {STEMMER.stem(normalizar(p)) for p in self.palabras}
        self.frases = [normalizar(f) for f in self.frases]

    def puntuar(self, stems: list[str], texto_norm: str) -> float:
        """Frase exacta = 2 pts, palabra clave = 1 pt, coincidencia aproximada = 0.5 pts."""
        score = sum(2.0 for f in self.frases if f in texto_norm)
        for t in stems:
            if t in self.stems:
                score += 1.0
            elif len(t) >= 5 and any(len(k) >= 5 and edit_distance(t, k) <= 1 for k in self.stems):
                score += 0.5          # tolerancia a errores de dedo
        return score * (PESO_SOCIAL if self.social else 1.0)


INTENCIONES: list[Intencion] = [
    Intencion("estatus", ["estatus", "status", "avance", "seguimiento", "progreso", "estado"],
              ["como va", "en que va", "como esta mi"], requiere_id=True),
    Intencion("comentarios", ["comentario", "comentarios", "notas", "observaciones", "respondieron", "dijeron"],
              ["que me contestaron"], requiere_id=True),
    Intencion("resolucion", ["resolucion", "resuelta", "resolvieron", "solucion", "entregable", "cerraron"],
              ["como quedo"], requiere_id=True),
    Intencion("estados_significado", ["significa", "significado", "estado", "new", "active", "resolved", "closed", "blocked"],
              ["que significa", "que quiere decir"],
              ["Los estados de un work item en Azure DevOps son:\n" +
               "\n".join(f"  - {k}: {v}" for k, v in ESTADOS.items())]),
    Intencion("obtener_id", ["id", "numero", "folio", "identificador", "perdi"],
              ["no me mostraron", "no me dieron", "no se mi", "no tengo el"],
              ["El formulario aun no muestra el ID al enviarlo. Puedo buscar tus solicitudes "
               "por correo: escribe 'mi correo es nombre@empresa.com'."]),
    Intencion("mis_solicitudes", ["correo", "email", "mail", "listar", "todas"], ["mis solicitudes"]),
    Intencion("tipos_solicitud", ["tipos", "tipo", "catalogo", "servicios", "categorias"],
              ["que puedo solicitar", "tipos de solicitud", "que puedo pedir"],
              ["Puedes solicitar: (1) ETL/ingesta de tablas al Lakehouse, (2) sabanas de datos a "
               "detalle en CSV/XLSX, (3) atencion de incidencias en datos o procesos, (4) accesos y "
               "permisos a esquemas/tablas, (5) modificaciones a procesos existentes. "
               "Preguntame por cualquiera para saber que informacion incluir."]),
    Intencion("llenado_formulario", ["llenar", "llenado", "formulario", "form", "levantar", "registrar", "crear"],
              ["como levanto", "como lleno"],
              [f"Para levantar una solicitud entra a {FORM_URL}: 1) usa tu correo corporativo, "
               "2) elige el tipo de solicitud, 3) describe el requerimiento y su objetivo de negocio, "
               "4) indica tablas/fuentes involucradas, 5) fecha requerida y prioridad, "
               "6) adjunta ejemplos si los tienes. Escribe 'campos obligatorios' para ver el detalle."]),
    Intencion("campos_obligatorios", ["campos", "campo", "obligatorio", "obligatorios", "requisitos", "requerido"],
              ["que informacion", "que datos necesito"],
              ["Campos obligatorios: nombre, correo corporativo, area/gerencia, tipo de solicitud, "
               "descripcion detallada, justificacion de negocio, fecha requerida y prioridad. "
               "Opcionales: tablas/fuentes, filtros, formato de salida (CSV/XLSX) y adjuntos."]),
    Intencion("solicitud_etl", ["etl", "ingesta", "tabla", "lakehouse", "pipeline", "integrar", "cargar"],
              ["tabla nueva"],
              ["Para un ETL elige 'ETL / Ingesta' e indica: base y tabla origen, frecuencia de "
               "actualizacion (diaria, semanal...), tipo de carga (completa o incremental), "
               "columnas clave, data owner que autoriza y usuarios que consumiran la tabla."]),
    Intencion("solicitud_sabana", ["sabana", "csv", "xlsx", "excel", "extraccion", "archivo", "detalle", "descarga"],
              ["sabana de datos"],
              ["Para una sabana de datos indica: tablas o indicadores, columnas requeridas, "
               "filtros (fechas, regiones, etc.), formato (CSV recomendado para mas de 1 M de "
               "filas; XLSX tiene limite de 1,048,576 filas) y si es unica o recurrente."]),
    Intencion("reportar_incidencia", ["incidencia", "error", "falla", "fallo", "problema", "caido", "bug"],
              ["datos incorrectos", "no carga", "no aparecen", "no se actualizo"],
              ["Registra la incidencia con el tipo 'Incidencia' e incluye: proceso/tabla/tablero "
               "afectado, fecha y hora en que se detecto, captura de pantalla, ejemplo de registro "
               "erroneo y el impacto en el negocio. Las incidencias criticas se atienden primero."]),
    Intencion("tiempos_atencion", ["tiempo", "tarda", "tardan", "cuanto", "sla", "plazo", "demora"],
              ["cuantos dias", "cuando estara"],
              ["Tiempos objetivo de atencion (dias habiles): incidencia critica 1, incidencia normal 3, "
               "sabana de datos 3-5, accesos 2, ETL nuevo 10-15 segun complejidad. El tiempo corre "
               "desde que la solicitud pasa a 'Active'."]),
    Intencion("prioridad", ["prioridad", "urgente", "urgencia", "critico", "critica"], [],
              ["Prioridades: Critica (operacion detenida o afectacion a clientes), Alta (afecta una "
               "decision o cierre con fecha comprometida), Media (mejora o requerimiento planeado), "
               "Baja (sin fecha). Una prioridad Critica requiere justificacion y visto bueno de tu gerente."]),
    Intencion("escalar", ["hablar", "persona", "humano", "analista", "escalar", "contacto", "queja"], [],
              [f"Puedes contactar al Area de Administracion de Datos en {CORREO_AREA} o en el canal "
               "de Teams 'Soporte Datos'. Incluye siempre el ID de tu solicitud."]),
    Intencion("ayuda", ["ayuda", "menu", "opciones"], ["que puedes hacer", "en que me ayudas"],
              ["Puedo ayudarte a: consultar el estatus, comentarios o resolucion de una solicitud "
               "(dame su ID), recuperar tus IDs con tu correo, explicarte los tipos de solicitud, "
               "el llenado del formulario, prioridades y tiempos de atencion."]),
    Intencion("saludo", ["hola", "buenos", "buenas", "saludos", "hey"], ["que tal"],
              ["Hola, soy el asistente del Area de Administracion de Datos. Puedo darte "
               "seguimiento a tus solicitudes o ayudarte a llenar el formulario. Que necesitas?"],
              social=True),
    Intencion("agradecimiento", ["gracias", "agradezco", "excelente", "perfecto"], [],
              ["Con gusto. Hay algo mas en lo que te pueda ayudar?"], social=True),
    Intencion("despedida", ["adios", "bye", "salir"], ["hasta luego", "nos vemos"],
              ["Hasta luego. Recuerda que puedes consultar tus solicitudes cuando quieras."], social=True),
]


# ---------------------------------------------------------------------------
# 3. Motor de dialogo
# ---------------------------------------------------------------------------
class ChatbotSolicitudes:
    def __init__(self, semilla: int = 42) -> None:
        self.rnd = random.Random(semilla)
        self.pendiente: str | None = None   # intencion esperando un ID
        self.ultimo_id: int | None = None   # memoria de corto plazo
        self.fallos = 0
        self.ultima_intencion = ""
        self.ultimo_score = 0.0

    # -- clasificacion de intencion ------------------------------------------
    def clasificar(self, mensaje: str) -> tuple[str, float, list[int], str | None]:
        texto = normalizar(mensaje)
        ids = [int(x) for x in RE_ID.findall(texto)]
        email_m = RE_EMAIL.search(texto)
        email = email_m.group(0) if email_m else None
        stems = preprocesar(re.sub(RE_EMAIL, " ", texto))
        scores = {i.nombre: i.puntuar(stems, texto) for i in INTENCIONES}
        if ids:
            scores["obtener_id"] = 0.0
        if email:
            scores["mis_solicitudes"] += 3.0
        nombre, score = max(scores.items(), key=lambda kv: kv[1])
        if ids and score < 1.0:             # el usuario solo envio un numero
            nombre, score = (self.pendiente or "estatus"), 1.0
        return nombre, score, ids, email

    # -- respuesta -----------------------------------------------------------
    def responder(self, mensaje: str) -> str:
        nombre, score, ids, email = self.clasificar(mensaje)
        self.ultima_intencion, self.ultimo_score = nombre, score
        if score < UMBRAL:
            self.ultima_intencion = "fallback"
            self.fallos += 1
            if self.fallos >= 2:
                return ("Sigo sin entenderte. Escribe 'ayuda' para ver lo que puedo hacer o "
                        f"contacta a {CORREO_AREA}.")
            return ("No entendi tu mensaje. Puedes preguntarme por el estatus de una solicitud "
                    "(ej. 'estatus 48213') o como llenar el formulario.")
        self.fallos = 0
        intencion = next(i for i in INTENCIONES if i.nombre == nombre)

        if nombre == "mis_solicitudes":
            return self._listar_por_correo(email)
        if intencion.requiere_id:
            wid = ids[0] if ids else self.ultimo_id
            if wid is None:
                self.pendiente = nombre
                return ("Con gusto. Indicame el ID de tu solicitud (ej. 48213). Si no lo "
                        "tienes, escribe tu correo corporativo y la busco.")
            self.pendiente = None
            return self._consultar(nombre, wid)
        return self.rnd.choice(intencion.respuestas)

    # -- acciones sobre la base de conocimiento ------------------------------
    def _consultar(self, nombre: str, wid: int) -> str:
        wi = WORK_ITEMS.get(wid)
        if wi is None:
            return (f"No encontre la solicitud #{wid}. Verifica el numero o escribe tu correo "
                    "para listar tus solicitudes.")
        self.ultimo_id = wid
        if nombre == "estatus":
            return (f"Solicitud #{wid} - {wi['titulo']}\n  Tipo: {wi['tipo']}\n"
                    f"  Estado: {wi['estado']} ({ESTADOS[wi['estado']]})\n"
                    f"  Asignada a: {wi['asignado']}\n  Alta: {wi['alta']} | "
                    f"Ultima actualizacion: {wi['actualizado']}")
        if nombre == "comentarios":
            if not wi["comentarios"]:
                return f"La solicitud #{wid} aun no tiene comentarios del analista."
            return f"Comentarios de la solicitud #{wid}:\n  " + "\n  ".join(wi["comentarios"])
        # resolucion
        if wi["resolucion"]:
            return f"Resolucion de la solicitud #{wid} ({wi['estado']}): {wi['resolucion']}"
        return (f"La solicitud #{wid} aun no tiene resolucion; su estado es {wi['estado']} "
                f"({ESTADOS[wi['estado']]})")

    def _listar_por_correo(self, email: str | None) -> str:
        if not email:
            return "Escribe tu correo corporativo (ej. nombre@empresa.com) para buscar tus solicitudes."
        propias = [(k, v) for k, v in WORK_ITEMS.items() if v["solicitante"] == email]
        if not propias:
            return f"No encontre solicitudes registradas con el correo {email}."
        lineas = [f"  #{k} | {v['estado']:<8} | {v['titulo']}" for k, v in sorted(propias)]
        return f"Encontre {len(propias)} solicitud(es) para {email}:\n" + "\n".join(lineas)


def main() -> None:
    bot = ChatbotSolicitudes()
    print("Asistente de Solicitudes - escribe 'salir' para terminar.")
    while True:
        try:
            mensaje = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not mensaje:
            continue
        print("Bot:", bot.responder(mensaje))
        if bot.ultima_intencion == "despedida":
            break


if __name__ == "__main__":
    main()
