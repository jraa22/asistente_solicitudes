"""
Asistente multimodal (texto + imagen) - Avance 2.

Integra:
  * Componente conversacional: clasificador de intenciones SVM + TF-IDF (clasificador_intenciones.py)
    con umbral de confianza, reutilizando el gestor de dialogo, la memoria de contexto y las
    respuestas del chatbot de reglas v1 (chatbot_reglas.py) sobre la base de tickets ampliada.
  * Componente visual: CNN EfficientNet-B0 con transferencia de aprendizaje (clasificador_imagenes.py)
    que interpreta la captura adjunta.
  * Fusion: la clase visual se traduce a un tipo de solicitud, prioridad sugerida e informacion a
    incluir; se verifica la coherencia entre lo que dice el texto y lo que muestra la imagen y se
    arma un borrador de solicitud.

Uso interactivo:  python asistente_multimodal.py
  Para adjuntar una imagen escribe el mensaje seguido de  img:/ruta/a/captura.png
"""
from __future__ import annotations

import re
from pathlib import Path

from base_conocimiento import cargar_base
from chatbot_reglas import CORREO_AREA, RE_EMAIL, RE_ID, ChatbotSolicitudes
from chatbot_reglas import normalizar as normalizar_v1
from clasificador_intenciones import ClasificadorIntenciones

UMBRAL_IMAGEN = 0.60

RESP_ACCESOS = ("Para solicitar accesos elige el tipo 'Accesos' e indica: catalogo/esquema/tabla, "
                "tipo de permiso (lectura o escritura), usuarios o grupo y el data owner que autoriza. "
                "Tiempo objetivo: 2 dias habiles.")

MAPEO_VISUAL = {
    "mensaje_error": {"desc": "un mensaje de error o traza de excepcion", "tipo": "Incidencia tecnologica",
                      "prioridad": "Alta", "compatibles": {"reportar_incidencia", "solicitud_accesos"},
                      "pide": "el proceso o tabla afectada, la hora en que ocurrio y el texto completo del error"},
    "tablero_graficas": {"desc": "un tablero con graficas", "tipo": "Incidencia en tablero/reporte",
                         "prioridad": "Media", "compatibles": {"reportar_incidencia"},
                         "pide": "el nombre del tablero, el indicador con problema y el valor que esperabas"},
    "tabla_datos": {"desc": "una tabla u hoja de datos", "tipo": "Incidencia de datos o Sabana de datos",
                    "prioridad": "Media", "compatibles": {"reportar_incidencia", "solicitud_sabana"},
                    "pide": "registros de ejemplo con el valor correcto, o el formato que necesitas si es una sabana"},
    "editor_sql": {"desc": "codigo SQL", "tipo": "ETL / Ingesta o Modificacion de proceso",
                   "prioridad": "Media", "compatibles": {"solicitud_etl", "reportar_incidencia"},
                   "pide": "las tablas origen, la frecuencia de carga y si la consulta debe automatizarse"},
    "formulario": {"desc": "el formulario de solicitudes", "tipo": "Duda de llenado del formulario",
                   "prioridad": "Baja", "compatibles": {"llenado_formulario", "campos_obligatorios", "tipos_solicitud"},
                   "pide": "el campo en el que tienes dudas"},
}
# Intenciones de tarea para las que tiene sentido validar coherencia texto-imagen
INTENCIONES_TAREA = {"reportar_incidencia", "solicitud_sabana", "solicitud_etl", "solicitud_accesos",
                     "llenado_formulario", "campos_obligatorios", "tipos_solicitud"}


class AsistenteMultimodal(ChatbotSolicitudes):
    def __init__(self, modelo_texto: ClasificadorIntenciones | None = None, modelo_imagen=None,
                 semilla: int = 42) -> None:
        super().__init__(semilla)
        cargar_base()
        self.nlp = modelo_texto or ClasificadorIntenciones()
        self._vision = modelo_imagen
        self.umbral = self.nlp.umbral
        self.ultimo_top: list[tuple[str, float]] = []
        self.ultimo_analisis: dict | None = None
        self.borrador: dict | None = None

    @property
    def vision(self):
        if self._vision is None:                     # carga diferida (torch es pesado)
            from clasificador_imagenes import ClasificadorImagenes
            self._vision = ClasificadorImagenes()
        return self._vision

    # -- componente textual (sustituye las reglas de la v1) -----------------------------
    def clasificar(self, mensaje: str):
        texto = normalizar_v1(mensaje)
        ids = [int(x) for x in RE_ID.findall(texto)]
        m = RE_EMAIL.search(texto)
        email = m.group(0) if m else None
        if ids and not re.sub(r"[#?.!,\s]", "", RE_ID.sub("", texto)):
            return (self.pendiente or "estatus"), 1.0, ids, email   # solo envio el numero
        intencion, prob, top = self.nlp.predecir(mensaje)
        if ids and intencion == "obtener_id" and len(top) > 1:     # ya trae el ID
            intencion, prob = top[1]
        if email and intencion != "mis_solicitudes" and prob < 0.6:
            intencion, prob = "mis_solicitudes", max(prob, self.umbral)
        self.ultimo_top = top
        if intencion == "fuera_dominio":
            prob = 0.0                                          # usa el fallback de la v1
        return intencion, prob, ids, email

    # -- componente visual -------------------------------------------------------------
    def analizar_imagen(self, imagen) -> dict:
        r = self.vision.predecir(imagen)
        self.ultimo_analisis = r
        return r

    def _explicar_imagen(self, a: dict, intencion_texto: str | None) -> str:
        info = MAPEO_VISUAL[a["clase"]]
        if a["probabilidad"] < UMBRAL_IMAGEN:
            (c1, p1), (c2, p2) = a["top3"][:2]
            self.borrador = None
            return (f"Revise tu imagen pero no estoy seguro: podria ser {MAPEO_VISUAL[c1]['desc']} ({p1:.0%}) "
                    f"o {MAPEO_VISUAL[c2]['desc']} ({p2:.0%}). Me confirmas de que se trata?")
        texto = (f"Revise tu imagen: parece {info['desc']} (confianza {a['probabilidad']:.0%}). "
                 f"Te sugiero registrarla como '{info['tipo']}' con prioridad {info['prioridad']}. "
                 f"Incluye {info['pide']}.")
        if intencion_texto in INTENCIONES_TAREA and intencion_texto not in info["compatibles"]:
            texto += (f" Ojo: tu mensaje parece de '{intencion_texto}', pero la imagen corresponde a "
                      f"{info['desc']}; verifica que adjuntaste la captura correcta.")
        if self.ultimo_id:
            texto += f" La agregare como evidencia de la solicitud #{self.ultimo_id}."
        self.borrador = {"tipo_sugerido": info["tipo"], "prioridad_sugerida": info["prioridad"],
                         "evidencia": a["clase"], "confianza": round(a["probabilidad"], 3),
                         "solicitud_relacionada": self.ultimo_id, "intencion_texto": intencion_texto}
        return texto

    # -- fusion ------------------------------------------------------------------------
    def responder(self, mensaje: str = "", imagen=None) -> str:
        partes: list[str] = []
        analisis = self.analizar_imagen(imagen) if imagen is not None else None
        intencion_texto = None
        if mensaje.strip():
            intencion, prob, _, _ = self.clasificar(mensaje)
            if prob >= self.umbral and intencion == "solicitud_accesos":
                self.ultima_intencion, self.ultimo_score, self.fallos = intencion, prob, 0
                partes.append(RESP_ACCESOS)
            elif prob >= self.umbral and intencion == "enviar_captura":
                self.ultima_intencion, self.ultimo_score, self.fallos = intencion, prob, 0
                if analisis is None:
                    partes.append("Claro, adjunta la captura de pantalla y la reviso.")
            elif analisis is not None and prob < self.umbral:
                self.ultima_intencion, self.ultimo_score = "enviar_captura", prob  # la imagen guia
            else:
                partes.append(super().responder(mensaje))
            intencion_texto = self.ultima_intencion
        if analisis is not None:
            partes.append(self._explicar_imagen(analisis, intencion_texto))
        return "\n".join(partes) if partes else "En que te puedo ayudar?"


def main() -> None:
    bot = AsistenteMultimodal()
    print("Asistente multimodal - escribe 'salir' para terminar. Adjunta imagenes con img:/ruta.png")
    while True:
        try:
            linea = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not linea:
            continue
        m = re.search(r"img:(\S+)", linea)
        imagen = Path(m.group(1)) if m else None
        texto = re.sub(r"img:\S+", "", linea).strip()
        print("Bot:", bot.responder(texto, imagen))
        if bot.ultima_intencion == "despedida":
            break


if __name__ == "__main__":
    main()
