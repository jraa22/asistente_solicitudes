"""
Pruebas funcionales del chatbot: 10 conversaciones con la intencion esperada
por turno. Genera la evidencia de ejecucion (transcripcion + tabla de aciertos).

Uso:  python pruebas_conversaciones.py
"""
from __future__ import annotations

from chatbot_reglas import ChatbotSolicitudes

CONVERSACIONES: dict[str, list[tuple[str, str]]] = {
    "C01 Saludo, menu y despedida": [
        ("Hola", "saludo"),
        ("Que puedes hacer?", "ayuda"),
        ("Adios", "despedida"),
    ],
    "C02 Estatus y comentarios con memoria de ID": [
        ("Buenos dias, cual es el estatus de mi solicitud 48213?", "estatus"),
        ("Y que comentarios tiene?", "comentarios"),
        ("Gracias", "agradecimiento"),
    ],
    "C03 Recuperar ID por correo": [
        ("Envie el formulario pero no me mostraron el numero de mi solicitud", "obtener_id"),
        ("Mi correo es maria.lopez@empresa.com", "mis_solicitudes"),
        ("Como va la 48213?", "estatus"),
    ],
    "C04 Llenado del formulario": [
        ("Como lleno el formulario?", "llenado_formulario"),
        ("Que campos son obligatorios?", "campos_obligatorios"),
    ],
    "C05 Sabana de datos y tiempos": [
        ("Necesito una sabana de datos en Excel", "solicitud_sabana"),
        ("Cuanto tarda?", "tiempos_atencion"),
    ],
    "C06 ETL y prioridad": [
        ("Quiero que integren una tabla nueva al lakehouse con un ETL", "solicitud_etl"),
        ("Que prioridad debo poner?", "prioridad"),
    ],
    "C07 Incidencia urgente": [
        ("El reporte de inventarios tiene datos incorrectos, hay una falla", "reportar_incidencia"),
        ("Es urgente", "prioridad"),
    ],
    "C08 Resolucion y significado de estados": [
        ("Quiero ver la resolucion de la solicitud 48250", "resolucion"),
        ("Que significa el estado Resolved?", "estados_significado"),
    ],
    "C09 Contexto pendiente, error de dedo y escalamiento": [
        ("Como va mi solicitud?", "estatus"),
        ("48333", "estatus"),
        ("estatuz de la 99999", "estatus"),
        ("Quiero hablar con una persona", "escalar"),
    ],
    "C10 Fuera de dominio y tipos de solicitud": [
        ("Cual es el clima de manana?", "fallback"),
        ("xyz", "fallback"),
        ("Que tipos de solicitudes puedo levantar?", "tipos_solicitud"),
    ],
}


def ejecutar_pruebas() -> tuple[int, int]:
    total = aciertos = 0
    for nombre, turnos in CONVERSACIONES.items():
        bot = ChatbotSolicitudes()          # sesion nueva por conversacion
        print("=" * 78 + f"\n{nombre}\n" + "=" * 78)
        for mensaje, esperada in turnos:
            respuesta = bot.responder(mensaje)
            ok = bot.ultima_intencion == esperada
            total += 1
            aciertos += ok
            print(f"Usuario: {mensaje}")
            print(f"Bot    : {respuesta}")
            print(f"         [intencion={bot.ultima_intencion} score={bot.ultimo_score:.1f} "
                  f"esperada={esperada} -> {'OK' if ok else 'FALLA'}]\n")
    print(f"RESULTADO: {aciertos}/{total} turnos correctos ({aciertos / total:.0%})")
    return aciertos, total


if __name__ == "__main__":
    ejecutar_pruebas()
