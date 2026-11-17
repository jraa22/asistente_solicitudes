"""
Pruebas funcionales del asistente multimodal - Avance 2.

10 conversaciones que combinan texto e imagenes. Las capturas se generan con semillas
distintas a las del entrenamiento (imagenes nunca vistas por la CNN); la conversacion 9
usa una captura 'dificil' (cambio de distribucion).
Cada turno indica las intenciones aceptables y la clase visual esperada.

Uso:  python pruebas_multimodales.py   (requiere haber entrenado ambos modelos)
"""
from __future__ import annotations

import random

from asistente_multimodal import AsistenteMultimodal
from clasificador_imagenes import ClasificadorImagenes
from clasificador_intenciones import ClasificadorIntenciones
from dataset_imagenes import generar_imagen

# (texto, (clase_imagen, dificil) | None, intenciones_aceptables | None, clase_visual_esperada | None)
CONVERSACIONES = {
    "M01 Texto: saludo, estatus y comentarios con contexto": [
        ("Holi", None, {"saludo"}, None),
        ("Ya alguien tomo mi ticket 48213?", None, {"estatus"}, None),
        ("Y que notas le dejaron?", None, {"comentarios"}, None),
    ],
    "M02 Incidencia con captura de error": [
        ("El proceso de cartera fallo en la noche", ("mensaje_error", False), {"reportar_incidencia"}, "mensaje_error"),
    ],
    "M03 Solo imagen (sin texto)": [
        ("", ("tablero_graficas", False), None, "tablero_graficas"),
    ],
    "M04 Envio de captura con texto generico": [
        ("Te comparto un screenshot", ("tabla_datos", False), {"enviar_captura"}, "tabla_datos"),
    ],
    "M05 Anuncia captura y la envia despues": [
        ("Te paso un pantallazo de lo que me sale", None, {"enviar_captura"}, None),
        ("", ("mensaje_error", False), None, "mensaje_error"),
    ],
    "M06 Texto e imagen incoherentes": [
        ("Necesito una sabana de clientes en Excel", ("formulario", False), {"solicitud_sabana"}, "formulario"),
    ],
    "M07 Seguimiento y evidencia asociada al ID": [
        ("Como va la 48301?", None, {"estatus"}, None),
        ("Mira, asi se ve el tablero sin datos", ("tablero_graficas", False), {"reportar_incidencia", "enviar_captura"}, "tablero_graficas"),
    ],
    "M08 Accesos con captura de error": [
        ("No me deja entrar al esquema de finanzas", ("mensaje_error", False), {"solicitud_accesos"}, "mensaje_error"),
    ],
    "M09 Captura de baja calidad (cambio de distribucion)": [
        ("Que ves en esta imagen?", ("editor_sql", True), {"enviar_captura"}, "editor_sql"),
    ],
    "M10 Fuera de dominio, correo y formulario": [
        ("Quien gano el clasico?", None, {"fallback"}, None),
        ("soy maria.lopez@empresa.com, que tickets tengo?", None, {"mis_solicitudes"}, None),
        ("Mandame la liga del form", None, {"llenado_formulario"}, None),
    ],
}


def ejecutar_pruebas(semilla_imagenes: int = 2026) -> dict:
    nlp, vision = ClasificadorIntenciones(), ClasificadorImagenes()
    r = random.Random(semilla_imagenes)
    tot_t = ok_t = tot_v = ok_v = 0
    for nombre, turnos in CONVERSACIONES.items():
        bot = AsistenteMultimodal(nlp, vision)
        print("=" * 90 + f"\n{nombre}\n" + "=" * 90)
        for texto, img_cfg, esperadas, clase_esp in turnos:
            imagen = generar_imagen(img_cfg[0], r, dificil=img_cfg[1]) if img_cfg else None
            respuesta = bot.responder(texto, imagen)
            print(f"Usuario: {texto or '(sin texto)'}" + (f"  [adjunta captura: {img_cfg[0]}{' - dificil' if img_cfg[1] else ''}]" if img_cfg else ""))
            print(f"Bot    : {respuesta}")
            marcas = []
            if esperadas:
                ok = bot.ultima_intencion in esperadas
                tot_t, ok_t = tot_t + 1, ok_t + ok
                marcas.append(f"intencion={bot.ultima_intencion} ({bot.ultimo_score:.2f}) {'OK' if ok else 'FALLA'}")
            if clase_esp:
                a = bot.ultimo_analisis
                ok = a["clase"] == clase_esp
                tot_v, ok_v = tot_v + 1, ok_v + ok
                marcas.append(f"imagen={a['clase']} ({a['probabilidad']:.2f}) {'OK' if ok else 'FALLA'}")
                if bot.borrador:
                    marcas.append(f"borrador={bot.borrador}")
            print("         [" + " | ".join(marcas) + "]\n")
    print(f"RESULTADO TEXTO : {ok_t}/{tot_t} turnos correctos")
    print(f"RESULTADO IMAGEN: {ok_v}/{tot_v} imagenes correctas")
    return {"texto": (ok_t, tot_t), "imagen": (ok_v, tot_v)}


if __name__ == "__main__":
    ejecutar_pruebas()
