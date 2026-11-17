"""
Conjunto de imagenes del dominio - Avance 2.

Los usuarios suelen adjuntar CAPTURAS DE PANTALLA a sus solicitudes e incidencias. Como no se
cuenta con un repositorio etiquetado de capturas reales (y estas pueden contener datos
confidenciales), se genera un conjunto sintetico y reproducible de capturas en 5 clases:

  mensaje_error     ventanas de error / trazas de excepcion   -> incidencia tecnologica
  tablero_graficas  dashboards con graficas y KPIs            -> incidencia en tablero/reporte
  tabla_datos       hojas de calculo / resultados tabulares   -> datos incorrectos o sabana
  editor_sql        editores de codigo SQL                    -> ETL / consulta
  formulario        formularios web (tipo Google Forms)       -> duda de llenado

Cada imagen varia en tamano, tema claro/oscuro, colores, barra de navegador, textos, ruido,
compresion JPEG, desenfoque y elementos distractores (p. ej. un error sobre un tablero, o una
rejilla de resultados bajo un editor SQL). El modo 'dificil' genera un conjunto de prueba
con cambio de distribucion (otra paleta, baja resolucion, rotacion) para medir robustez.

Estructura generada (compatible con torchvision.datasets.ImageFolder):
  salidas/imagenes/{train,val,test,ood}/{clase}/img_00001.png

Uso:  python dataset_imagenes.py
"""
from __future__ import annotations

import io
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SALIDAS = Path(os.environ.get("ASISTENTE_SALIDAS", Path(__file__).resolve().parent / "salidas"))
CLASES = ["mensaje_error", "tablero_graficas", "tabla_datos", "editor_sql", "formulario"]

PALABRAS = ("ventas inventario cliente cartera region sucursal total monto fecha estatus proveedor "
            "cobranza credito pedido nomina tienda producto categoria saldo importe").split()
ERRORES = ["No se pudo conectar con el servidor", "java.sql.SQLException: timeout",
           "Permission denied: table finanzas", "NullPointerException en el proceso",
           "Error 500: Internal Server Error", "AnalysisException: Table not found",
           "El archivo no tiene el formato esperado", "Connection reset by peer"]
SQL_KW = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "INSERT INTO", "MERGE INTO",
          "LEFT JOIN", "HAVING", "WITH", "CREATE TABLE", "AND", "ON", "AS", "SUM(", "COUNT("]

_FUENTES_TTF = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf", "C:/Windows/Fonts/arial.ttf"]
_FUENTES_MONO = ["DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "Consolas.ttf", "C:/Windows/Fonts/consola.ttf"]
_cache_fuentes: dict = {}


def fuente(tam: int, mono: bool = False):
    clave = (tam, mono)
    if clave not in _cache_fuentes:
        f = None
        for ruta in (_FUENTES_MONO if mono else []) + _FUENTES_TTF:
            try:
                f = ImageFont.truetype(ruta, tam)
                break
            except OSError:
                continue
        if f is None:
            try:
                f = ImageFont.load_default(size=tam)
            except TypeError:
                f = ImageFont.load_default()
        _cache_fuentes[clave] = f
    return _cache_fuentes[clave]


def _color(r: random.Random, claro: bool) -> tuple[int, int, int]:
    base = (170, 255) if claro else (20, 90)
    return tuple(r.randint(*base) for _ in range(3))


def _vivo(r: random.Random) -> tuple[int, int, int]:
    paleta = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88), (103, 58, 183),
              (0, 150, 136), (255, 112, 67), (63, 81, 181), (121, 85, 72), (233, 30, 99)]
    c = r.choice(paleta)
    return tuple(max(0, min(255, v + r.randint(-25, 25))) for v in c)


def _frase(r: random.Random, n: int = 3) -> str:
    return " ".join(r.choice(PALABRAS) for _ in range(n)).capitalize()


def _barra_navegador(d: ImageDraw.ImageDraw, w: int, r: random.Random) -> int:
    alto = r.randint(22, 34)
    d.rectangle([0, 0, w, alto], fill=(r.randint(215, 240),) * 3)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([8 + i * 16, alto // 2 - 5, 18 + i * 16, alto // 2 + 5], fill=c)
    d.rounded_rectangle([70, 5, w - 40, alto - 5], radius=6, fill=(255, 255, 255))
    d.text((80, 7), "https://" + r.choice(["databricks", "powerbi", "forms", "portal"]) + ".empresa.com",
           fill=(90, 90, 90), font=fuente(10))
    return alto


# ------------------------------------------------------------------- dibujantes por clase
def _tablero(img: Image.Image, r: random.Random, y0: int) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    oscuro = r.random() < 0.3
    d.rectangle([0, y0, w, h], fill=(30, 33, 40) if oscuro else (r.randint(235, 250),) * 3)
    d.rectangle([0, y0, w, y0 + 30], fill=_vivo(r))
    d.text((10, y0 + 8), "Tablero de " + r.choice(PALABRAS), fill=(255, 255, 255), font=fuente(13))
    cols, filas = r.choice([(2, 2), (3, 2), (2, 1), (3, 1), (4, 2)])
    pw, ph = (w - 10) // cols, (h - y0 - 40) // filas
    for i in range(cols):
        for j in range(filas):
            x1, y1 = 5 + i * pw + 4, y0 + 36 + j * ph + 4
            x2, y2 = x1 + pw - 8, y1 + ph - 8
            d.rectangle([x1, y1, x2, y2], fill=(45, 49, 58) if oscuro else (255, 255, 255),
                        outline=(200, 200, 200))
            tipo = r.choice(["barras", "lineas", "pastel", "kpi", "barras", "lineas"])
            iw, ih = x2 - x1, y2 - y1
            if tipo == "kpi" or ih < 40:
                d.text((x1 + 8, y1 + 6), r.choice(PALABRAS).upper(), fill=(150, 150, 150), font=fuente(10))
                d.text((x1 + 8, y1 + 20), f"{r.randint(1, 999)},{r.randint(100, 999)}", fill=_vivo(r), font=fuente(20))
            elif tipo == "barras":
                n = r.randint(4, 10)
                bw = max(2, (iw - 20) // n - 3)
                for k in range(n):
                    bh = r.randint(5, ih - 25)
                    d.rectangle([x1 + 10 + k * (bw + 3), y2 - 8 - bh, x1 + 10 + k * (bw + 3) + bw, y2 - 8], fill=_vivo(r) if r.random() < 0.2 else (66, 133, 244))
            elif tipo == "lineas":
                for _ in range(r.randint(1, 3)):
                    pts = [(x1 + 10 + k * (iw - 20) // 9, y2 - 10 - r.randint(0, ih - 30)) for k in range(10)]
                    d.line(pts, fill=_vivo(r), width=2)
            else:
                cx, cy, rad = (x1 + x2) // 2, (y1 + y2) // 2, min(iw, ih) // 2 - 8
                ang = 0
                for _ in range(r.randint(3, 6)):
                    a = r.randint(30, 120)
                    d.pieslice([cx - rad, cy - rad, cx + rad, cy + rad], ang, min(360, ang + a), fill=_vivo(r))
                    ang += a


def _tabla(img: Image.Image, r: random.Random, y0: int, x0: int = 0, y_fin: int | None = None) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    y_fin = y_fin or h
    d.rectangle([x0, y0, w, y_fin], fill=(255, 255, 255))
    hoja = r.random() < 0.6
    if hoja:  # cinta tipo Excel
        d.rectangle([x0, y0, w, y0 + 26], fill=(33, 115, 70) if r.random() < 0.7 else _vivo(r))
        d.text((x0 + 8, y0 + 6), r.choice(["Libro1.xlsx", "sabana.csv", "export.xlsx"]), fill=(255, 255, 255), font=fuente(11))
        y0 += 26
    alto_f = r.randint(16, 24)
    ncol = r.randint(4, 9)
    anchos = [r.randint(50, 110) for _ in range(ncol)]
    enc = _vivo(r) if r.random() < 0.6 else (230, 230, 230)
    y = y0 + 4
    fila = 0
    while y + alto_f < y_fin:
        x = x0 + (30 if hoja else 4)
        if hoja:
            d.rectangle([x0, y, x0 + 30, y + alto_f], fill=(235, 235, 235), outline=(200, 200, 200))
            d.text((x0 + 6, y + 3), str(fila + 1), fill=(90, 90, 90), font=fuente(10))
        for c, aw in enumerate(anchos):
            relleno = enc if fila == 0 else ((248, 248, 248) if fila % 2 and r.random() < 0.8 else (255, 255, 255))
            d.rectangle([x, y, x + aw, y + alto_f], fill=relleno, outline=(205, 205, 205))
            txt = r.choice(PALABRAS).upper() if fila == 0 else (
                f"{r.randint(0, 99999):,}" if r.random() < 0.6 else r.choice(PALABRAS))
            d.text((x + 4, y + 3), txt[: max(3, aw // 7)], fill=(255, 255, 255) if fila == 0 and enc != (230, 230, 230) else (40, 40, 40), font=fuente(10))
            x += aw
            if x > w:
                break
        y += alto_f
        fila += 1


def _sql(img: Image.Image, r: random.Random, y0: int) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    oscuro = r.random() < 0.6
    fondo = (r.randint(25, 45), r.randint(25, 45), r.randint(30, 50)) if oscuro else (r.randint(245, 255),) * 3
    d.rectangle([0, y0, w, h], fill=fondo)
    d.rectangle([0, y0, 34, h], fill=tuple(max(0, c - 12) for c in fondo) if oscuro else (238, 238, 238))
    resultados = r.random() < 0.3
    y_lim = int(h * r.uniform(0.55, 0.7)) if resultados else h
    y, linea = y0 + 6, 1
    tam = r.randint(10, 13)
    kw_color = r.choice([(86, 156, 214), (197, 134, 192), (0, 0, 255), (204, 120, 50)])
    while y + tam + 4 < y_lim:
        d.text((6, y), f"{linea:>3}", fill=(120, 120, 120), font=fuente(tam - 1, mono=True))
        x = 42 + r.choice([0, 0, 16, 32])
        for _ in range(r.randint(1, 5)):
            if r.random() < 0.45:
                tok, col = r.choice(SQL_KW), kw_color
            elif r.random() < 0.2:
                tok, col = f"'{r.choice(PALABRAS)}'", (206, 145, 120)
            else:
                tok, col = r.choice(PALABRAS) + r.choice(["", "_id", ",", ".total", "_dt"]), (220, 220, 220) if oscuro else (30, 30, 30)
            d.text((x, y), tok, fill=col, font=fuente(tam, mono=True))
            x += int(len(tok) * tam * 0.62) + 6
            if x > w - 40:
                break
        y += tam + 5
        linea += 1
    if resultados:
        _tabla(img, r, y_lim + 4, x0=0, y_fin=h)


def _error(img: Image.Image, r: random.Random, y0: int) -> None:
    w, h = img.size
    if r.random() < 0.35:  # traza en consola
        d = ImageDraw.Draw(img)
        d.rectangle([0, y0, w, h], fill=(r.randint(0, 30),) * 3)
        y = y0 + 6
        lineas = ["Traceback (most recent call last):", f'  File "etl_{r.choice(PALABRAS)}.py", line {r.randint(10, 400)}',
                  "    df = spark.read.table(origen)"] + [f"  at com.empresa.{r.choice(PALABRAS)}.Job.run(Job.java:{r.randint(10, 99)})" for _ in range(r.randint(2, 6))]
        lineas.append(r.choice(ERRORES))
        for i, l in enumerate(lineas * 2):
            if y > h - 14:
                break
            d.text((8, y), l, fill=(255, 85, 85) if "Error" in l or "Exception" in l or i == len(lineas) - 1 else (200, 200, 200), font=fuente(11, mono=True))
            y += 15
        return
    # ventana de dialogo, a veces encima de otra pantalla (distractor)
    if r.random() < 0.5:
        r.choice([_tablero, _tabla, _formulario])(img, r, y0)
        img.paste(Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), r.uniform(0.2, 0.5)))
    else:
        ImageDraw.Draw(img).rectangle([0, y0, w, h], fill=_color(r, True))
    d = ImageDraw.Draw(img)
    dw, dh = int(w * r.uniform(0.45, 0.8)), int((h - y0) * r.uniform(0.35, 0.6))
    x1, y1 = (w - dw) // 2 + r.randint(-30, 30), y0 + (h - y0 - dh) // 2 + r.randint(-20, 20)
    d.rectangle([x1 + 4, y1 + 4, x1 + dw + 4, y1 + dh + 4], fill=(60, 60, 60))
    d.rectangle([x1, y1, x1 + dw, y1 + dh], fill=(255, 255, 255), outline=(150, 150, 150))
    barra = r.choice([(200, 40, 40), (60, 60, 60), (0, 90, 180), (240, 240, 240)])
    d.rectangle([x1, y1, x1 + dw, y1 + 24], fill=barra)
    d.text((x1 + 8, y1 + 5), r.choice(["Error", "Error de conexion", "Excepcion", "Aviso"]),
           fill=(255, 255, 255) if barra != (240, 240, 240) else (0, 0, 0), font=fuente(12))
    cx, cy = x1 + 30, y1 + 24 + (dh - 24) // 2 - 8
    if r.random() < 0.7:
        d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(220, 40, 40))
        d.line([cx - 7, cy - 7, cx + 7, cy + 7], fill=(255, 255, 255), width=3)
        d.line([cx - 7, cy + 7, cx + 7, cy - 7], fill=(255, 255, 255), width=3)
    else:
        d.polygon([(cx, cy - 15), (cx - 15, cy + 12), (cx + 15, cy + 12)], fill=(245, 180, 0))
        d.text((cx - 3, cy - 6), "!", fill=(0, 0, 0), font=fuente(14))
    for k, linea in enumerate([r.choice(ERRORES), _frase(r, 4)]):
        d.text((x1 + 56, y1 + 34 + k * 16), linea[: dw // 7], fill=(30, 30, 30), font=fuente(11))
    bx = x1 + dw - 80
    d.rounded_rectangle([bx, y1 + dh - 30, bx + 66, y1 + dh - 8], radius=4, fill=(0, 120, 215))
    d.text((bx + 10, y1 + dh - 26), "Aceptar", fill=(255, 255, 255), font=fuente(11))


def _formulario(img: Image.Image, r: random.Random, y0: int) -> None:
    d = ImageDraw.Draw(img)
    w, h = img.size
    tema = _vivo(r)
    d.rectangle([0, y0, w, h], fill=tuple(min(255, c + 150) for c in tema))
    mx = int(w * r.uniform(0.08, 0.2))
    y = y0 + 10
    d.rectangle([mx, y, w - mx, y + 10], fill=tema)
    d.rectangle([mx, y + 10, w - mx, y + 60], fill=(255, 255, 255))
    d.text((mx + 12, y + 18), "Solicitud al Area de Datos", fill=(30, 30, 30), font=fuente(15))
    d.text((mx + 12, y + 40), "* Obligatorio", fill=(200, 0, 0), font=fuente(10))
    y += 70
    while y + 70 < h:
        d.rectangle([mx, y, w - mx, y + 64], fill=(255, 255, 255))
        d.text((mx + 12, y + 8), r.choice(["Nombre", "Correo", "Tipo de solicitud", "Descripcion",
                                            "Prioridad", "Fecha requerida", "Area"]) + " *", fill=(30, 30, 30), font=fuente(12))
        tipo = r.choice(["texto", "radio", "check"])
        if tipo == "texto":
            d.line([mx + 12, y + 48, mx + (w - 2 * mx) * 0.6, y + 48], fill=(150, 150, 150), width=1)
            d.text((mx + 12, y + 30), "Tu respuesta", fill=(160, 160, 160), font=fuente(10))
        else:
            for k in range(r.randint(2, 3)):
                bx = mx + 12 + k * 110
                if tipo == "radio":
                    d.ellipse([bx, y + 32, bx + 12, y + 44], outline=(100, 100, 100), width=2)
                else:
                    d.rectangle([bx, y + 32, bx + 12, y + 44], outline=(100, 100, 100), width=2)
                d.text((bx + 18, y + 31), r.choice(["ETL", "Sabana", "Incidencia", "Alta", "Media"]), fill=(60, 60, 60), font=fuente(10))
        y += 72
    if y + 30 < h:
        d.rounded_rectangle([mx, y, mx + 80, y + 28], radius=4, fill=tema)
        d.text((mx + 18, y + 7), "Enviar", fill=(255, 255, 255), font=fuente(12))


DIBUJANTES = {"mensaje_error": _error, "tablero_graficas": _tablero, "tabla_datos": _tabla,
              "editor_sql": _sql, "formulario": _formulario}


def generar_imagen(clase: str, r: random.Random, dificil: bool = False) -> Image.Image:
    w, h = r.randint(420, 720), r.randint(300, 520)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    y0 = _barra_navegador(ImageDraw.Draw(img), w, r) if r.random() < 0.6 else 0
    DIBUJANTES[clase](img, r, y0)
    # degradaciones
    if dificil:
        img = img.rotate(r.uniform(-6, 6), expand=True, fillcolor=(255, 255, 255))
        f = r.uniform(0.25, 0.45)
        img = img.resize((int(img.width * f), int(img.height * f))).resize((img.width, img.height))
        if r.random() < 0.5:
            img = Image.eval(img, lambda v: 255 - v)   # paleta invertida
    if r.random() < (0.6 if dificil else 0.3):
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.5 if dificil else 1.0)))
    if r.random() < 0.3:  # recorte parcial
        cx, cy = int(img.width * r.uniform(0, 0.15)), int(img.height * r.uniform(0, 0.15))
        img = img.crop((cx, cy, img.width - int(img.width * r.uniform(0, 0.15)), img.height))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=r.randint(15, 40) if dificil else r.randint(35, 95))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def generar_dataset(por_clase: int = 160, ood_por_clase: int = 30, semilla: int = 11,
                    destino: Path | None = None) -> Path:
    """Genera train/val/test (70/15/15) y un conjunto 'ood' con cambio de distribucion."""
    destino = destino or SALIDAS / "imagenes"
    r = random.Random(semilla)
    n_tr, n_va = int(por_clase * 0.70), int(por_clase * 0.15)
    for clase in CLASES:
        for i in range(por_clase + ood_por_clase):
            if i < por_clase:
                split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
                img = generar_imagen(clase, r)
            else:
                split, img = "ood", generar_imagen(clase, r, dificil=True)
            carpeta = destino / split / clase
            carpeta.mkdir(parents=True, exist_ok=True)
            img.save(carpeta / f"img_{i:05d}.png")
    return destino


if __name__ == "__main__":
    ruta = generar_dataset()
    for split in ("train", "val", "test", "ood"):
        print(split, {c: len(list((ruta / split / c).glob("*.png"))) for c in CLASES})
