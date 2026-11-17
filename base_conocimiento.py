"""
Base de conocimiento ampliada - Avance 2.

Genera tickets ficticios (work items simulados de Azure DevOps) con distintas categorias,
descripciones, estados, analistas y comentarios, ademas de los 5 tickets de demostracion
de la v1. Los tickets se agregan al diccionario WORK_ITEMS que usa el chatbot.
En la version final este modulo se reemplazara por la API REST de Azure DevOps.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

from chatbot_reglas import WORK_ITEMS

USUARIOS = ["maria.lopez", "juan.perez", "ana.ruiz", "luis.garcia", "carla.mendez", "pedro.soto",
            "sofia.reyes", "diego.castro", "laura.vega", "jorge.nunez", "paola.rios", "miguel.ortiz"]
EQUIPOS = {"ETL / Ingesta": "Equipo de Ingesta", "Sabana de datos": "Equipo de Explotacion",
           "Incidencia": "Equipo de Soporte", "Accesos": "Equipo de Gobierno de Datos",
           "Modificacion de proceso": "Equipo de Ingesta"}
TABLAS = ["ventas_diarias", "inventarios", "clientes", "cartera", "proveedores", "cobranza",
          "nomina", "creditos", "pedidos", "sucursales"]
TITULOS = {
    "ETL / Ingesta": ["ETL de tabla {t} hacia el Lakehouse", "Ingesta incremental de {t} desde {o}",
                      "Migracion de {t} de {o} a Databricks"],
    "Sabana de datos": ["Sabana de {t} a detalle (XLSX)", "Extraccion CSV de {t} del ultimo trimestre",
                        "Base completa de {t} por region"],
    "Incidencia": ["Incidencia: tablero de {t} sin datos del dia", "Incidencia: cifras de {t} duplicadas",
                   "Incidencia: falla en la carga de {t}"],
    "Accesos": ["Permiso de lectura al esquema {t}", "Alta de usuario en grupo {t}_lectura"],
    "Modificacion de proceso": ["Agregar columnas al proceso de {t}", "Cambiar frecuencia de carga de {t}"],
}
ORIGENES = ["SQL Server", "Oracle", "SAP", "PostgreSQL", "SFTP", "Salesforce"]
COMENTARIOS = ["Se valido el acceso a la base origen.", "Pipeline en pruebas en ambiente QA.",
               "Se solicito al usuario confirmar los filtros.", "En espera de autorizacion del data owner.",
               "Se identifico la causa raiz en la fuente.", "Se programo la entrega para esta semana."]
RESOLUCIONES = {"ETL / Ingesta": "Tabla publicada en el Lakehouse con carga diaria.",
                "Sabana de datos": "Archivo entregado en la carpeta compartida /sabanas/{id}.",
                "Incidencia": "Se corrigio la carga y se reprocesaron los datos.",
                "Accesos": "Permiso otorgado al grupo solicitado.",
                "Modificacion de proceso": "Cambio liberado a produccion."}
PESOS_TIPO = {"ETL / Ingesta": 0.30, "Sabana de datos": 0.25, "Incidencia": 0.17, "Accesos": 0.15,
              "Modificacion de proceso": 0.13}
PESOS_ESTADO = {"New": 0.12, "Active": 0.28, "Blocked": 0.08, "Resolved": 0.17, "Closed": 0.35}


def generar_tickets(n: int = 300, semilla: int = 3, id_inicial: int = 49000, hoy: date = date(2026, 11, 16)) -> dict[int, dict]:
    r = random.Random(semilla)
    tickets = {}
    for k in range(n):
        tipo = r.choices(list(PESOS_TIPO), weights=list(PESOS_TIPO.values()))[0]
        estado = r.choices(list(PESOS_ESTADO), weights=list(PESOS_ESTADO.values()))[0]
        alta = hoy - timedelta(days=r.randint(1, 120))
        wid = id_inicial + k
        n_com = 0 if estado == "New" else r.choice([0, 0, 1, 1, 2, 3])
        comentarios = [f"{alta + timedelta(days=i + 1)} - Analista: {r.choice(COMENTARIOS)}" for i in range(n_com)]
        tickets[wid] = {
            "titulo": r.choice(TITULOS[tipo]).format(t=r.choice(TABLAS), o=r.choice(ORIGENES)),
            "tipo": tipo, "estado": estado,
            "solicitante": f"{r.choice(USUARIOS)}@empresa.com",
            "asignado": "Sin asignar" if estado == "New" else EQUIPOS[tipo],
            "alta": str(alta), "actualizado": str(min(hoy, alta + timedelta(days=r.randint(0, 20)))),
            "comentarios": comentarios,
            "resolucion": RESOLUCIONES[tipo].format(id=wid) if estado in ("Resolved", "Closed") else None,
        }
    return tickets


def cargar_base(n: int = 300) -> dict[int, dict]:
    """Agrega los tickets ficticios a WORK_ITEMS (conserva los 5 de demostracion)."""
    for wid, t in generar_tickets(n).items():
        WORK_ITEMS.setdefault(wid, t)
    return WORK_ITEMS


def resumen() -> pd.DataFrame:
    df = pd.DataFrame.from_dict(WORK_ITEMS, orient="index")
    return df.groupby(["tipo", "estado"]).size().unstack(fill_value=0)


if __name__ == "__main__":
    cargar_base()
    print(f"Tickets en la base: {len(WORK_ITEMS)}")
    print(resumen())
