"""
Representacion (BoW / TF-IDF) y clasificacion de intenciones (Naive Bayes / SVM) - Avance 2.

Proceso:
  1. Genera el dataset sintetico (dataset_intenciones.py) y lo preprocesa (preprocesamiento.py).
  2. Compara 4 modelos con validacion cruzada estratificada POR PLANTILLA (StratifiedGroupKFold),
     para medir el desempeno en estructuras de frase no vistas:
        NB + BoW, NB + TF-IDF, SVM lineal + TF-IDF (palabras) y SVM lineal + TF-IDF (palabras + caracteres).
  3. Evalua el mejor en un conjunto de prueba retenido por plantillas (reporte y matriz de confusion).
  4. Entrena el modelo final con todos los datos, lo calibra para obtener probabilidades y lo
     evalua en la prueba retadora (frases escritas a mano), comparandolo con el chatbot de reglas v1.
  5. Guarda el modelo en salidas/modelo_intenciones.joblib.

Uso:  python clasificador_intenciones.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from dataset_intenciones import generar_dataset, prueba_retadora
from preprocesamiento import normalizar, preprocesar_texto

SALIDAS = Path(os.environ.get("ASISTENTE_SALIDAS", Path(__file__).resolve().parent / "salidas"))
RUTA_MODELO = SALIDAS / "modelo_intenciones.joblib"
UMBRAL_CONFIANZA = 0.30   # por debajo, el asistente pide reformular


def _separar(texto: str) -> list[str]:
    return texto.split()


def modelos_candidatos() -> dict[str, Pipeline]:
    """Todos reciben texto crudo; el preprocesamiento NLTK va dentro del vectorizador.

    Canal de palabras: normalizacion + tokenizacion + stopwords + stemming, n-gramas (1,2).
    Canal de caracteres: n-gramas (2,5) sobre el texto normalizado (robusto a errores de dedo).
    """
    palabras = dict(preprocessor=preprocesar_texto, tokenizer=_separar, token_pattern=None, ngram_range=(1, 2))
    caracteres = dict(preprocessor=normalizar, analyzer="char_wb", ngram_range=(2, 5))
    return {
        "NB + BoW": Pipeline([("vec", CountVectorizer(**palabras)),
                              ("clf", MultinomialNB(alpha=0.3))]),
        "NB + TF-IDF": Pipeline([("vec", TfidfVectorizer(**palabras, sublinear_tf=True)),
                                 ("clf", MultinomialNB(alpha=0.1))]),
        "SVM + TF-IDF palabras": Pipeline([("vec", TfidfVectorizer(**palabras, sublinear_tf=True)),
                                           ("clf", LinearSVC(C=1.0, class_weight="balanced"))]),
        "SVM + TF-IDF palabras+caracteres": Pipeline([
            ("vec", FeatureUnion([
                ("palabras", TfidfVectorizer(**palabras, sublinear_tf=True)),
                ("caracteres", TfidfVectorizer(**caracteres, sublinear_tf=True)),
            ])),
            ("clf", LinearSVC(C=0.8, class_weight="balanced")),
        ]),
    }


class ClasificadorIntenciones:
    """Envoltura de inferencia: texto crudo -> (intencion, probabilidad, top-3)."""

    def __init__(self, ruta: Path | str = RUTA_MODELO) -> None:
        datos = joblib.load(ruta)
        self.modelo = datos["modelo"]
        self.umbral = datos.get("umbral", UMBRAL_CONFIANZA)
        self.clases = list(self.modelo.classes_)

    def predecir(self, texto: str) -> tuple[str, float, list[tuple[str, float]]]:
        proba = self.modelo.predict_proba([texto])[0]
        orden = np.argsort(proba)[::-1][:3]
        top = [(self.clases[i], float(proba[i])) for i in orden]
        return top[0][0], top[0][1], top


def _intencion_reglas(texto: str) -> str:
    """Clasificacion del chatbot de reglas v1 (para comparar)."""
    from chatbot_reglas import ChatbotSolicitudes
    bot = ChatbotSolicitudes()
    bot.responder(texto)
    return "fuera_dominio" if bot.ultima_intencion == "fallback" else bot.ultima_intencion


def entrenar_y_evaluar(semilla: int = 7) -> dict:
    SALIDAS.mkdir(parents=True, exist_ok=True)
    df = generar_dataset(semilla=semilla)
    X, y, g = df.texto.values, df.intencion.values, df.grupo.values
    print(f"Dataset: {len(df)} ejemplos | {df.intencion.nunique()} intenciones | {df.grupo.nunique()} plantillas")

    # 1) Comparacion de modelos: CV aleatoria estratificada (optimista: comparte plantillas)
    #    y CV por plantilla (pesimista: estructuras de frase nunca vistas). Se elige por la segunda.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    cv_aleatoria = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
    comparacion = []
    for nombre, modelo in modelos_candidatos().items():
        ra = cross_validate(modelo, X, y, cv=cv_aleatoria, scoring=["accuracy", "f1_macro"])
        r = cross_validate(modelo, X, y, groups=g, cv=cv, scoring=["accuracy", "f1_macro"])
        comparacion.append({"modelo": nombre,
                            "acc_aleatoria": ra["test_accuracy"].mean(), "f1_aleatoria": ra["test_f1_macro"].mean(),
                            "acc_plantilla": r["test_accuracy"].mean(), "f1_plantilla": r["test_f1_macro"].mean(),
                            "f1_plantilla_std": r["test_f1_macro"].std()})
    comp = pd.DataFrame(comparacion).sort_values("f1_plantilla", ascending=False)
    print("\nValidacion cruzada 5 folds (aleatoria vs por plantilla):\n" + comp.round(3).to_string(index=False))
    mejor = comp.iloc[0]["modelo"]

    # 2) Predicciones fuera de muestra por plantilla (cada ejemplo se predice con un modelo
    #    que no vio su plantilla) -> reporte por clase y matriz de confusion del mejor modelo
    pred = cross_val_predict(modelos_candidatos()[mejor], X, y, groups=g, cv=cv)
    etiquetas = sorted(set(y))
    reporte = classification_report(y, pred, labels=etiquetas, zero_division=0, digits=3)
    print(f"\nReporte fuera de muestra por plantilla ({len(y)} ejemplos) - {mejor}:\n{reporte}")
    mc = pd.DataFrame(confusion_matrix(y, pred, labels=etiquetas), index=etiquetas, columns=etiquetas)
    mc.to_csv(SALIDAS / "matriz_confusion_intenciones.csv")
    _graficar_matriz(mc, SALIDAS / "matriz_confusion_intenciones.png", f"Intenciones - {mejor}")

    # 3) Modelo final calibrado (probabilidades) con todos los datos
    final = CalibratedClassifierCV(modelos_candidatos()[mejor], cv=3, method="sigmoid").fit(X, y)
    joblib.dump({"modelo": final, "nombre": mejor, "umbral": UMBRAL_CONFIANZA}, RUTA_MODELO)

    # 4) Prueba retadora: modelo ML vs reglas v1
    ret = prueba_retadora()
    proba = final.predict_proba(ret.texto.values)
    ret["pred_ml"] = final.classes_[proba.argmax(1)]
    ret["confianza"] = proba.max(1)
    ret["pred_reglas"] = ret.texto.map(_intencion_reglas)
    acc_ml = accuracy_score(ret.intencion, ret.pred_ml)
    acc_rg = accuracy_score(ret.intencion, ret.pred_reglas)
    print(f"\nPrueba retadora ({len(ret)} frases escritas a mano):")
    print(f"  Reglas v1 : accuracy={acc_rg:.3f}  F1-macro={f1_score(ret.intencion, ret.pred_reglas, average='macro'):.3f}")
    print(f"  {mejor}: accuracy={acc_ml:.3f}  F1-macro={f1_score(ret.intencion, ret.pred_ml, average='macro'):.3f}")
    errores = ret[ret.intencion != ret.pred_ml]
    print("\nErrores del modelo ML en la prueba retadora:\n" +
          errores[["texto", "intencion", "pred_ml", "confianza"]].round(2).to_string(index=False))
    ret.to_csv(SALIDAS / "prueba_retadora_resultados.csv", index=False)

    resumen = {"mejor_modelo": mejor, "n_ejemplos": int(len(df)), "comparacion_cv": comp.round(4).to_dict("records"),
               "plantilla_accuracy": float(accuracy_score(y, pred)),
               "plantilla_f1_macro": float(f1_score(y, pred, average="macro")),
               "retadora_accuracy_ml": float(acc_ml), "retadora_accuracy_reglas": float(acc_rg)}
    (SALIDAS / "metricas_intenciones.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    return resumen


def _solo_normalizar(texto: str) -> str:
    from preprocesamiento import tokenizar
    return " ".join(tokenizar(normalizar(texto)))


def _sin_stopwords(texto: str) -> str:
    from preprocesamiento import quitar_stopwords, tokenizar
    return " ".join(quitar_stopwords(tokenizar(normalizar(texto))))


def ablacion_preprocesamiento(semilla: int = 7) -> pd.DataFrame:
    """Efecto de cada etapa del preprocesamiento en el canal de palabras (SVM palabras+caracteres)."""
    df = generar_dataset(semilla=semilla)
    ret = prueba_retadora()
    X, y, g = df.texto.values, df.intencion.values, df.grupo.values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    filas = []
    for nombre, prep in [("normalizacion + tokenizacion", _solo_normalizar),
                         ("+ eliminacion de stopwords", _sin_stopwords),
                         ("+ stemming (pipeline completo)", preprocesar_texto)]:
        modelo = Pipeline([("vec", FeatureUnion([
            ("palabras", TfidfVectorizer(preprocessor=prep, tokenizer=_separar, token_pattern=None,
                                         ngram_range=(1, 2), sublinear_tf=True)),
            ("caracteres", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                           ngram_range=(2, 5), sublinear_tf=True))])),
            ("clf", LinearSVC(C=0.8, class_weight="balanced"))])
        r = cross_validate(modelo, X, y, groups=g, cv=cv, scoring="f1_macro")
        modelo.fit(X, y)
        filas.append({"preprocesamiento": nombre, "f1_plantilla": r["test_score"].mean(),
                      "acc_retadora": accuracy_score(ret.intencion, modelo.predict(ret.texto.values))})
    tabla = pd.DataFrame(filas)
    print("\nAblacion del preprocesamiento:\n" + tabla.round(3).to_string(index=False))
    return tabla


def _graficar_matriz(mc: pd.DataFrame, ruta: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(mc.values, cmap="Blues")
    ax.set_xticks(range(len(mc.columns)), mc.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(mc.index)), mc.index, fontsize=8)
    for i in range(mc.shape[0]):
        for j in range(mc.shape[1]):
            if mc.values[i, j]:
                ax.text(j, i, mc.values[i, j], ha="center", va="center", fontsize=7,
                        color="white" if mc.values[i, j] > mc.values.max() / 2 else "black")
    ax.set_xlabel("Prediccion"), ax.set_ylabel("Real"), ax.set_title(titulo)
    fig.colorbar(im), fig.tight_layout(), fig.savefig(ruta, dpi=120), plt.close(fig)


if __name__ == "__main__":
    entrenar_y_evaluar()
    ablacion_preprocesamiento()
