"""
Preprocesamiento de texto (NLTK) para el clasificador de intenciones - Avance 2.

Etapas:
  1. Normalizacion: minusculas, sin acentos, correos -> 'correoelectronico',
     IDs de solicitud (4-6 digitos) -> 'numid', se eliminan otros numeros.
  2. Tokenizacion: nltk.word_tokenize (punkt_tab, espanol); respaldo RegexpTokenizer.
  3. Eliminacion de stopwords: lista en espanol de NLTK, conservando negaciones
     e interrogativos que cambian la intencion (no, como, cuando, donde, cuanto...).
  4. Stemming: SnowballStemmer('spanish').
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_ID = re.compile(r"(?<![\w@.])#?\d{4,6}(?![\w@])")
RE_NUM = re.compile(r"\d+")

_STEMMER = SnowballStemmer("spanish")
_TOKENIZER_RESPALDO = RegexpTokenizer(r"[a-z]+")

# Palabras de la lista de stopwords que SI aportan a la intencion.
CONSERVAR = {"no", "como", "cuando", "donde", "quien", "cual", "cuales", "cuanto",
             "cuanta", "cuantos", "cuantas", "sin", "mas",
             "estado", "estados"}  # 'estado' es participio de 'estar' en NLTK, aqui es dominio
# Respaldo si no se puede descargar el corpus de NLTK.
STOPWORDS_RESPALDO = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "una", "su", "al", "lo", "es", "o", "mi", "mis", "me", "le", "les",
    "ya", "muy", "este", "esta", "eso", "esa", "tu", "te", "si", "pero", "son", "hay",
}


def _asegurar_recursos() -> None:
    for recurso, ruta in (("stopwords", "corpora/stopwords"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(ruta)
        except LookupError:
            try:
                nltk.download(recurso, quiet=True)
            except Exception:  # sin internet: se usan los respaldos
                pass


def _quitar_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=1)
def stopwords_es() -> frozenset[str]:
    _asegurar_recursos()
    try:
        from nltk.corpus import stopwords
        base = {_quitar_acentos(w.lower()) for w in stopwords.words("spanish")}
    except LookupError:
        base = set(STOPWORDS_RESPALDO)
    return frozenset(base - CONSERVAR)


def normalizar(texto: str) -> str:
    texto = _quitar_acentos(texto.lower())
    texto = RE_EMAIL.sub(" correoelectronico ", texto)
    texto = RE_ID.sub(" numid ", texto)
    texto = RE_NUM.sub(" ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto: str) -> list[str]:
    try:
        tokens = nltk.word_tokenize(texto, language="spanish")
    except LookupError:
        tokens = _TOKENIZER_RESPALDO.tokenize(texto)
    return [t for t in tokens if t.isalpha()]


def quitar_stopwords(tokens: list[str]) -> list[str]:
    sw = stopwords_es()
    return [t for t in tokens if t not in sw and len(t) > 1]


def stemizar(tokens: list[str]) -> list[str]:
    return [_STEMMER.stem(t) for t in tokens]


def preprocesar(texto: str) -> list[str]:
    return stemizar(quitar_stopwords(tokenizar(normalizar(texto))))


def preprocesar_texto(texto: str) -> str:
    """Cadena lista para el vectorizador TF-IDF / BoW."""
    return " ".join(preprocesar(texto))


def mostrar_pasos(texto: str) -> dict[str, object]:
    """Devuelve cada etapa del preprocesamiento (util para documentar evidencias)."""
    norm = normalizar(texto)
    tokens = tokenizar(norm)
    sin_sw = quitar_stopwords(tokens)
    return {"original": texto, "normalizado": norm, "tokens": tokens,
            "sin_stopwords": sin_sw, "stems": stemizar(sin_sw)}


if __name__ == "__main__":
    for k, v in mostrar_pasos("Hola! Ya atendieron mi ticket #48301? Mi correo es ana.ruiz@empresa.com").items():
        print(f"{k:>14}: {v}")
