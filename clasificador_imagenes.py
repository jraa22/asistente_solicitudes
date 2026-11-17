"""
Clasificador de capturas de pantalla con CNN y transferencia de aprendizaje - Avance 2.

Arquitectura: EfficientNet-B0 preentrenada en ImageNet (torchvision). Alternativa: ResNet50
(--arquitectura resnet50). Se reemplaza la cabeza por Dropout(0.3) + Linear(n_clases).

Entrenamiento en dos fases:
  Fase 1 (extraccion de caracteristicas): backbone congelado, solo se entrena la cabeza.
  Fase 2 (ajuste fino): se descongelan los ultimos bloques convolucionales con tasa de
          aprendizaje menor. Se conserva el modelo con mejor accuracy de validacion.

Evaluacion en 'test' (misma distribucion) y 'ood' (cambio de distribucion): accuracy,
precision, recall, F1 (macro y por clase) y matriz de confusion.

Uso:  python clasificador_imagenes.py [--arquitectura efficientnet_b0|resnet50]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

SALIDAS = Path(os.environ.get("ASISTENTE_SALIDAS", Path(__file__).resolve().parent / "salidas"))
RUTA_MODELO = SALIDAS / "modelo_imagenes.pt"
MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TAM = 224

TRANSF_ENTRENAMIENTO = transforms.Compose([
    transforms.Resize((TAM, TAM)),
    transforms.RandomResizedCrop(TAM, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESV),
])
TRANSF_EVALUACION = transforms.Compose([
    transforms.Resize((TAM, TAM)), transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)])


def construir_modelo(n_clases: int, arquitectura: str = "efficientnet_b0", pesos: bool = True) -> nn.Module:
    if arquitectura == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pesos else None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_clases))
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pesos else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, n_clases))
    return m


def _cabeza(m: nn.Module, arquitectura: str) -> nn.Module:
    return m.fc if arquitectura == "resnet50" else m.classifier


def _ultimos_bloques(m: nn.Module, arquitectura: str) -> list[nn.Module]:
    return [m.layer4] if arquitectura == "resnet50" else list(m.features[-3:])


def _congelar(m: nn.Module, arquitectura: str, fase: int) -> list[nn.Parameter]:
    for p in m.parameters():
        p.requires_grad = False
    entrenables = [_cabeza(m, arquitectura)] + (_ultimos_bloques(m, arquitectura) if fase == 2 else [])
    for bloque in entrenables:
        for p in bloque.parameters():
            p.requires_grad = True
    return [p for p in m.parameters() if p.requires_grad]


def _epoca(m, cargador, criterio, optim=None) -> tuple[float, float]:
    entrenar = optim is not None
    m.train(entrenar)
    perdida, aciertos, n = 0.0, 0, 0
    with torch.set_grad_enabled(entrenar):
        for x, y in cargador:
            salida = m(x)
            loss = criterio(salida, y)
            if entrenar:
                optim.zero_grad()
                loss.backward()
                optim.step()
            perdida += loss.item() * len(y)
            aciertos += (salida.argmax(1) == y).sum().item()
            n += len(y)
    return perdida / n, aciertos / n


@torch.no_grad()
def _predecir(m, cargador) -> tuple[np.ndarray, np.ndarray]:
    m.eval()
    ys, ps = [], []
    for x, y in cargador:
        ps.append(m(x).argmax(1).numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluar(m, carpeta: Path, clases: list[str], nombre: str) -> dict:
    ds = datasets.ImageFolder(carpeta, transform=TRANSF_EVALUACION)
    assert ds.classes == clases, "Las clases no coinciden"
    y, p = _predecir(m, DataLoader(ds, batch_size=32))
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    mc = confusion_matrix(y, p, labels=range(len(clases)))
    print(f"\n=== {nombre} ({len(y)} imagenes) ===")
    print(f"accuracy={accuracy_score(y, p):.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print(classification_report(y, p, target_names=clases, digits=3, zero_division=0))
    print("Matriz de confusion (filas=real, columnas=prediccion):\n" + _texto_matriz(mc, clases))
    _graficar_matriz(mc, clases, SALIDAS / f"matriz_confusion_imagenes_{nombre}.png", f"Imagenes - {nombre}")
    return {"accuracy": accuracy_score(y, p), "precision_macro": prec, "recall_macro": rec,
            "f1_macro": f1, "matriz_confusion": mc.tolist()}


def entrenar(arquitectura: str = "efficientnet_b0", epocas_f1: int = 5, epocas_f2: int = 3,
             datos: Path | None = None, semilla: int = 0) -> dict:
    torch.manual_seed(semilla)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    datos = datos or SALIDAS / "imagenes"
    ds_tr = datasets.ImageFolder(datos / "train", transform=TRANSF_ENTRENAMIENTO)
    ds_va = datasets.ImageFolder(datos / "val", transform=TRANSF_EVALUACION)
    clases = ds_tr.classes
    tr = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=0)
    va = DataLoader(ds_va, batch_size=32)
    print(f"{arquitectura} | clases={clases} | train={len(ds_tr)} val={len(ds_va)}")

    m = construir_modelo(len(clases), arquitectura)
    criterio = nn.CrossEntropyLoss(label_smoothing=0.05)
    mejor_acc, mejor_estado, historia = -1.0, None, []
    plan = [(1, e, 1e-3) for e in range(epocas_f1)] + [(2, e, 1e-4) for e in range(epocas_f2)]
    optim, fase_actual = None, 0
    for fase, e, lr in plan:
        if fase != fase_actual:
            params = _congelar(m, arquitectura, fase)
            optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            fase_actual = fase
            print(f"-- Fase {fase}: {sum(p.numel() for p in params):,} parametros entrenables")
        t = time.time()
        l_tr, a_tr = _epoca(m, tr, criterio, optim)
        l_va, a_va = _epoca(m, va, criterio)
        historia.append({"fase": fase, "epoca": len(historia) + 1, "loss_train": l_tr, "acc_train": a_tr,
                         "loss_val": l_va, "acc_val": a_va})
        print(f"Fase {fase} epoca {len(historia)}: loss {l_tr:.3f}/{l_va:.3f} acc {a_tr:.3f}/{a_va:.3f} ({time.time() - t:.0f}s)")
        if a_va > mejor_acc:
            mejor_acc, mejor_estado = a_va, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(mejor_estado)
    SALIDAS.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": m.state_dict(), "clases": clases, "arquitectura": arquitectura}, RUTA_MODELO)

    resultados = {"arquitectura": arquitectura, "historia": historia, "mejor_acc_val": mejor_acc,
                  "test": evaluar(m, datos / "test", clases, "test")}
    if (datos / "ood").exists():
        resultados["ood"] = evaluar(m, datos / "ood", clases, "ood")
    (SALIDAS / "metricas_imagenes.json").write_text(json.dumps(resultados, indent=2))
    return resultados


class ClasificadorImagenes:
    """Inferencia: imagen (ruta o PIL) -> clase, probabilidad y top-3."""

    def __init__(self, ruta: Path | str = RUTA_MODELO) -> None:
        datos = torch.load(ruta, map_location="cpu", weights_only=False)
        self.clases = datos["clases"]
        self.modelo = construir_modelo(len(self.clases), datos["arquitectura"], pesos=False)
        self.modelo.load_state_dict(datos["estado"])
        self.modelo.eval()

    @torch.no_grad()
    def predecir(self, imagen: str | Path | Image.Image) -> dict:
        img = imagen if isinstance(imagen, Image.Image) else Image.open(imagen)
        x = TRANSF_EVALUACION(img.convert("RGB")).unsqueeze(0)
        proba = torch.softmax(self.modelo(x), dim=1)[0].numpy()
        orden = np.argsort(proba)[::-1][:3]
        return {"clase": self.clases[orden[0]], "probabilidad": float(proba[orden[0]]),
                "top3": [(self.clases[i], float(proba[i])) for i in orden]}


def _texto_matriz(mc: np.ndarray, clases: list[str]) -> str:
    cortas = [c[:10] for c in clases]
    filas = [" " * 18 + "".join(f"{c:>11}" for c in cortas)]
    filas += [f"{clases[i]:>18}" + "".join(f"{v:>11}" for v in fila) for i, fila in enumerate(mc)]
    return "\n".join(filas)


def _graficar_matriz(mc: np.ndarray, clases: list[str], ruta: Path, titulo: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mc, cmap="Blues")
    ax.set_xticks(range(len(clases)), clases, rotation=45, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, mc[i, j], ha="center", va="center", color="white" if mc[i, j] > mc.max() / 2 else "black")
    ax.set_xlabel("Prediccion"), ax.set_ylabel("Real"), ax.set_title(titulo)
    fig.tight_layout(), fig.savefig(ruta, dpi=120), plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquitectura", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    ap.add_argument("--epocas-f1", type=int, default=5)
    ap.add_argument("--epocas-f2", type=int, default=3)
    args = ap.parse_args()
    if not (SALIDAS / "imagenes" / "train").exists():
        from dataset_imagenes import generar_dataset
        generar_dataset()
    entrenar(args.arquitectura, args.epocas_f1, args.epocas_f2)
