"""Métricas comuns a todas as abordagens, para que a comparação seja justa:
mesmas classes, mesma ordem, mesmo conjunto de teste."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from avaliacoes.dados import ROTULOS


def avaliar(verdadeiros, previstos, *, nome: str, extras: dict | None = None) -> dict:
    """F1-macro é a métrica principal: as classes são desbalanceadas
    (~62% positivas, ~9% neutras) e a acurácia esconderia um modelo que ignora o neutro."""
    relatorio = classification_report(
        verdadeiros, previstos, labels=ROTULOS, output_dict=True, zero_division=0
    )
    return {
        "modelo": nome,
        "n_teste": len(verdadeiros),
        "f1_macro": round(f1_score(verdadeiros, previstos, labels=ROTULOS, average="macro", zero_division=0), 4),
        "acuracia": round(accuracy_score(verdadeiros, previstos), 4),
        "f1_por_classe": {r: round(relatorio[r]["f1-score"], 4) for r in ROTULOS},
        "matriz_confusao": {
            "classes": ROTULOS,
            "valores": confusion_matrix(verdadeiros, previstos, labels=ROTULOS).tolist(),
        },
        **(extras or {}),
    }


def salvar(resultado: dict, pasta: str | Path = "results") -> Path:
    caminho = Path(pasta) / f"{resultado['modelo']}.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminho
