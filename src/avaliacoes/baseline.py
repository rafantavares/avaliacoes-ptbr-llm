"""Baseline clássico: TF-IDF (palavras + n-gramas de caracteres) com
regressão logística. Roda em segundos na CPU e define o piso que os
modelos maiores precisam superar para justificar o custo."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from avaliacoes.metricas import avaliar, salvar


def criar_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("palavras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            # n-gramas de caracteres pegam erros de digitação ("pessimo", "pésimo").
            ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                           sublinear_tf=True)),
        ])),
        # class_weight compensa o desbalanceamento (poucas avaliações neutras).
        ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")),
    ])


def treinar_e_avaliar(pasta_dados: str | Path = "data/processed",
                      pasta_modelo: str | Path = "models") -> dict:
    pasta = Path(pasta_dados)
    treino = pd.read_csv(pasta / "treino.csv")
    teste = pd.read_csv(pasta / "teste.csv")

    pipeline = criar_pipeline()
    inicio = time.perf_counter()
    pipeline.fit(treino["texto"], treino["rotulo"])
    segundos_treino = time.perf_counter() - inicio

    inicio = time.perf_counter()
    previstos = pipeline.predict(teste["texto"])
    ms_por_amostra = (time.perf_counter() - inicio) * 1000 / len(teste)

    Path(pasta_modelo).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, Path(pasta_modelo) / "baseline_tfidf_logreg.joblib")

    resultado = avaliar(teste["rotulo"], previstos, nome="tfidf_logreg", extras={
        "segundos_treino": round(segundos_treino, 1),
        "ms_por_amostra": round(ms_por_amostra, 3),
        "hardware": "CPU",
        "custo_usd_por_1k": 0.0,
    })
    salvar(resultado)
    return resultado


if __name__ == "__main__":
    import json

    print(json.dumps(treinar_e_avaliar(), ensure_ascii=False, indent=2))
