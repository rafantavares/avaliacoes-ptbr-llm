"""Carga, limpeza, rotulagem e divisão das avaliações da Olist.

A divisão treino/validação/teste é determinística (semente fixa) e feita
depois de remover textos duplicados: avaliações curtas como "Ótimo produto"
aparecem centenas de vezes, e deixá-las em mais de uma partição inflaria as
métricas (vazamento de dados).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROTULOS = ["negativo", "neutro", "positivo"]
SEMENTE = 42

COLUNA_TEXTO = "review_comment_message"
COLUNA_TITULO = "review_comment_title"
COLUNA_NOTA = "review_score"


def rotulo_da_nota(nota: int) -> str:
    """1–2 estrelas = negativo, 3 = neutro, 4–5 = positivo."""
    if nota <= 2:
        return "negativo"
    if nota == 3:
        return "neutro"
    return "positivo"


def limpar_texto(texto: str) -> str:
    """Normaliza quebras de linha e espaços; mantém acentos e pontuação,
    que carregam sentimento ("péssimo!!!")."""
    texto = unicodedata.normalize("NFC", str(texto))
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    return re.sub(r"\s{2,}", " ", texto).strip()


def chave_duplicata(texto: str) -> str:
    """Forma canônica usada só para detectar duplicatas: minúsculas, sem
    acento e sem pontuação ("Ótimo produto!" == "otimo produto")."""
    sem_acento = unicodedata.normalize("NFKD", texto.lower())
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", "", sem_acento).strip()


def carregar(caminho_csv: str | Path) -> pd.DataFrame:
    """Lê o CSV original e devolve `texto`, `nota` e `rotulo`, só com as
    avaliações que têm comentário, sem duplicatas de texto."""
    bruto = pd.read_csv(caminho_csv)
    titulo = bruto[COLUNA_TITULO].fillna("")
    mensagem = bruto[COLUNA_TEXTO].fillna("")
    # O título é curto mas costuma ser o resumo da opinião ("Recomendo").
    texto = (titulo + ". " + mensagem).where(titulo != "", mensagem)

    df = pd.DataFrame({"texto": texto.map(limpar_texto), "nota": bruto[COLUNA_NOTA]})
    df = df[df["texto"].str.len() >= 3]

    # Um mesmo texto com notas diferentes é ambíguo: fica a nota mais comum.
    df["chave"] = df["texto"].map(chave_duplicata)
    df = df[df["chave"] != ""]
    nota_mais_comum = df.groupby("chave")["nota"].agg(lambda s: s.mode().iloc[0])
    df = df.drop_duplicates("chave").copy()
    df["nota"] = df["chave"].map(nota_mais_comum)

    df["rotulo"] = df["nota"].map(rotulo_da_nota)
    return df.drop(columns="chave").reset_index(drop=True)


def dividir(df: pd.DataFrame, teste: float = 0.15, validacao: float = 0.15):
    """Divisão estratificada pelo rótulo: (treino, validacao, teste)."""
    treino_val, df_teste = train_test_split(
        df, test_size=teste, stratify=df["rotulo"], random_state=SEMENTE
    )
    proporcao_val = validacao / (1 - teste)
    df_treino, df_val = train_test_split(
        treino_val, test_size=proporcao_val, stratify=treino_val["rotulo"], random_state=SEMENTE
    )
    return (
        df_treino.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_teste.reset_index(drop=True),
    )


def impressao_digital(df: pd.DataFrame) -> str:
    """Hash do conteúdo de uma partição. Permite conferir que o notebook no
    Kaggle avaliou exatamente o mesmo conjunto de teste que os demais modelos."""
    conteudo = "\n".join(df["texto"] + "\t" + df["rotulo"])
    return hashlib.sha256(conteudo.encode("utf-8")).hexdigest()[:16]


def preparar(caminho_csv: str | Path, pasta_saida: str | Path) -> dict:
    """Gera `treino.csv`, `validacao.csv` e `teste.csv` — os mesmos arquivos
    usados pelo baseline, pelo notebook de fine-tuning e pela avaliação do LLM."""
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    partes = dict(zip(["treino", "validacao", "teste"], dividir(carregar(caminho_csv))))
    for nome, parte in partes.items():
        parte[["texto", "nota", "rotulo"]].to_csv(pasta / f"{nome}.csv", index=False)
    return {
        "tamanhos": {nome: len(parte) for nome, parte in partes.items()},
        "impressao_teste": impressao_digital(partes["teste"]),
    }


if __name__ == "__main__":
    import sys

    origem = sys.argv[1] if len(sys.argv) > 1 else "data/raw/olist_order_reviews_dataset.csv"
    print(preparar(origem, "data/processed"))
