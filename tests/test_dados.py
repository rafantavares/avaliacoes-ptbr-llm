import pandas as pd
import pytest

from avaliacoes.dados import (
    carregar,
    chave_duplicata,
    dividir,
    impressao_digital,
    limpar_texto,
    rotulo_da_nota,
)


@pytest.mark.parametrize("nota, esperado", [
    (1, "negativo"), (2, "negativo"), (3, "neutro"), (4, "positivo"), (5, "positivo"),
])
def test_rotulo_da_nota(nota, esperado):
    assert rotulo_da_nota(nota) == esperado


def test_limpar_texto_normaliza_espacos_e_mantem_pontuacao():
    assert limpar_texto("  Péssimo!!!\r\n\r\nNão  recomendo ") == "Péssimo!!! Não recomendo"


def test_chave_duplicata_ignora_acento_caixa_e_pontuacao():
    assert chave_duplicata("Ótimo produto!") == chave_duplicata("otimo PRODUTO")


def _csv_olist(tmp_path, linhas):
    df = pd.DataFrame(linhas, columns=["review_comment_title", "review_comment_message", "review_score"])
    caminho = tmp_path / "reviews.csv"
    df.to_csv(caminho, index=False)
    return caminho


def test_carregar_descarta_vazios_e_duplicatas(tmp_path):
    caminho = _csv_olist(tmp_path, [
        (None, "Ótimo produto", 5),
        (None, "otimo produto!", 5),     # duplicata após normalização
        (None, None, 4),                 # sem comentário
        (None, "ok", 3),                 # curto demais (< 3 caracteres)
        ("Recomendo", "Chegou antes do prazo", 5),
    ])
    df = carregar(caminho)
    assert len(df) == 2
    assert "Recomendo. Chegou antes do prazo" in df["texto"].tolist()


def test_carregar_resolve_texto_com_notas_conflitantes_pela_moda(tmp_path):
    caminho = _csv_olist(tmp_path, [
        (None, "Não recebi", 1),
        (None, "não recebi", 1),
        (None, "Não recebi.", 4),
    ])
    df = carregar(caminho)
    assert len(df) == 1
    assert df.loc[0, "rotulo"] == "negativo"


def _df_sintetico(n=600):
    rotulos = ["positivo"] * int(n * 0.6) + ["negativo"] * int(n * 0.3) + ["neutro"] * int(n * 0.1)
    return pd.DataFrame({"texto": [f"avaliação {i}" for i in range(len(rotulos))],
                         "nota": 0, "rotulo": rotulos})


def test_dividir_sem_vazamento_e_estratificado():
    treino, val, teste = dividir(_df_sintetico())
    textos = [set(p["texto"]) for p in (treino, val, teste)]
    assert not (textos[0] & textos[1] or textos[0] & textos[2] or textos[1] & textos[2])
    for parte in (treino, val, teste):
        assert parte["rotulo"].value_counts(normalize=True)["neutro"] == pytest.approx(0.1, abs=0.02)


def test_dividir_e_deterministico():
    df = _df_sintetico()
    assert impressao_digital(dividir(df)[2]) == impressao_digital(dividir(df)[2])
