import pandas as pd
import pytest

from avaliacoes import llm_zero_shot as mod


@pytest.mark.parametrize("texto, esperado", [
    ('{"sentimento": "neutro"}', "neutro"),
    ('{"sentimento": "talvez"}', None),     # fora do enum
    ('{"outro": "positivo"}', None),        # chave errada
    ("não é json", None),
    ("", None),
])
def test_extrair_rotulo(texto, esperado):
    assert mod.extrair_rotulo(texto) == esperado


def test_amostra_estratificada_mantem_proporcao():
    teste = pd.DataFrame({
        "texto": [f"t{i}" for i in range(1000)],
        "rotulo": ["positivo"] * 600 + ["negativo"] * 300 + ["neutro"] * 100,
    })
    amostra = mod.amostra_estratificada(teste, 100)
    assert len(amostra) == 100
    assert amostra["rotulo"].value_counts()["neutro"] == 10
    assert amostra.equals(mod.amostra_estratificada(teste, 100))  # determinística


def test_carregar_env_nao_sobrescreve(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comentário\nCHAVE_A=\"valor-a\"\nCHAVE_B=valor-b\n", encoding="utf-8")
    monkeypatch.setenv("CHAVE_B", "ja-definida")
    monkeypatch.delenv("CHAVE_A", raising=False)
    mod.carregar_env(env)
    import os
    assert os.environ["CHAVE_A"] == "valor-a"
    assert os.environ["CHAVE_B"] == "ja-definida"


class ProvedorFalso:
    """Simula um provedor: erra os neutros e devolve uma resposta inválida."""

    def __init__(self, modelos):
        self.chamadas = 0

    def classificar(self, texto):
        self.chamadas += 1
        rotulo = {"bom": "positivo", "ruim": "negativo", "meh": "positivo"}.get(texto)
        return {"rotulo": rotulo, "modelo": "falso-1", "latencia_s": 0.5,
                "tokens_entrada": 100, "tokens_saida": 10}


def test_avaliar_llm_ponta_a_ponta_com_provedor_falso(tmp_path, monkeypatch):
    pasta = tmp_path / "dados"
    pasta.mkdir()
    pd.DataFrame({
        "texto": ["bom"] * 6 + ["ruim"] * 3 + ["meh"],
        "nota": 0,
        "rotulo": ["positivo"] * 6 + ["negativo"] * 3 + ["neutro"],
    }).to_csv(pasta / "teste.csv", index=False)
    monkeypatch.setitem(mod.PROVEDORES, "falso", (ProvedorFalso, ["falso-1"]))
    monkeypatch.chdir(tmp_path)

    r = mod.avaliar_llm("falso", n=10, pasta_dados=pasta)

    assert r["acuracia"] == 0.9
    assert r["f1_por_classe"]["neutro"] == 0
    assert r["tokens_por_1k"] == {"entrada": 100_000, "saida": 10_000}
    assert r["modelos_que_responderam"] == {"falso-1": 10}
    assert (tmp_path / "results" / "zero_shot_falso-1.json").exists()
