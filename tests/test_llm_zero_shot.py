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


def test_gemini_tenta_de_novo_em_timeout_e_troca_de_modelo(monkeypatch):
    import httpx

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    gemini = mod.Gemini.__new__(mod.Gemini)
    gemini.modelos = ["principal", "reserva"]
    chamadas = []

    def chamar(modelo, texto):
        chamadas.append(modelo)
        if modelo == "principal":
            raise httpx.ReadTimeout("lento")
        return {"rotulo": "positivo", "modelo": modelo}

    gemini._chamar = chamar
    assert gemini.classificar("bom")["modelo"] == "reserva"
    assert chamadas == ["principal"] * mod.TENTATIVAS_POR_MODELO + ["reserva"]


def test_gemini_400_nao_repete_no_mesmo_modelo_mas_tenta_o_proximo(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    gemini = mod.Gemini.__new__(mod.Gemini)
    gemini.modelos = ["principal", "reserva"]
    chamadas = []

    def chamar(modelo, texto):
        chamadas.append(modelo)
        raise errors.APIError(400, {"error": {"message": "pedido inválido"}})

    gemini._chamar = chamar
    with pytest.raises(mod.ProvedorIndisponivel, match="pedido inválido"):
        gemini.classificar("bom")
    assert chamadas == ["principal", "reserva"]


def test_config_de_raciocinio_por_familia_de_modelo():
    from google.genai import types

    gemini = mod.Gemini.__new__(mod.Gemini)
    gemini.types = types
    assert gemini._config_raciocinio("gemini-2.5-flash").thinking_budget == 0
    assert gemini._config_raciocinio("gemini-flash-lite-latest").thinking_level is not None


def test_ritmo_espaca_as_requisicoes(monkeypatch):
    relogio = [100.0]
    esperas = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: relogio[0])
    monkeypatch.setattr(mod.time, "sleep", esperas.append)
    ritmo = mod.Ritmo(por_minuto=30)  # uma a cada 2 s
    for _ in range(3):
        ritmo.aguardar()
    assert esperas == [0.0, 2.0, 4.0]


@pytest.fixture
def teste_csv(tmp_path):
    pasta = tmp_path / "dados"
    pasta.mkdir()
    pd.DataFrame({
        "texto": [f"bom {i}" for i in range(6)] + [f"ruim {i}" for i in range(3)] + ["meh"],
        "nota": 0,
        "rotulo": ["positivo"] * 6 + ["negativo"] * 3 + ["neutro"],
    }).to_csv(pasta / "teste.csv", index=False)
    return pasta


class ProvedorFalso:
    """Erra o neutro (diz positivo) e conta quantas chamadas recebeu."""

    chamadas = 0
    falhar_em = set()

    def __init__(self, modelos):
        pass

    def classificar(self, texto):
        ProvedorFalso.chamadas += 1
        if texto in ProvedorFalso.falhar_em:
            raise RuntimeError("falha simulada")
        rotulo = "negativo" if texto.startswith("ruim") else "positivo"
        return {"rotulo": rotulo, "modelo": "falso-1", "latencia_s": 0.5,
                "tokens_entrada": 100, "tokens_saida": 10}


@pytest.fixture
def provedor_falso(monkeypatch):
    ProvedorFalso.chamadas = 0
    ProvedorFalso.falhar_em = set()
    monkeypatch.setitem(mod.PROVEDORES, "falso", (ProvedorFalso, ["falso-1"]))
    return ProvedorFalso


def test_avaliar_llm_ponta_a_ponta(tmp_path, teste_csv, provedor_falso):
    r = mod.avaliar_llm("falso", n=10, pasta_dados=teste_csv, pasta_resultados=tmp_path / "res")

    assert r["acuracia"] == 0.9
    assert r["f1_por_classe"]["neutro"] == 0
    assert r["tokens_por_1k"] == {"entrada": 100_000, "saida": 10_000}
    assert r["modelos_que_responderam"] == {"falso-1": 10}
    assert r["erros_de_chamada"] == 0
    assert (tmp_path / "res" / "zero_shot_falso-1.json").exists()


def test_erro_numa_avaliacao_nao_derruba_a_execucao(tmp_path, teste_csv, provedor_falso):
    provedor_falso.falhar_em = {"bom 0"}
    r = mod.avaliar_llm("falso", n=10, pasta_dados=teste_csv, pasta_resultados=tmp_path / "res")
    assert r["erros_de_chamada"] == 1
    assert "falha simulada" in r["exemplo_erro"]
    assert r["acuracia"] == 0.8  # a avaliação que falhou conta como erro


def test_retoma_do_cache_sem_chamar_de_novo(tmp_path, teste_csv, provedor_falso):
    provedor_falso.falhar_em = {"bom 0"}
    mod.avaliar_llm("falso", n=10, pasta_dados=teste_csv, pasta_resultados=tmp_path / "res")
    assert provedor_falso.chamadas == 10

    provedor_falso.falhar_em = set()
    r = mod.avaliar_llm("falso", n=10, pasta_dados=teste_csv, pasta_resultados=tmp_path / "res")
    assert provedor_falso.chamadas == 11  # só a que tinha falhado foi refeita
    assert r["erros_de_chamada"] == 0
