from avaliacoes.metricas import avaliar


def test_f1_macro_penaliza_quem_ignora_a_classe_minoritaria():
    verdadeiros = ["positivo"] * 8 + ["negativo"] + ["neutro"]
    sempre_positivo = ["positivo"] * 10
    resultado = avaliar(verdadeiros, sempre_positivo, nome="trivial")
    assert resultado["acuracia"] == 0.8
    assert resultado["f1_macro"] < 0.35
    assert resultado["f1_por_classe"]["neutro"] == 0


def test_matriz_de_confusao_na_ordem_dos_rotulos():
    resultado = avaliar(["negativo", "neutro", "positivo"], ["negativo", "positivo", "positivo"],
                        nome="x", extras={"hardware": "CPU"})
    assert resultado["matriz_confusao"]["classes"] == ["negativo", "neutro", "positivo"]
    assert resultado["matriz_confusao"]["valores"] == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]
    assert resultado["hardware"] == "CPU"


def test_rotulo_fora_da_lista_conta_como_erro():
    resultado = avaliar(["positivo", "negativo"], ["positivo", "recusa"], nome="llm")
    assert resultado["acuracia"] == 0.5
