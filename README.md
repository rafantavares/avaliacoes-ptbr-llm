<h1 align="center">Classificação de avaliações de clientes (análise de sentimento)</h1>

<p align="center">
  Modelo clássico, <b>fine-tuning</b> (BERTimbau e Qwen2.5 + LoRA) e <b>LLM zero-shot</b> comparados
  no mesmo conjunto de teste — em <b>qualidade, latência e custo</b>.
</p>

<p align="center">
  <a href="https://github.com/rafantavares/avaliacoes-ptbr-llm/actions/workflows/testes.yml"><img alt="Testes" src="https://github.com/rafantavares/avaliacoes-ptbr-llm/actions/workflows/testes.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Transformers" src="https://img.shields.io/badge/🤗-Transformers%20%2B%20PEFT-yellow">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white">
</p>

---

## A pergunta

Para classificar milhares de avaliações de clientes por dia, vale mais **chamar um LLM grande por
requisição** ou **ajustar um modelo pequeno e rodá-lo na própria infraestrutura**? A resposta depende
de qualidade, mas também de latência e custo — então este projeto mede os três.

**Dados:** 40.977 avaliações reais com comentário (35.056 após remover duplicatas) da [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
com comentário. A nota vira o rótulo: 1–2 estrelas = **negativo**, 3 = **neutro**, 4–5 = **positivo**.

## Resultados

Conjunto de teste: **5.259 avaliações** (15%, estratificado). Métrica principal: **F1-macro**, porque as
classes são desbalanceadas (62% positivas, 9% neutras) e a acurácia esconderia um modelo que ignora o neutro.

| Abordagem | F1-macro | Acurácia | F1 neutro | Latência / avaliação | Onde roda |
|---|---|---|---|---|---|
| TF-IDF + regressão logística | 0,662 | 81,4% | 0,271 | 0,16 ms | CPU |
| BERTimbau base (ajuste completo) | *em treino* | | | | GPU (Kaggle) |
| Qwen2.5-0.5B + LoRA | *em treino* | | | | GPU (Kaggle) |
| Gemini Flash zero-shot | *em execução* | | | | API |

O baseline já mostra onde está o desafio: acerta bem positivo (F1 0,91) e negativo (0,81), mas quase
não reconhece o **neutro** — quem dá 3 estrelas costuma escrever algo ambíguo. É nessa classe que os
modelos maiores precisam provar que valem o custo.

## Decisões de engenharia

| Decisão | Por quê |
|---|---|
| **Remover duplicatas antes de dividir** | "Ótimo produto" aparece centenas de vezes. Com a mesma frase no treino e no teste, as métricas ficariam infladas (vazamento). A comparação é feita por uma forma normalizada do texto (sem acento, caixa e pontuação). |
| **Mesma partição para todas as abordagens** | A divisão tem semente fixa e uma impressão digital (hash) do teste. O notebook do Kaggle confere o hash antes de treinar, garantindo que todos os modelos foram avaliados nas mesmas avaliações. |
| **Perda ponderada por classe** | Sem isso, os modelos aprendem a quase nunca prever "neutro". |
| **LoRA no Qwen** | Treina ~1% dos parâmetros: cabe numa GPU gratuita e o resultado é um adaptador de poucos MB. |
| **Saída estruturada no LLM** | O LLM responde num esquema JSON com enum fechado; resposta fora do esquema conta como erro, não é "consertada" depois. |
| **LLM com plano B** | No limite do nível gratuito (429) ou em sobrecarga (5xx), espera de forma progressiva e troca de modelo, em vez de perder a avaliação. |

## Estrutura

```
src/avaliacoes/
  dados.py           # limpeza, deduplicação, rótulos, divisão estratificada + hash do teste
  baseline.py        # TF-IDF (palavras + n-gramas de caracteres) + regressão logística
  finetune.py        # BERTimbau e Qwen2.5 + LoRA com o mesmo laço de treino
  llm_zero_shot.py   # Gemini (gratuito) ou Claude, com latência e tokens medidos
  metricas.py        # F1-macro, F1 por classe e matriz de confusão — iguais para todos
notebooks/
  finetune_kaggle.ipynb   # treino na GPU gratuita do Kaggle
results/             # métricas de cada abordagem em JSON
tests/               # 22 testes (dados, métricas, LLM com provedor simulado)
```

## Como reproduzir

```bash
pip install -e ".[dev]"
# baixe olist_order_reviews_dataset.csv para data/raw/ (Kaggle: olistbr/brazilian-ecommerce)
python -m avaliacoes.dados        # gera data/processed/{treino,validacao,teste}.csv
python -m avaliacoes.baseline     # baseline em segundos, na CPU

pip install -e ".[llm]"
echo GEMINI_API_KEY=... > .env    # chave gratuita do Google AI Studio
python -m avaliacoes.llm_zero_shot -n 500
```

O fine-tuning roda no [notebook do Kaggle](notebooks/finetune_kaggle.ipynb) (GPU gratuita).

## Limitações

- O rótulo vem da **nota**, não de anotação humana do texto: há avaliações com texto positivo e nota baixa
  (o problema foi na entrega, não no produto). Parte do erro "inevitável" está aí.
- O LLM zero-shot é avaliado numa amostra estratificada de 500 avaliações do teste, não no teste inteiro,
  para caber no limite do nível gratuito.
- Dados sob licença **CC BY-NC-SA 4.0** (Olist): uso não comercial.

## Autor

**Rafael Nunes Tavares** — [github.com/rafantavares](https://github.com/rafantavares) ·
[LinkedIn](https://www.linkedin.com/in/rafael-nunes-tavares-20a76628b)

Desenvolvido com apoio do [Claude Code](https://claude.com/claude-code); desenho do experimento,
decisões e análise dos resultados feitos por mim.
