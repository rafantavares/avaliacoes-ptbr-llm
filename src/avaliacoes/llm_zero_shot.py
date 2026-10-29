"""Classificação zero-shot com um LLM via API, sem nenhum treino.

Serve de referência para a pergunta que interessa num produto: um modelo
pequeno, ajustado e rodando na nossa infraestrutura compensa frente a chamar
um LLM grande por requisição? Por isso, além da qualidade, medimos a latência
de cada chamada e os tokens consumidos (e o custo, quando o preço é conhecido).

Dois provedores, atrás da mesma interface:
- Gemini (padrão): nível gratuito do Google AI Studio. Requer GEMINI_API_KEY.
  No nível gratuito o Google pode usar o conteúdo enviado para melhorar seus
  produtos; aqui só trafegam avaliações que já são públicas.
- Claude: requer ANTHROPIC_API_KEY (ou `ant auth login`).

As chaves podem ficar num arquivo `.env` na raiz do projeto (não versionado).
"""
from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from avaliacoes.dados import ROTULOS, SEMENTE
from avaliacoes.metricas import avaliar, salvar

INSTRUCOES = """Você classifica avaliações de clientes de um e-commerce brasileiro.

Responda com o sentimento geral do cliente sobre a compra:
- "positivo": satisfeito, recomenda, elogia (mesmo com uma ressalva pequena).
- "neutro": misto ou morno — algo bom e algo ruim com peso parecido, ou um
  relato sem opinião clara ("chegou hoje", "ainda não testei").
- "negativo": insatisfeito — produto errado, defeito, atraso, não recebeu,
  arrependimento.

Avalie a experiência como um todo, não palavras isoladas: "não tenho do que
reclamar" é positivo."""

ESQUEMA_JSON = {
    "type": "object",
    "properties": {"sentimento": {"type": "string", "enum": ROTULOS}},
    "required": ["sentimento"],
    "additionalProperties": False,
}

TENTATIVAS_POR_MODELO = 4


class ProvedorIndisponivel(Exception):
    """Todos os modelos falharam por limite de uso ou indisponibilidade."""


def carregar_env(caminho: str | Path = ".env") -> None:
    """Lê `CHAVE=valor` de um .env sem sobrescrever variáveis já definidas."""
    arquivo = Path(caminho)
    if not arquivo.exists():
        return
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        if "=" in linha and not linha.lstrip().startswith("#"):
            chave, valor = linha.split("=", 1)
            os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


def extrair_rotulo(texto_json: str) -> str | None:
    """Valida a resposta estruturada; qualquer coisa fora do esquema vira None."""
    try:
        rotulo = json.loads(texto_json)["sentimento"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return rotulo if rotulo in ROTULOS else None


# --------------------------------------------------------------------- Gemini

class Gemini:
    """Gemini com plano B: em limite de uso (429) ou sobrecarga (5xx) espera e
    tenta de novo; esgotadas as tentativas, passa para o próximo modelo."""

    CODIGOS_TRANSITORIOS = {429, 500, 502, 503, 504}

    def __init__(self, modelos: list[str]):
        from google import genai
        from google.genai import types

        self.types = types
        self.modelos = modelos
        self.cliente = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options=types.HttpOptions(
                timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    def classificar(self, texto: str) -> dict:
        from google.genai import errors

        for modelo in self.modelos:
            for tentativa in range(TENTATIVAS_POR_MODELO):
                try:
                    return self._chamar(modelo, texto)
                except errors.APIError as erro:
                    if erro.code not in self.CODIGOS_TRANSITORIOS:
                        raise
                    time.sleep(min(60, 5 * 2 ** tentativa))  # 5, 10, 20, 40 s
        raise ProvedorIndisponivel(f"nenhum modelo respondeu: {self.modelos}")

    def _chamar(self, modelo: str, texto: str) -> dict:
        inicio = time.perf_counter()
        resposta = self.cliente.models.generate_content(
            model=modelo,
            contents=f"Avaliação: {texto}",
            config=self.types.GenerateContentConfig(
                system_instruction=INSTRUCOES,
                response_mime_type="application/json",
                response_json_schema=ESQUEMA_JSON,
                # Classificação simples não ganha com raciocínio longo.
                thinking_config=self.types.ThinkingConfig(thinking_budget=0),
            ),
        )
        uso = resposta.usage_metadata
        return {
            "rotulo": extrair_rotulo(resposta.text or ""),
            "modelo": modelo,
            "latencia_s": time.perf_counter() - inicio,
            "tokens_entrada": uso.prompt_token_count or 0,
            "tokens_saida": (uso.candidates_token_count or 0) + (uso.thoughts_token_count or 0),
        }


# --------------------------------------------------------------------- Claude

# US$ por milhão de tokens (entrada, saída), tabela de preços de out/2026.
PRECOS_CLAUDE_USD_POR_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class Claude:
    def __init__(self, modelos: list[str]):
        import anthropic

        self.modelo = modelos[0]
        self.cliente = anthropic.Anthropic()  # o SDK já repete 429/5xx com espera

    def classificar(self, texto: str) -> dict:
        inicio = time.perf_counter()
        resposta = self.cliente.beta.messages.create(
            model=self.modelo,
            max_tokens=1024,
            system=INSTRUCOES,
            messages=[{"role": "user", "content": f"Avaliação: {texto}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": ESQUEMA_JSON}},
            # Se o modelo recusar por política, a API refaz no modelo reserva.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        rotulo = None
        if resposta.stop_reason != "refusal":
            rotulo = extrair_rotulo(next((b.text for b in resposta.content if b.type == "text"), ""))
        return {
            "rotulo": rotulo,
            "modelo": resposta.model,
            "latencia_s": time.perf_counter() - inicio,
            "tokens_entrada": resposta.usage.input_tokens,
            "tokens_saida": resposta.usage.output_tokens,
        }


# ------------------------------------------------------------------ avaliação

PROVEDORES = {
    "gemini": (Gemini, ["gemini-flash-latest", "gemini-2.5-flash", "gemini-flash-lite-latest"]),
    "claude": (Claude, ["claude-opus-5-5"]),
}


def amostra_estratificada(teste: pd.DataFrame, n: int) -> pd.DataFrame:
    """Mesma proporção de classes do teste completo, com semente fixa."""
    fracao = min(1.0, n / len(teste))
    partes = [g.sample(frac=fracao, random_state=SEMENTE) for _, g in teste.groupby("rotulo")]
    return pd.concat(partes).reset_index(drop=True)


def avaliar_llm(provedor: str = "gemini", modelos: list[str] | None = None, n: int = 500,
                paralelo: int = 2, pasta_dados: str | Path = "data/processed") -> dict:
    classe, modelos_padrao = PROVEDORES[provedor]
    modelos = modelos or modelos_padrao
    cliente = classe(modelos)

    teste = pd.read_csv(Path(pasta_dados) / "teste.csv")
    amostra = amostra_estratificada(teste, n)
    with ThreadPoolExecutor(max_workers=paralelo) as executor:
        saidas = list(executor.map(cliente.classificar, amostra["texto"]))

    invalidas = sum(s["rotulo"] is None for s in saidas)
    # Recusa ou resposta fora do esquema conta como erro: o produto ficaria sem resposta.
    previstos = [s["rotulo"] or "invalida" for s in saidas]
    latencias = sorted(s["latencia_s"] for s in saidas)
    tokens_entrada = sum(s["tokens_entrada"] for s in saidas)
    tokens_saida = sum(s["tokens_saida"] for s in saidas)

    extras = {
        "provedor": provedor,
        "modelos_que_responderam": pd.Series([s["modelo"] for s in saidas]).value_counts().to_dict(),
        "respostas_invalidas": invalidas,
        "latencia_mediana_s": round(statistics.median(latencias), 3),
        "latencia_p95_s": round(latencias[int(0.95 * (len(latencias) - 1))], 3),
        "tokens_por_1k": {
            "entrada": round(tokens_entrada / len(saidas) * 1000),
            "saida": round(tokens_saida / len(saidas) * 1000),
        },
        "hardware": "API",
    }
    if provedor == "claude" and modelos[0] in PRECOS_CLAUDE_USD_POR_MTOK:
        preco_entrada, preco_saida = PRECOS_CLAUDE_USD_POR_MTOK[modelos[0]]
        custo = (tokens_entrada * preco_entrada + tokens_saida * preco_saida) / 1e6
        extras["custo_usd_por_1k"] = round(custo / len(saidas) * 1000, 4)
    elif provedor == "gemini":
        extras["custo_usd_por_1k"] = 0.0
        extras["observacao_custo"] = "nível gratuito do Google AI Studio"

    resultado = avaliar(amostra["rotulo"], previstos, nome=f"zero_shot_{modelos[0]}", extras=extras)
    salvar(resultado)
    return resultado


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provedor", default="gemini", choices=sorted(PROVEDORES))
    parser.add_argument("--modelo", action="append", help="pode repetir; o primeiro é o principal")
    parser.add_argument("-n", type=int, default=500, help="tamanho da amostra do teste")
    parser.add_argument("--paralelo", type=int, default=2)
    args = parser.parse_args()

    carregar_env()
    print(json.dumps(avaliar_llm(args.provedor, args.modelo, args.n, args.paralelo),
                     ensure_ascii=False, indent=2))
