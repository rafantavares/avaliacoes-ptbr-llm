"""Fine-tuning para classificação de sentimento, pensado para uma GPU
gratuita (Kaggle T4/P100, 16 GB).

Dois modelos, com o mesmo laço de treino e a mesma avaliação:
- BERTimbau base (`neuralmind/bert-base-portuguese-cased`): encoder treinado
  em português, ajuste completo dos 110M parâmetros.
- Qwen2.5-0.5B-Instruct: um LLM pequeno e multilíngue com cabeça de
  classificação, ajustado com LoRA — só ~1% dos parâmetros é treinado, o que
  cabe na GPU gratuita e gera um adaptador de poucos MB para publicar.

As classes são desbalanceadas (~9% neutras), então a perda é ponderada pelo
inverso da frequência de cada classe.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from avaliacoes.dados import ROTULOS, SEMENTE
from avaliacoes.metricas import avaliar, salvar

ID_POR_ROTULO = {r: i for i, r in enumerate(ROTULOS)}


@dataclass
class Config:
    nome: str
    modelo_base: str
    lora: bool = False
    epocas: float = 2
    taxa_aprendizado: float = 2e-5
    lote: int = 32
    max_tokens: int = 128
    alvos_lora: list[str] = field(default_factory=list)


BERTIMBAU = Config(nome="bertimbau_base", modelo_base="neuralmind/bert-base-portuguese-cased")

QWEN_LORA = Config(
    nome="qwen2.5_0.5b_lora",
    modelo_base="Qwen/Qwen2.5-0.5B-Instruct",
    lora=True,
    taxa_aprendizado=2e-4,
    lote=16,
    alvos_lora=["q_proj", "k_proj", "v_proj", "o_proj"],
)


class TrainerPonderado(Trainer):
    """Trainer com entropia cruzada ponderada por classe."""

    def __init__(self, *args, pesos_classes: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesos_classes = pesos_classes

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        rotulos = inputs.pop("labels")
        saidas = model(**inputs)
        perda = torch.nn.functional.cross_entropy(
            saidas.logits.float(), rotulos, weight=self.pesos_classes.to(saidas.logits.device)
        )
        return (perda, saidas) if return_outputs else perda


def _carregar_modelo(cfg: Config, tokenizer):
    modelo = AutoModelForSequenceClassification.from_pretrained(
        cfg.modelo_base,
        num_labels=len(ROTULOS),
        id2label=dict(enumerate(ROTULOS)),
        label2id=ID_POR_ROTULO,
        # Se o modelo base já tiver uma cabeça de classificação com outro número
        # de classes, ela é recriada em vez de dar erro.
        ignore_mismatched_sizes=True,
    )
    if tokenizer.pad_token is None:  # LLMs decoder-only não têm token de padding
        tokenizer.pad_token = tokenizer.eos_token
    modelo.config.pad_token_id = tokenizer.pad_token_id

    if cfg.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        modelo = get_peft_model(modelo, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=cfg.alvos_lora,
        ))
        modelo.print_trainable_parameters()
    return modelo


def _tokenizar(df: pd.DataFrame, tokenizer, max_tokens: int) -> Dataset:
    ds = Dataset.from_dict({
        "texto": df["texto"].tolist(),
        "labels": df["rotulo"].map(ID_POR_ROTULO).tolist(),
    })
    return ds.map(lambda lote: tokenizer(lote["texto"], truncation=True, max_length=max_tokens),
                  batched=True, remove_columns=["texto"])


def _pesos_classes(treino: pd.DataFrame) -> torch.Tensor:
    contagem = treino["rotulo"].value_counts().reindex(ROTULOS).to_numpy(dtype=float)
    pesos = contagem.sum() / (len(ROTULOS) * contagem)
    return torch.tensor(pesos, dtype=torch.float)


def treinar_e_avaliar(cfg: Config, pasta_dados: str | Path = "data/processed",
                      pasta_saida: str | Path = "models") -> dict:
    pasta_dados = Path(pasta_dados)
    treino = pd.read_csv(pasta_dados / "treino.csv")
    validacao = pd.read_csv(pasta_dados / "validacao.csv")
    teste = pd.read_csv(pasta_dados / "teste.csv")

    tokenizer = AutoTokenizer.from_pretrained(cfg.modelo_base)
    modelo = _carregar_modelo(cfg, tokenizer)
    ds_treino, ds_val, ds_teste = (
        _tokenizar(df, tokenizer, cfg.max_tokens) for df in (treino, validacao, teste)
    )

    saida = Path(pasta_saida) / cfg.nome
    args = TrainingArguments(
        output_dir=str(saida / "checkpoints"),
        num_train_epochs=cfg.epocas,
        learning_rate=cfg.taxa_aprendizado,
        per_device_train_batch_size=cfg.lote,
        per_device_eval_batch_size=cfg.lote * 2,
        warmup_steps=0.06,  # fração dos passos de treino (transformers >= 5)
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEMENTE,
    )

    def metricas_validacao(pred):
        previstos = [ROTULOS[i] for i in np.argmax(pred.predictions, axis=-1)]
        verdadeiros = [ROTULOS[i] for i in pred.label_ids]
        return {"f1_macro": avaliar(verdadeiros, previstos, nome=cfg.nome)["f1_macro"]}

    trainer = TrainerPonderado(
        model=modelo,
        args=args,
        train_dataset=ds_treino,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metricas_validacao,
        pesos_classes=_pesos_classes(treino),
    )

    inicio = time.perf_counter()
    trainer.train()
    segundos_treino = time.perf_counter() - inicio

    inicio = time.perf_counter()
    predicoes = trainer.predict(ds_teste)
    ms_por_amostra = (time.perf_counter() - inicio) * 1000 / len(teste)
    previstos = [ROTULOS[i] for i in np.argmax(predicoes.predictions, axis=-1)]

    trainer.save_model(str(saida / "final"))
    tokenizer.save_pretrained(str(saida / "final"))

    resultado = avaliar(teste["rotulo"], previstos, nome=cfg.nome, extras={
        "modelo_base": cfg.modelo_base,
        "lora": cfg.lora,
        "parametros_treinaveis": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        "segundos_treino": round(segundos_treino, 1),
        "ms_por_amostra": round(ms_por_amostra, 3),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    })
    salvar(resultado)
    return resultado
