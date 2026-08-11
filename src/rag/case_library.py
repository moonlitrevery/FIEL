"""
Serializacao de transacoes em texto estruturado para a biblioteca de casos do RAG.

REGRA FUNDAMENTAL DO PROJETO: a biblioteca de casos nunca e gerada por LLM
nem contem texto ficticio. Cada documento aqui vem de uma transacao de
FRAUDE CONFIRMADA (isFraud=1) do proprio conjunto de TREINO do PaySim,
serializada por um template deterministico (f-strings simples, sem nenhuma
chamada de modelo de linguagem). Isso importa para o rigor academico do
projeto: o RAG recupera "precedentes reais" (dentro do que o dataset
sintetico representa), nao alucinacoes empilhadas sobre dado sintetico.

O mesmo template de serializacao (serialize_transaction_to_text) e usado
tanto para montar a biblioteca de casos (a partir do treino) quanto para
descrever, em retrieval.py, a transacao suspeita sendo analisada no momento
da consulta - assim, a busca por similaridade compara "texto no mesmo
formato" dos dois lados, o que e importante para embeddings funcionarem bem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Casas decimais usadas para exibir valores monetarios no texto serializado.
# O PaySim nao especifica moeda; usamos apenas um formato numerico consistente
# (separador de milhar "," e decimal "."), sem assumir R$/US$.
_CURRENCY_FORMAT = ",.2f"


@dataclass
class FraudCase:
    """Um caso de fraude confirmada, pronto para ser indexado pelo RAG.

    case_id: identificador unico e rastreavel ate a linha original do PaySim
        (permite, se necessario na apresentacao, apontar de volta pro dado
        bruto que originou o caso).
    text: descricao textual estruturada da transacao (ver
        serialize_transaction_to_text), usada para gerar o embedding.
    metadata: campos numericos/categoricos da transacao, guardados a parte
        do texto para permitir filtros e exibicao estruturada na UI sem
        precisar re-parsear a string.
    """

    case_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def serialize_transaction_to_text(row: pd.Series) -> str:
    """Converte uma linha de transacao (com features ja calculadas por
    data_prep.engineer_features) em uma descricao textual estruturada.

    E uma funcao puramente determinística: mesmo input sempre produz o
    mesmo texto. Nao ha geracao de linguagem natural livre nem chamada a
    LLM - cada frase e um template fixo preenchido com valores reais da
    transacao. Funciona tanto para casos de fraude confirmada (biblioteca do
    RAG) quanto para uma transacao suspeita nova sendo consultada
    (retrieval.py), garantindo que os dois lados da busca por similaridade
    estejam no mesmo formato.
    """
    account_drained = row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0
    balance_inconsistent_orig = abs(row["error_balance_orig"]) > 0.01
    balance_inconsistent_dest = abs(row["error_balance_dest"]) > 0.01

    lines = [
        f"Transacao do tipo {row['type']}, no valor de {row['amount']:{_CURRENCY_FORMAT}}.",
        f"Ocorreu no step {int(row['step'])} da simulacao (hora {int(row['hour_of_day'])} de um ciclo de 24h).",
        (
            f"Conta de origem: saldo anterior {row['oldbalanceOrg']:{_CURRENCY_FORMAT}}, "
            f"saldo posterior {row['newbalanceOrig']:{_CURRENCY_FORMAT}}."
        ),
        (
            f"Conta de destino: saldo anterior {row['oldbalanceDest']:{_CURRENCY_FORMAT}}, "
            f"saldo posterior {row['newbalanceDest']:{_CURRENCY_FORMAT}}."
        ),
    ]

    if account_drained:
        lines.append("A conta de origem foi totalmente esvaziada por essa transacao.")

    if balance_inconsistent_orig:
        lines.append(
            "Ha uma inconsistencia contabil na conta de origem "
            f"(diferenca de {row['error_balance_orig']:{_CURRENCY_FORMAT}} entre saldo esperado e saldo real), "
            "um padrao tipico de fraude nesse tipo de transacao."
        )

    if balance_inconsistent_dest:
        lines.append(
            "Ha uma inconsistencia contabil na conta de destino "
            f"(diferenca de {row['error_balance_dest']:{_CURRENCY_FORMAT}} entre saldo esperado e saldo real), "
            "sugerindo que o valor recebido nao foi refletido corretamente no saldo."
        )

    return " ".join(lines)


def build_case_id(row: pd.Series) -> str:
    """Gera um identificador unico e rastreavel para o caso.

    Combina os identificadores originais (nameOrig/nameDest) e o step da
    simulacao, permitindo localizar a linha exata no CSV do PaySim que deu
    origem ao caso, se preciso justificar a proveniencia na apresentacao.
    """
    return f"{row['nameOrig']}-{row['nameDest']}-step{int(row['step'])}"


def row_to_metadata(row: pd.Series) -> dict:
    """Extrai os campos estruturados da transacao usados como metadata no
    ChromaDB (retrieval.py). Mantidos separados do texto para permitir
    exibir a transacao de forma tabular na UI, sem re-parsear a string."""
    return {
        "transaction_type": str(row["type"]),
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
        "error_balance_orig": float(row["error_balance_orig"]),
        "error_balance_dest": float(row["error_balance_dest"]),
        "hour_of_day": int(row["hour_of_day"]),
        "step": int(row["step"]),
    }


def build_case_library(df_train: pd.DataFrame, max_cases: int | None = None) -> list[FraudCase]:
    """Monta a biblioteca de casos a partir das fraudes confirmadas do treino.

    Filtra df_train para isFraud == 1 (nunca usamos transacoes legitimas nem
    dados de teste aqui - a biblioteca so pode conter fraudes confirmadas do
    treino, para nao vazar informacao do conjunto de avaliacao) e serializa
    cada uma em um FraudCase.

    max_cases: limite opcional de casos (amostrados de forma deterministica,
    pegando os primeiros N apos ordenar por case_id) - util para iteracao
    rapida durante o desenvolvimento do RAG. Por padrao (None), usa todas as
    fraudes de treino disponiveis (tipicamente alguns milhares no PaySim
    filtrado, volume tranquilo para embeddings + ChromaDB).
    """
    fraud_rows = df_train[df_train["isFraud"] == 1].copy()

    cases = [
        FraudCase(
            case_id=build_case_id(row),
            text=serialize_transaction_to_text(row),
            metadata=row_to_metadata(row),
        )
        for _, row in fraud_rows.iterrows()
    ]
    cases.sort(key=lambda case: case.case_id)

    if max_cases is not None:
        cases = cases[:max_cases]

    return cases


if __name__ == "__main__":
    # Execucao manual: `uv run python -m src.rag.case_library`
    from src.data_prep import load_raw_data, prepare_train_test_split

    raw_df = load_raw_data("data/paysim.csv")
    _, _, _, _, df_train, _ = prepare_train_test_split(raw_df)

    cases = build_case_library(df_train)
    print(f"Biblioteca de casos construida com {len(cases):,} fraudes confirmadas do treino.")
    print("\nExemplo de caso serializado:")
    print(f"  case_id: {cases[0].case_id}")
    print(f"  text: {cases[0].text}")
    print(f"  metadata: {cases[0].metadata}")
