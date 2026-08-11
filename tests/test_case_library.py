"""
Testes do modulo rag.case_library.

Focam em garantir a regra mais importante do projeto: a biblioteca de casos
so pode conter fraudes confirmadas (isFraud=1) e o texto e 100% derivado dos
valores reais da transacao (nada de texto livre/ficticio).
"""

import pandas as pd
import pytest

from src.data_prep import engineer_features
from src.rag.case_library import (
    build_case_id,
    build_case_library,
    row_to_metadata,
    serialize_transaction_to_text,
)


def make_fake_train_df() -> pd.DataFrame:
    """DataFrame ja no formato pos-engineer_features, com fraudes e
    transacoes legitimas misturadas (como seria df_train de verdade)."""
    raw = pd.DataFrame(
        {
            "step": [10, 20, 30],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
            "amount": [5000.0, 300.0, 8000.0],
            "nameOrig": ["C1", "C2", "C3"],
            "oldbalanceOrg": [5000.0, 1000.0, 8000.0],
            "newbalanceOrig": [0.0, 700.0, 0.0],
            "nameDest": ["M1", "M2", "M3"],
            "oldbalanceDest": [0.0, 200.0, 0.0],
            "newbalanceDest": [0.0, 500.0, 0.0],
            "isFraud": [1, 0, 1],
            "isFlaggedFraud": [0, 0, 0],
        }
    )
    return engineer_features(raw)


def test_serialize_transaction_to_text_is_deterministic():
    df = make_fake_train_df()
    row = df.iloc[0]

    text_a = serialize_transaction_to_text(row)
    text_b = serialize_transaction_to_text(row)

    assert text_a == text_b
    assert "TRANSFER" in text_a
    assert "5,000.00" in text_a


def test_serialize_transaction_to_text_flags_drained_account_and_inconsistency():
    df = make_fake_train_df()
    fraud_row = df.iloc[0]  # conta esvaziada, destino nao recebeu o valor

    text = serialize_transaction_to_text(fraud_row)

    assert "totalmente esvaziada" in text
    assert "inconsistencia contabil na conta de destino" in text


def test_serialize_transaction_to_text_does_not_flag_consistent_transaction():
    df = make_fake_train_df()
    legit_row = df.iloc[1]  # saldo bate certinho (nao esvaziada, sem erro de saldo)

    text = serialize_transaction_to_text(legit_row)

    assert "totalmente esvaziada" not in text
    assert "inconsistencia contabil" not in text


def test_build_case_id_is_unique_and_traceable():
    df = make_fake_train_df()
    row = df.iloc[0]

    case_id = build_case_id(row)

    assert case_id == "C1-M1-step10"


def test_row_to_metadata_contains_only_simple_types():
    df = make_fake_train_df()
    row = df.iloc[0]

    metadata = row_to_metadata(row)

    for value in metadata.values():
        assert isinstance(value, (str, int, float))


def test_build_case_library_only_includes_confirmed_fraud():
    df = make_fake_train_df()

    cases = build_case_library(df)

    assert len(cases) == 2  # so as 2 linhas com isFraud=1
    for case in cases:
        assert "TRANSFER" in case.text


def test_build_case_library_respects_max_cases():
    df = make_fake_train_df()

    cases = build_case_library(df, max_cases=1)

    assert len(cases) == 1


def test_build_case_library_raises_no_llm_generated_text():
    """Checagem de regressao conceitual: o texto de cada caso deve ser
    inteiramente reconstruivel a partir dos proprios metadados numericos,
    sem nenhum conteudo externo (prova indireta de que nao ha geracao livre)."""
    df = make_fake_train_df()

    cases = build_case_library(df)

    for case in cases:
        amount_str = f"{case.metadata['amount']:,.2f}"
        assert amount_str in case.text
        assert case.metadata["transaction_type"] in case.text
