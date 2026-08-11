"""
Testes do modulo rag.retrieval.

Sao testes de integracao (usam o modelo real de sentence-transformers e um
ChromaDB persistido em diretorio temporario) em vez de mocks, porque o
proprio comportamento que queremos validar - "textos parecidos tem
embeddings parecidos e sao recuperados primeiro" - so existe de verdade com
o modelo real. O modelo e pequeno (~118MB) e fica em cache local apos o
primeiro download/teste.

Os textos de exemplo sao gerados pela propria serialize_transaction_to_text
(nao escritos a mao), para que tenham exatamente a mesma estrutura de frases
que os casos reais da biblioteca - uma diferenca estrutural no texto (ex:
faltar a frase "Conta de destino: ...") mudaria o embedding por causa da
forma do texto, nao so do conteudo, o que invalidaria o teste.
"""

import pandas as pd
import pytest

from src.data_prep import engineer_features
from src.rag.case_library import build_case_library, serialize_transaction_to_text
from src.rag.retrieval import (
    RetrievedCase,
    index_cases,
    retrieve_similar_cases,
    retrieve_similar_cases_for_transaction,
)


def make_case_source_df() -> pd.DataFrame:
    """Duas fraudes confirmadas com padroes bem diferentes entre si: uma
    TRANSFER de alto valor com conta esvaziada, outra CASH_OUT pequena e sem
    inconsistencia de saldo."""
    raw = pd.DataFrame(
        {
            "step": [100, 200],
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [500_000.0, 45.0],
            "nameOrig": ["C1", "C2"],
            "oldbalanceOrg": [500_000.0, 1000.0],
            "newbalanceOrig": [0.0, 955.0],
            "nameDest": ["M1", "M2"],
            "oldbalanceDest": [0.0, 200.0],
            "newbalanceDest": [0.0, 245.0],
            "isFraud": [1, 1],
            "isFlaggedFraud": [0, 0],
        }
    )
    return engineer_features(raw)


def make_query_df(transaction_type: str, amount: float) -> pd.DataFrame:
    """Constroi uma unica transacao TRANSFER com conta de origem esvaziada,
    no mesmo formato de df_train/df_test, para usar como consulta nos testes."""
    raw = pd.DataFrame(
        {
            "step": [999],
            "type": [transaction_type],
            "amount": [amount],
            "nameOrig": ["C9"],
            "oldbalanceOrg": [amount],
            "newbalanceOrig": [0.0],
            "nameDest": ["M9"],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [0.0],
            "isFraud": [1],
            "isFlaggedFraud": [0],
        }
    )
    return engineer_features(raw)


@pytest.fixture
def sample_cases():
    df = make_case_source_df()
    return build_case_library(df)


def test_index_and_retrieve_returns_most_similar_case_first(tmp_path, sample_cases):
    collection = index_cases(sample_cases, persist_directory=str(tmp_path), collection_name="test_cases")

    query_row = make_query_df("TRANSFER", 480_000.0).iloc[0]
    query_text = serialize_transaction_to_text(query_row)

    results = retrieve_similar_cases(query_text, collection, k=2)

    assert len(results) == 2
    assert isinstance(results[0], RetrievedCase)
    assert results[0].metadata["transaction_type"] == "TRANSFER"
    assert results[0].similarity > results[1].similarity


def test_retrieve_similar_cases_respects_k(tmp_path, sample_cases):
    collection = index_cases(sample_cases, persist_directory=str(tmp_path), collection_name="test_cases_k")

    results = retrieve_similar_cases("qualquer transacao", collection, k=1)

    assert len(results) == 1


def test_retrieve_similar_cases_for_transaction_uses_same_serialization(tmp_path, sample_cases):
    collection = index_cases(sample_cases, persist_directory=str(tmp_path), collection_name="test_cases_row")

    query_row = make_query_df("TRANSFER", 495_000.0).iloc[0]

    results = retrieve_similar_cases_for_transaction(query_row, collection, k=1)

    assert results[0].metadata["transaction_type"] == "TRANSFER"


def test_reindexing_replaces_previous_collection_contents(tmp_path, sample_cases):
    index_cases(sample_cases, persist_directory=str(tmp_path), collection_name="test_cases_reset")

    smaller_case_set = sample_cases[:1]
    collection = index_cases(
        smaller_case_set, persist_directory=str(tmp_path), collection_name="test_cases_reset"
    )

    assert collection.count() == 1
