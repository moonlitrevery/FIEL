"""
Testes do modulo generation.

As funcoes de formatacao de prompt e parsing de resposta sao puras e
testadas diretamente. A chamada real a API Gemini e testada com um cliente
falso (que imita a interface client.models.generate_content), para nao
depender de rede nem de uma chave de API valida nos testes automatizados -
isso valida a logica de construcao da chamada e parsing do nosso lado, que e
o que de fato escrevemos (a qualidade da resposta do modelo em si nao e algo
que testes unitarios devem cobrir).
"""

from types import SimpleNamespace

import pandas as pd
import pytest
from google.genai import types

from src.explain import FeatureContribution
from src.generation import (
    GeneratedExplanation,
    build_user_prompt,
    format_shap_features,
    format_similar_cases,
    format_transaction_summary,
    generate_explanation,
    parse_llm_response,
)
from src.rag.retrieval import RetrievedCase


class _FakeModels:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs = None

    def generate_content(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(text=self.response_text)


class FakeGeminiClient:
    """Imita o suficiente da interface google.genai.Client para os testes:
    client.models.generate_content(**kwargs) -> objeto com .text"""

    def __init__(self, response_text: str):
        self.models = _FakeModels(response_text)


@pytest.fixture
def sample_row() -> pd.Series:
    return pd.Series(
        {
            "type": "TRANSFER",
            "amount": 500000.0,
            "oldbalanceOrg": 500000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 100,
            "hour_of_day": 4,
        }
    )


@pytest.fixture
def sample_top_features() -> list[FeatureContribution]:
    return [
        FeatureContribution(feature_name="error_balance_orig", feature_value=0.0, shap_value=3.2),
        FeatureContribution(feature_name="amount", feature_value=500000.0, shap_value=1.1),
    ]


@pytest.fixture
def sample_similar_cases() -> list[RetrievedCase]:
    return [
        RetrievedCase(case_id="c1", text="Transacao TRANSFER similar.", metadata={}, similarity=0.98),
    ]


def test_format_transaction_summary_includes_key_fields(sample_row):
    summary = format_transaction_summary(sample_row)

    assert "TRANSFER" in summary
    assert "500,000.00" in summary


def test_format_shap_features_indicates_direction(sample_top_features):
    text = format_shap_features(sample_top_features)

    assert "error_balance_orig" in text
    assert "empurra para FRAUDE" in text


def test_format_similar_cases_handles_empty_list():
    text = format_similar_cases([])

    assert "nenhum caso similar" in text


def test_build_user_prompt_contains_all_context_blocks(sample_row, sample_top_features, sample_similar_cases):
    prompt = build_user_prompt(sample_row, 0.97, sample_top_features, sample_similar_cases)

    assert "DADOS DA TRANSACAO" in prompt
    assert "97.00%" in prompt
    assert "error_balance_orig" in prompt
    assert "Transacao TRANSFER similar." in prompt


def test_parse_llm_response_accepts_valid_json():
    raw = '{"narrativa": "texto", "features_citadas": ["amount"], "acao_recomendada": "bloquear"}'

    result = parse_llm_response(raw)

    assert isinstance(result, GeneratedExplanation)
    assert result.narrativa == "texto"
    assert result.features_citadas == ["amount"]
    assert result.acao_recomendada == "bloquear"


def test_parse_llm_response_raises_on_missing_keys():
    raw = '{"narrativa": "texto"}'

    with pytest.raises(ValueError, match="chaves obrigatorias"):
        parse_llm_response(raw)


def test_parse_llm_response_raises_on_invalid_json():
    raw = "isso nao e json"

    with pytest.raises(ValueError, match="JSON valido"):
        parse_llm_response(raw)


def test_parse_llm_response_raises_on_wrong_type_for_features_citadas():
    raw = '{"narrativa": "texto", "features_citadas": "amount", "acao_recomendada": "bloquear"}'

    with pytest.raises(ValueError, match="features_citadas"):
        parse_llm_response(raw)


def test_generate_explanation_uses_response_schema_and_parses_result(
    sample_row, sample_top_features, sample_similar_cases
):
    fake_response_text = (
        '{"narrativa": "A transacao foi sinalizada por esvaziar a conta de origem.", '
        '"features_citadas": ["error_balance_orig"], '
        '"acao_recomendada": "bloquear e contatar cliente"}'
    )
    client = FakeGeminiClient(fake_response_text)

    result = generate_explanation(
        row=sample_row,
        predicted_probability=0.99,
        top_features=sample_top_features,
        similar_cases=sample_similar_cases,
        client=client,
    )

    assert result.features_citadas == ["error_balance_orig"]
    assert result.acao_recomendada == "bloquear e contatar cliente"

    # A chamada deve pedir explicitamente saida JSON estruturada via schema,
    # nao so confiar em instrucao textual no prompt.
    sent_config = client.models.last_kwargs["config"]
    assert sent_config.response_mime_type == "application/json"
    assert sent_config.response_schema is not None
    # thinking_level minimal: evita que o raciocinio interno do modelo
    # consuma o orcamento de tokens que deveria ir para a resposta em JSON.
    assert sent_config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL


def test_generate_explanation_raises_clear_error_on_empty_response(
    sample_row, sample_top_features, sample_similar_cases
):
    client = FakeGeminiClient(response_text="")

    with pytest.raises(ValueError, match="Resposta vazia"):
        generate_explanation(
            row=sample_row,
            predicted_probability=0.99,
            top_features=sample_top_features,
            similar_cases=sample_similar_cases,
            client=client,
        )
