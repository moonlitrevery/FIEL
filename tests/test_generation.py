"""
Testes do modulo generation.

As funcoes de formatacao de prompt e parsing de resposta sao puras e
testadas diretamente. A chamada real a API Claude e testada com um cliente
falso (que imita a interface client.messages.create), para nao depender de
rede nem de uma chave de API valida nos testes automatizados - isso valida
a logica de prefill/parsing do nosso lado, que e o que de fato escrevemos
(a qualidade da resposta do modelo em si nao e algo que testes unitarios
devem cobrir).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

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


class _FakeMessages:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text=self.response_text)])


class FakeAnthropicClient:
    """Imita o suficiente da interface anthropic.Anthropic para os testes:
    client.messages.create(**kwargs) -> objeto com .content[0].text"""

    def __init__(self, response_text: str):
        self.messages = _FakeMessages(response_text)


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


def test_parse_llm_response_strips_markdown_fences():
    raw = '```json\n{"narrativa": "texto", "features_citadas": [], "acao_recomendada": "revisar"}\n```'

    result = parse_llm_response(raw)

    assert result.acao_recomendada == "revisar"


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


def test_generate_explanation_uses_assistant_prefill_and_parses_result(
    sample_row, sample_top_features, sample_similar_cases
):
    # O cliente falso simula a resposta da API JA CONSIDERANDO o prefill "{"
    # que generate_explanation envia como ultimo turno "assistant" - ou
    # seja, a API so devolve a CONTINUACAO apos a chave de abertura.
    fake_response_after_prefill = (
        '"narrativa": "A transacao foi sinalizada por esvaziar a conta de origem.", '
        '"features_citadas": ["error_balance_orig"], '
        '"acao_recomendada": "bloquear e contatar cliente"}'
    )
    client = FakeAnthropicClient(fake_response_after_prefill)

    result = generate_explanation(
        row=sample_row,
        predicted_probability=0.99,
        top_features=sample_top_features,
        similar_cases=sample_similar_cases,
        client=client,
    )

    assert result.features_citadas == ["error_balance_orig"]
    assert result.acao_recomendada == "bloquear e contatar cliente"

    # O ultimo turno enviado a API deve ser o prefill "{" do assistente,
    # tecnica usada para forcar a saida a comecar direto no JSON.
    sent_messages = client.messages.last_kwargs["messages"]
    assert sent_messages[-1] == {"role": "assistant", "content": "{"}
