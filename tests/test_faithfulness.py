"""
Testes do modulo faithfulness - o verificador de fidelidade, parte mais
importante do projeto. Cobrimos cada componente da metrica isoladamente
(precision, recall, F1, recall ponderado, alucinacao vs. irrelevancia) com
valores calculados a mao, para que qualquer um possa conferir a aritmetica
na apresentacao.
"""

from types import SimpleNamespace

import pytest

from src.explain import FeatureContribution
from src.faithfulness import (
    aggregate_faithfulness_results,
    check_faithfulness,
    compute_faithfulness,
)

REAL_TOP_FEATURES = [
    FeatureContribution(feature_name="error_balance_orig", feature_value=0.0, shap_value=4.0),
    FeatureContribution(feature_name="newbalanceOrig", feature_value=0.0, shap_value=2.0),
    FeatureContribution(feature_name="amount", feature_value=500000.0, shap_value=1.0),
    FeatureContribution(feature_name="oldbalanceOrg", feature_value=500000.0, shap_value=-1.0),
]
# soma dos |shap|: 4 + 2 + 1 + 1 = 8


def test_perfect_citation_has_precision_recall_f1_equal_to_one():
    cited = ["error_balance_orig", "newbalanceOrig", "amount", "oldbalanceOrg"]

    result = compute_faithfulness(cited, REAL_TOP_FEATURES)

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.weighted_recall == 1.0
    assert result.is_faithful is True
    assert result.omitted_features == []


def test_partial_citation_computes_precision_and_recall_correctly():
    # Cita 2 das 4 features reais (ambas corretas) -> precision=1.0, recall=0.5
    cited = ["error_balance_orig", "newbalanceOrig"]

    result = compute_faithfulness(cited, REAL_TOP_FEATURES)

    assert result.precision == 1.0
    assert result.recall == 0.5
    assert result.f1 == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))
    assert result.omitted_features == ["amount", "oldbalanceOrg"]


def test_weighted_recall_gives_more_credit_to_dominant_feature():
    # Citar so a feature dominante (shap=4, de um total de |shap|=8) deve dar
    # weighted_recall=0.5, MUITO maior que o recall simples (1/4=0.25) -
    # essa e exatamente a diferenca que a metrica ponderada deve capturar.
    cited = ["error_balance_orig"]

    result = compute_faithfulness(cited, REAL_TOP_FEATURES)

    assert result.recall == 0.25
    assert result.weighted_recall == 0.5
    assert result.weighted_recall > result.recall


def test_hallucinated_feature_is_flagged_separately_from_irrelevant():
    cited = ["error_balance_orig", "feature_que_nao_existe"]

    result = compute_faithfulness(cited, REAL_TOP_FEATURES, valid_feature_names=["error_balance_orig", "amount"])

    assert result.hallucinated_features == ["feature_que_nao_existe"]
    assert result.irrelevant_features == []
    assert result.correctly_cited == ["error_balance_orig"]


def test_irrelevant_feature_is_valid_but_not_top_k():
    # "step" e uma feature valida do modelo, mas nao esta no top-k real desse
    # exemplo -> deve contar como irrelevante, nao como alucinacao.
    cited = ["step"]

    result = compute_faithfulness(cited, REAL_TOP_FEATURES, valid_feature_names=["step", "amount"])

    assert result.irrelevant_features == ["step"]
    assert result.hallucinated_features == []
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_empty_citation_scores_zero_not_undefined():
    result = compute_faithfulness([], REAL_TOP_FEATURES)

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.weighted_recall == 0.0
    assert result.is_faithful is False


def test_duplicate_citations_are_deduplicated():
    cited = ["error_balance_orig", "error_balance_orig", "error_balance_orig"]

    result = compute_faithfulness(cited, REAL_TOP_FEATURES)

    assert result.cited_features == ["error_balance_orig"]
    assert result.precision == 1.0
    assert result.recall == 0.25


def test_is_faithful_respects_custom_threshold():
    cited = ["error_balance_orig"]  # recall=0.25, precision=1.0, f1=0.4

    strict_result = compute_faithfulness(cited, REAL_TOP_FEATURES, faithfulness_threshold=0.5)
    lenient_result = compute_faithfulness(cited, REAL_TOP_FEATURES, faithfulness_threshold=0.3)

    assert strict_result.is_faithful is False
    assert lenient_result.is_faithful is True


def test_check_faithfulness_accepts_generated_explanation_like_object():
    fake_explanation = SimpleNamespace(features_citadas=["error_balance_orig", "amount"])

    result = check_faithfulness(fake_explanation, REAL_TOP_FEATURES)

    assert result.correctly_cited == ["error_balance_orig", "amount"]


def test_aggregate_faithfulness_results_computes_means_and_rates():
    result_faithful = compute_faithfulness(
        ["error_balance_orig", "newbalanceOrig", "amount", "oldbalanceOrg"], REAL_TOP_FEATURES
    )
    result_hallucinating = compute_faithfulness(
        ["nome_inventado"], REAL_TOP_FEATURES, valid_feature_names=["error_balance_orig"]
    )

    report = aggregate_faithfulness_results([result_faithful, result_hallucinating])

    assert report.n_explanations == 2
    assert report.mean_precision == pytest.approx((1.0 + 0.0) / 2)
    assert report.faithful_rate == 0.5
    assert report.hallucination_rate == 0.5


def test_aggregate_faithfulness_results_handles_empty_list():
    report = aggregate_faithfulness_results([])

    assert report.n_explanations == 0
    assert report.faithful_rate == 0.0
    assert report.hallucination_rate == 0.0
