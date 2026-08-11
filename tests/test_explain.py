"""
Testes do modulo explain.

Treinamos um modelo pequeno em dados sinteticos (mesmo padrao usado em
test_model.py) e verificamos as propriedades matematicas do SHAP que o
restante do pipeline depende: a decomposicao aditiva (base_value + soma dos
shap values reproduz a saida do modelo) e a ordenacao correta das top-k
features por importancia absoluta.
"""

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    FeatureContribution,
    build_explainer,
    explain_transaction,
    get_top_k_features,
)
from src.model import train_model


@pytest.fixture
def trained_model_and_data():
    rng = np.random.default_rng(42)
    n = 1200
    y = (rng.random(n) < 0.05).astype(int)

    amount = np.where(y == 1, rng.exponential(5000, size=n), rng.exponential(300, size=n))
    error_balance = np.where(y == 1, rng.exponential(3000, size=n), rng.normal(0, 5, size=n))

    X = pd.DataFrame(
        {
            "amount": amount,
            "error_balance_orig": error_balance,
            "error_balance_dest": error_balance,
            "type_TRANSFER": rng.integers(0, 2, size=n),
        }
    )
    y = pd.Series(y)

    model = train_model(X, y)
    return model, X


def test_get_top_k_features_orders_by_absolute_shap_value():
    shap_row = np.array([0.1, -0.9, 0.3, -0.05])
    feature_values = pd.Series([1.0, 2.0, 3.0, 4.0])
    feature_names = ["a", "b", "c", "d"]

    top = get_top_k_features(shap_row, feature_values, feature_names, k=2)

    assert [f.feature_name for f in top] == ["b", "c"]
    assert isinstance(top[0], FeatureContribution)


def test_explain_transaction_additivity_matches_model_output(trained_model_and_data):
    model, X = trained_model_and_data
    explainer = build_explainer(model)
    x_row = X.iloc[[0]]

    explanation = explain_transaction(explainer, x_row, k=2)

    # A soma (base_value + shap_sum) deve reproduzir a saida bruta (margin,
    # em log-odds) do modelo para essa transacao - e a propriedade aditiva
    # fundamental do SHAP que sustenta a interpretacao "gabarito" no projeto.
    reconstructed_margin = explanation["base_value"] + explanation["shap_sum"]
    actual_margin = model.predict(x_row, output_margin=True)[0]

    assert reconstructed_margin == pytest.approx(actual_margin, abs=1e-3)
    assert len(explanation["top_features"]) == 2


def test_explain_transaction_rejects_multiple_rows(trained_model_and_data):
    model, X = trained_model_and_data
    explainer = build_explainer(model)

    with pytest.raises(ValueError):
        explain_transaction(explainer, X.iloc[:2])


def test_feature_contribution_to_dict():
    contribution = FeatureContribution(feature_name="amount", feature_value=100.0, shap_value=0.5)

    result = contribution.to_dict()

    assert result == {"feature_name": "amount", "feature_value": 100.0, "shap_value": 0.5}
