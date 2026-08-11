"""
Testes do modulo model.

Usamos um dataset sintetico simples e claramente separavel (fraude tem
amount e error_balance muito maiores que o normal) para verificar que o
pipeline de treino, avaliacao e serializacao funciona de ponta a ponta, sem
depender do PaySim real nem exigir que o modelo atinja uma performance
minima especifica.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    compute_scale_pos_weight,
    evaluate_model,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def synthetic_dataset():
    """Gera X/y sinteticos com um padrao de fraude facil de aprender,
    grande o suficiente para o early stopping e a estratificacao funcionarem."""
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

    split = int(n * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def test_compute_scale_pos_weight_matches_class_ratio():
    y_train = pd.Series([0] * 90 + [1] * 10)

    weight = compute_scale_pos_weight(y_train)

    assert weight == pytest.approx(9.0)


def test_train_model_learns_separable_pattern(synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # Com um padrao tao separavel, o AUC-PR deve ficar bem acima do baseline
    # (que seria proximo da taxa de fraude, ~0.05).
    assert metrics["auc_pr"] > 0.7
    assert metrics["n_test"] == len(y_test)
    assert metrics["n_fraud_test"] == int(y_test.sum())


def test_save_and_load_model_preserves_predictions(tmp_path, synthetic_dataset):
    X_train, X_test, y_train, y_test = synthetic_dataset
    model = train_model(X_train, y_train)

    model_path = str(tmp_path / "model.json")
    save_model(model, model_path)
    loaded_model = load_model(model_path)

    original_scores = model.predict_proba(X_test)[:, 1]
    loaded_scores = loaded_model.predict_proba(X_test)[:, 1]

    np.testing.assert_allclose(original_scores, loaded_scores)
