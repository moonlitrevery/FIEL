"""
Testes do modulo data_prep.

Usamos DataFrames sinteticos pequenos (em memoria, sem depender do CSV real
do PaySim) que imitam o schema original, para validar a logica de filtro,
feature engineering e split sem precisar do dataset completo baixado.
"""

import pandas as pd

from src.data_prep import (
    engineer_features,
    filter_relevant_transaction_types,
    get_feature_columns,
    prepare_train_test_split,
)


def make_fake_paysim_df() -> pd.DataFrame:
    """Cria um DataFrame com o mesmo schema de colunas do PaySim.

    A linha de indice 4 simula o padrao classico de fraude do PaySim: a
    conta de origem e esvaziada (oldbalanceOrg -> newbalanceOrig = 0), mas o
    saldo da conta de destino NAO reflete o valor recebido (oldbalanceDest
    permanece igual a newbalanceDest), gerando uma inconsistencia contabil.
    """
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 6],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
            "amount": [100.0, 200.0, 50.0, 300.0, 900.0, 20.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "oldbalanceOrg": [100.0, 200.0, 50.0, 300.0, 900.0, 20.0],
            "newbalanceOrig": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "nameDest": ["M1", "M2", "M3", "M4", "M5", "M6"],
            "oldbalanceDest": [0.0, 0.0, 0.0, 0.0, 500.0, 0.0],
            "newbalanceDest": [100.0, 200.0, 50.0, 300.0, 500.0, 20.0],
            "isFraud": [1, 0, 0, 0, 1, 0],
            "isFlaggedFraud": [0, 0, 0, 0, 0, 0],
        }
    )


def make_fake_paysim_df_for_split(n_per_class: int = 10) -> pd.DataFrame:
    """Cria um DataFrame maior, com classes balanceadas o suficiente para
    que o split estratificado treino/teste seja viavel (sklearn exige um
    numero minimo de amostras por classe)."""
    rows = []
    for i in range(n_per_class):
        rows.append(
            {
                "step": i + 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "nameOrig": f"C{i}",
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 0.0,
                "nameDest": f"M{i}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 1,
                "isFlaggedFraud": 0,
            }
        )
        rows.append(
            {
                "step": i + 1,
                "type": "CASH_OUT",
                "amount": 100.0,
                "nameOrig": f"D{i}",
                "oldbalanceOrg": 200.0,
                "newbalanceOrig": 100.0,
                "nameDest": f"N{i}",
                "oldbalanceDest": 50.0,
                "newbalanceDest": 150.0,
                "isFraud": 0,
                "isFlaggedFraud": 0,
            }
        )
    return pd.DataFrame(rows)


def test_filter_relevant_transaction_types_keeps_only_transfer_and_cash_out():
    df = make_fake_paysim_df()

    filtered = filter_relevant_transaction_types(df)

    assert set(filtered["type"].unique()) == {"TRANSFER", "CASH_OUT"}
    assert len(filtered) == 3


def test_engineer_features_creates_expected_columns():
    df = filter_relevant_transaction_types(make_fake_paysim_df())

    enriched = engineer_features(df)

    for column in ["error_balance_orig", "error_balance_dest", "hour_of_day", "type_TRANSFER"]:
        assert column in enriched.columns

    # Linha de fraude "classica": saldo de destino nao atualizado apesar do
    # valor recebido -> error_balance_dest deve capturar essa inconsistencia.
    fraud_row = enriched[enriched["isFraud"] == 1].iloc[-1]
    assert fraud_row["error_balance_dest"] != 0


def test_get_feature_columns_matches_engineered_dataframe():
    df = engineer_features(filter_relevant_transaction_types(make_fake_paysim_df()))

    feature_columns = get_feature_columns()

    assert set(feature_columns).issubset(set(df.columns))


def test_prepare_train_test_split_shapes_and_stratification():
    df = make_fake_paysim_df_for_split(n_per_class=10)

    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_split(df)

    assert list(X_train.columns) == get_feature_columns()
    assert len(X_train) + len(X_test) == len(df_train) + len(df_test)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    # A estratificacao deve manter as duas classes presentes em ambos os
    # conjuntos, mesmo com o dataset pequeno usado no teste.
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    # df_train/df_test devem preservar as colunas identificadoras originais,
    # necessarias para a biblioteca de casos do RAG.
    assert "nameOrig" in df_train.columns
    assert "nameDest" in df_train.columns
