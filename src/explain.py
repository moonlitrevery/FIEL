"""
Calculo de valores SHAP para explicar predicoes individuais do modelo de fraude.

Este modulo e o elo entre o classificador "caixa-preta" (XGBoost) e o gerador
de explicacoes em linguagem natural: para cada transacao suspeita, calculamos
os valores SHAP reais (nao aproximados por texto ou heuristica de prompt) e
extraimos as features que mais empurraram a predicao para "fraude". Esses
valores servem tanto de contexto factual para o prompt da LLM (generation.py)
quanto de "gabarito" para o verificador de fidelidade (faithfulness.py)
checar se a LLM realmente citou o que o modelo usou para decidir.

Usamos shap.TreeExplainer, que calcula valores SHAP EXATOS (nao aproximados
por amostragem, como o KernelExplainer generico) para modelos baseados em
arvore como o XGBoost, em tempo polinomial - viavel mesmo com milhoes de
transacoes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Numero padrao de features citadas na explicacao. Cinco e um numero
# pequeno o suficiente para caber numa explicacao em linguagem natural
# concisa, mas grande o suficiente para capturar a maioria do "peso" da
# decisao do modelo na pratica (poucas features costumam dominar o SHAP
# nas transacoes de fraude do PaySim).
DEFAULT_TOP_K = 5


@dataclass
class FeatureContribution:
    """Contribuicao de uma unica feature para a predicao de uma transacao."""

    feature_name: str
    feature_value: float
    shap_value: float

    def to_dict(self) -> dict:
        """Serializa a contribuicao para um dicionario simples (JSON-friendly)."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "shap_value": self.shap_value,
        }


def build_explainer(model: xgb.XGBClassifier) -> shap.TreeExplainer:
    """Cria o explainer SHAP para o modelo XGBoost ja treinado.

    O TreeExplainer explora a estrutura interna das arvores para calcular a
    contribuicao exata de cada feature em cada predicao, decompondo a saida
    do modelo (em log-odds) como: saida = valor_base + soma(shap_i para cada
    feature i). Essa decomposicao aditiva e o que torna o SHAP diferente de
    uma heuristica: e uma propriedade matematica exata do modelo treinado.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Calcula a matriz de valores SHAP (uma linha por transacao, uma coluna por feature).

    Os valores retornados estao no espaco de log-odds (saida bruta do
    XGBoost, antes da sigmoide) - e o espaco em que a decomposicao aditiva
    "valor_base + soma(shap)" e exata. Como o objetivo aqui e comparar a
    IMPORTANCIA RELATIVA das features entre si (quais pesam mais na decisao),
    e nao reportar probabilidades, trabalhar em log-odds nao prejudica a
    interpretacao.
    """
    return explainer.shap_values(X)


def get_top_k_features(
    shap_values_row: np.ndarray,
    feature_values_row: pd.Series,
    feature_names: list[str],
    k: int = DEFAULT_TOP_K,
) -> list[FeatureContribution]:
    """Extrai as k features com maior contribuicao ABSOLUTA para uma predicao.

    Ordenamos por |shap_value| (valor absoluto), nao pelo valor com sinal:
    uma feature que empurra fortemente para "fraude" e uma que empurra
    fortemente para "nao fraude" sao igualmente relevantes para EXPLICAR a
    decisao do modelo - o sinal indica a direcao do efeito, nao o quanto ele
    importou.
    """
    order = np.argsort(-np.abs(shap_values_row))[:k]
    return [
        FeatureContribution(
            feature_name=feature_names[i],
            feature_value=float(feature_values_row.iloc[i]),
            shap_value=float(shap_values_row[i]),
        )
        for i in order
    ]


def explain_transaction(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    k: int = DEFAULT_TOP_K,
) -> dict:
    """Explica uma unica transacao (DataFrame de exatamente 1 linha).

    Retorna um dicionario com:
    - base_value: valor esperado do modelo (log-odds medio no conjunto de
      treino usado para ajustar o explainer), o "ponto de partida" antes de
      considerar as features dessa transacao especifica.
    - shap_sum: soma de TODAS as contribuicoes SHAP (nao so as top-k), que
      somada ao base_value reproduz exatamente a saida do modelo em log-odds
      para essa transacao - serve como checagem de consistencia.
    - top_features: lista das k features mais influentes, como objetos
      FeatureContribution.
    """
    if len(X_row) != 1:
        raise ValueError("explain_transaction espera exatamente 1 linha (uma transacao).")

    shap_values = compute_shap_values(explainer, X_row)
    shap_row = shap_values[0]
    feature_names = list(X_row.columns)

    top_features = get_top_k_features(shap_row, X_row.iloc[0], feature_names, k=k)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]

    return {
        "base_value": float(base_value),
        "shap_sum": float(shap_row.sum()),
        "top_features": top_features,
    }


if __name__ == "__main__":
    # Execucao manual: `uv run python -m src.explain`
    from src.data_prep import load_raw_data, prepare_train_test_split
    from src.model import load_model

    raw_df = load_raw_data("data/paysim.csv")
    _, X_test, _, y_test, _, df_test = prepare_train_test_split(raw_df)

    model = load_model()
    explainer = build_explainer(model)

    # Pega a primeira fraude real do conjunto de teste como exemplo.
    fraud_index = y_test[y_test == 1].index[0]
    fraud_row = X_test.loc[[fraud_index]]

    explanation = explain_transaction(explainer, fraud_row)

    print(f"Transacao de teste (indice {fraud_index}), fraude real: {y_test.loc[fraud_index]}")
    print(f"Probabilidade prevista pelo modelo: {model.predict_proba(fraud_row)[0, 1]:.4f}")
    print(f"Valor base (log-odds medio): {explanation['base_value']:.4f}")
    print(f"Soma dos valores SHAP: {explanation['shap_sum']:.4f}")
    print("\nTop features que mais influenciaram a decisao:")
    for contribution in explanation["top_features"]:
        sinal = "empurra p/ FRAUDE" if contribution.shap_value > 0 else "empurra p/ legitima"
        print(
            f"  {contribution.feature_name:<20} valor={contribution.feature_value:>14.2f}  "
            f"shap={contribution.shap_value:>+8.4f}  ({sinal})"
        )
