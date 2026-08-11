"""
Carregamento e preparo dos dados do PaySim.

Este modulo e responsavel por:
1. Carregar o CSV bruto do PaySim.
2. Filtrar apenas os tipos de transacao onde fraude de fato ocorre no dataset
   (TRANSFER e CASH_OUT) - nos demais tipos (PAYMENT, CASH_IN, DEBIT) o
   PaySim nunca marca isFraud=1, entao mante-los so adicionaria ruido e
   pioraria ainda mais o desbalanceamento de classes.
3. Fazer feature engineering simples e explicavel (erros de saldo e hora do
   dia), que sao justamente os sinais que o SHAP costuma apontar como mais
   relevantes para fraude nesse dataset.
4. Gerar a divisao treino/teste estratificada por classe, preservando tanto
   a matriz de features (para o modelo) quanto o DataFrame original completo
   (para a biblioteca de casos do RAG e para exibir a transacao "crua" na UI).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# Semente fixa para reprodutibilidade (mesmo split em todo o pipeline:
# treino do modelo, biblioteca de casos do RAG e avaliacao final).
RANDOM_STATE = 42

# Proporcao reservada para teste. O restante (80%) e usado para treinar o
# XGBoost e tambem para popular a biblioteca de casos do RAG - ou seja, o
# verificador de fidelidade e o RAG nunca "veem" as transacoes de teste.
TEST_SIZE = 0.2

# No PaySim, fraude so acontece nos tipos TRANSFER (transferencia entre
# contas) e CASH_OUT (saque). E um fato conhecido e documentado do dataset,
# nao uma suposicao nossa.
RELEVANT_TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT"]

# Colunas identificadoras: uteis para rastrear a transacao original (ex: no
# texto serializado da biblioteca de casos), mas nao devem entrar como
# feature do modelo por serem identificadores de altissima cardinalidade
# (praticamente unicos por linha, o modelo so decoraria o dataset de treino).
ID_COLUMNS = ["nameOrig", "nameDest"]

# Coluna de vazamento de dados (data leakage): isFlaggedFraud e um sinalizador
# de uma regra simples e legada do proprio simulador do PaySim (transferencias
# acima de um limiar fixo), que quase nunca dispara e nao reflete um processo
# de deteccao real. Mante-la como feature enviesaria o modelo.
LEAKY_COLUMNS = ["isFlaggedFraud"]

TARGET_COLUMN = "isFraud"


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV bruto do PaySim em um DataFrame do pandas.

    Parametros
    ----------
    csv_path: caminho para o arquivo paysim.csv (ex: "data/paysim.csv").

    Retorna
    -------
    DataFrame com as colunas originais do PaySim, sem nenhum filtro ou
    transformacao aplicada.
    """
    return pd.read_csv(csv_path)


def filter_relevant_transaction_types(df: pd.DataFrame) -> pd.DataFrame:
    """Mantem apenas os tipos de transacao onde fraude realmente ocorre.

    Reduz o volume de dados (de ~6.3M para ~2.7M linhas) e melhora a
    proporcao de fraudes no dataset, sem descartar nenhum caso positivo.
    """
    filtered = df[df["type"].isin(RELEVANT_TRANSACTION_TYPES)].copy()
    return filtered.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas que tornam os padroes de fraude mais visiveis.

    - error_balance_orig / error_balance_dest: no PaySim, fraudadores costumam
      esvaziar a conta de origem (oldbalanceOrg -> 0) sem que o saldo da conta
      de destino seja atualizado de forma consistente. Essas colunas medem a
      inconsistencia contabil entre saldo antes/depois e o valor transacionado,
      e sao historicamente as features mais discriminativas nesse dataset.
    - hour_of_day: a coluna "step" do PaySim representa 1 hora de simulacao
      (744 steps = 30 dias). Extraimos a hora do dia (0-23) como proxy de
      padroes temporais (ex: fraude concentrada em horarios de menor
      vigilancia).
    - type_TRANSFER: indicador binario do tipo de transacao (apos o filtro,
      restam apenas TRANSFER e CASH_OUT, entao uma unica coluna binaria basta
      para representar os dois tipos).
    """
    df = df.copy()

    df["error_balance_orig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["error_balance_dest"] = (
        df["oldbalanceOrg"] * 0 + df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    df["hour_of_day"] = df["step"] % 24
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Lista fixa e ordenada das colunas usadas como entrada (X) do modelo.

    Manter essa lista centralizada garante que model.py, explain.py e
    generation.py sempre se refiram as mesmas features, na mesma ordem -
    essencial para que os indices dos valores SHAP batam com os nomes das
    colunas em todo o pipeline.
    """
    return [
        "step",
        "hour_of_day",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "error_balance_orig",
        "error_balance_dest",
        "type_TRANSFER",
    ]


def prepare_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Aplica o filtro, o feature engineering e divide em treino/teste.

    Retorna
    -------
    X_train, X_test: matrizes de features (apenas colunas de get_feature_columns()).
    y_train, y_test: rotulos binarios (isFraud).
    df_train, df_test: DataFrames completos (features + colunas originais e
        identificadoras), na mesma ordem/indice de X_train/X_test. Sao usados
        por rag/case_library.py (para montar a biblioteca de casos a partir do
        treino) e por evaluate.py (para ter acesso aos dados "crus" da
        transacao de teste sendo explicada).
    A divisao e estratificada por isFraud para preservar a mesma proporcao
    (muito baixa) de fraudes em treino e teste.
    """
    filtered = filter_relevant_transaction_types(df)
    enriched = engineer_features(filtered)

    feature_columns = get_feature_columns()
    X = enriched[feature_columns]
    y = enriched[TARGET_COLUMN]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        df_train,
        df_test,
    ) = train_test_split(
        X,
        y,
        enriched,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, df_train, df_test


if __name__ == "__main__":
    # Execucao manual para checagem rapida: `uv run src/data_prep.py`
    raw_df = load_raw_data("data/paysim.csv")
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    print(f"Linhas totais (bruto): {len(raw_df):,}")
    print(f"Linhas apos filtro TRANSFER/CASH_OUT: {len(X_train) + len(X_test):,}")
    print(f"Treino: {len(X_train):,} linhas | Teste: {len(X_test):,} linhas")
    print(f"Taxa de fraude (treino): {y_train.mean():.4%}")
    print(f"Taxa de fraude (teste): {y_test.mean():.4%}")
    print(f"Features do modelo: {get_feature_columns()}")
