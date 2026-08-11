"""
Treino e avaliacao do classificador de fraude (XGBoost).

Este modulo e responsavel por:
1. Treinar um XGBoost binario para detectar fraude, usando os dados ja
   preparados por data_prep.py.
2. Lidar com o desbalanceamento extremo de classes (fraude é < 1% dos casos)
   via scale_pos_weight, em vez de tecnicas de reamostragem (undersampling/
   SMOTE), que distorceriam a distribuicao real das features usadas depois
   pelo SHAP.
3. Avaliar o modelo com AUC-PR (area sob a curva Precision-Recall), a metrica
   correta para datasets desbalanceados - acuracia seria enganosa aqui (um
   modelo que sempre prediz "nao fraude" já teria >99% de acuracia).
4. Salvar/carregar o modelo treinado em formato nativo do XGBoost (JSON), que
   e usado por explain.py, generation.py e evaluate.py.
"""

from __future__ import annotations

import os

import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from src.data_prep import RANDOM_STATE

# Caminho padrao onde o modelo treinado e salvo. Formato nativo do XGBoost
# (JSON) em vez de pickle: e estavel entre versoes da biblioteca e mais
# transparente (da pra abrir o arquivo e ver os parametros), o que facilita
# explicar na apresentacao.
DEFAULT_MODEL_PATH = "models/xgboost_fraud.json"

# Fracao do treino separada como conjunto de validacao, usada apenas para
# early stopping (monitorar overfitting durante o treino). O modelo nunca
# treina diretamente sobre essa fatia.
VALIDATION_SIZE = 0.1

# Numero de rodadas sem melhora na AUC-PR de validacao antes de parar o
# treino. Evita que o modelo "decore" os poucos exemplos de fraude do treino.
EARLY_STOPPING_ROUNDS = 20

# Limiar de decisao padrao (probabilidade >= 0.5 -> classificado como fraude).
# E usado so para o relatorio de classificacao no console; o restante do
# pipeline (explain.py, generation.py) trabalha com a probabilidade continua.
DEFAULT_THRESHOLD = 0.5


def compute_scale_pos_weight(y_train) -> float:
    """Calcula o peso da classe positiva (fraude) para o XGBoost.

    A formula padrao e (numero de negativos / numero de positivos). Isso faz
    com que erros ao classificar uma fraude como legitima custem mais caro,
    durante o treino, do que o inverso - compensando o fato de fraude ser
    uma classe rara sem alterar artificialmente os dados.
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    return n_negative / n_positive


def train_model(X_train, y_train, random_state: int = RANDOM_STATE) -> xgb.XGBClassifier:
    """Treina o classificador XGBoost com early stopping.

    Separa uma fatia do treino como validacao (estratificada) so para
    decidir quando parar de adicionar arvores - o modelo final e o mesmo
    objeto, so que com o numero de arvores (best_iteration) escolhido pela
    validacao, e nao um numero fixo arbitrario.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=random_state,
        stratify=y_train,
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=compute_scale_pos_weight(y_tr),
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate_model(model: xgb.XGBClassifier, X_test, y_test, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Avalia o modelo no conjunto de teste usando AUC-PR como metrica principal.

    Retorna um dicionario com:
    - auc_pr: area sob a curva Precision-Recall (metrica principal, robusta a
      desbalanceamento).
    - precision / recall / f1 no limiar fixo `threshold` (para dar uma nocao
      concreta de operacao do modelo, alem da metrica agregada).
    - n_test / n_fraud_test: tamanho do conjunto de teste e quantos sao fraude
      de fato, para contextualizar os numeros acima.
    """
    y_scores = model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    true_positives = ((y_pred == 1) & (y_test == 1)).sum()
    predicted_positives = (y_pred == 1).sum()
    actual_positives = (y_test == 1).sum()

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "n_test": len(y_test),
        "n_fraud_test": int(actual_positives),
    }


def get_best_threshold_by_f1(model: xgb.XGBClassifier, X_val, y_val) -> float:
    """Varre a curva Precision-Recall e retorna o limiar que maximiza o F1.

    Util para escolher um limiar de operacao mais informado do que o padrao
    0.5, que raramente e ideal em problemas desbalanceados. Nao e usado
    automaticamente em evaluate_model() para manter a funcao de avaliacao
    determinística e comparavel entre execucoes; fica disponivel para quem
    quiser calibrar o limiar de decisao do sistema.
    """
    y_scores = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)

    # precision_recall_curve retorna um ponto a mais que thresholds (o ultimo
    # ponto, correspondente a threshold=1.0); descartamos para alinhar os arrays.
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_index = f1_scores.argmax()
    return float(thresholds[best_index])


def save_model(model: xgb.XGBClassifier, path: str = DEFAULT_MODEL_PATH) -> None:
    """Salva o modelo treinado em formato nativo do XGBoost (JSON)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def load_model(path: str = DEFAULT_MODEL_PATH) -> xgb.XGBClassifier:
    """Carrega um modelo previamente salvo por save_model()."""
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


if __name__ == "__main__":
    # Execucao manual: `uv run python -m src.model`
    from src.data_prep import load_raw_data, prepare_train_test_split

    raw_df = load_raw_data("data/paysim.csv")
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    print(f"Treinando XGBoost em {len(X_train):,} transacoes "
          f"({y_train.sum():,} fraudes, {y_train.mean():.4%})...")
    model = train_model(X_train, y_train)
    print(f"Numero de arvores escolhido por early stopping: {model.best_iteration + 1}")

    metrics = evaluate_model(model, X_test, y_test)
    print("\nAvaliacao no conjunto de teste:")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} (threshold={metrics['threshold']})")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Teste: {metrics['n_test']:,} transacoes, {metrics['n_fraud_test']:,} fraudes reais")

    save_model(model)
    print(f"\nModelo salvo em {DEFAULT_MODEL_PATH}")
