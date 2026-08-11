"""
Execucao do pipeline completo (SHAP + RAG + geracao + fidelidade) sobre uma
amostra do conjunto de teste, produzindo um relatorio agregado.

Este e o modulo que "junta tudo": para cada transacao selecionada do teste,
roda a sequencia completa data -> modelo -> SHAP -> RAG -> LLM -> verificador
de fidelidade, e agrega os resultados numa taxa de fidelidade geral, pronta
para citar na secao de resultados do relatorio.

Selecionamos as transacoes avaliadas dentre as que o MODELO flagrou como
suspeitas (probabilidade >= limiar de decisao) - nao apenas as fraudes reais
- porque e exatamente esse o cenario de uso do sistema descrito no projeto:
"dado uma transacao flagrada como suspeita". Separamos a amostra em
verdadeiros positivos (fraude real, o modelo acertou) e falsos positivos
(o modelo errou, mas ainda assim precisa de uma explicacao) de proposito:
e uma pergunta interessante para a analise critica se a fidelidade da
explicacao muda entre esses dois grupos.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data_prep import RANDOM_STATE, load_raw_data, prepare_train_test_split
from src.explain import DEFAULT_TOP_K as DEFAULT_TOP_K_SHAP
from src.explain import build_explainer, explain_transaction
from src.faithfulness import FaithfulnessResult, aggregate_faithfulness_results, check_faithfulness
from src.generation import generate_explanation
from src.model import DEFAULT_MODEL_PATH
from src.model import DEFAULT_THRESHOLD as DECISION_THRESHOLD
from src.model import evaluate_model, load_model, save_model, train_model
from src.rag.retrieval import DEFAULT_TOP_K as DEFAULT_TOP_K_RAG
from src.rag.retrieval import index_case_library, retrieve_similar_cases_for_transaction

# Reaproveitamos o limiar padrao de model.py (DEFAULT_THRESHOLD) como
# DECISION_THRESHOLD, para que a definicao de "transacao flagrada como
# suspeita" seja a MESMA usada no relatorio de metricas do classificador -
# evita o risco de um numero "magico" duplicado ficar dessincronizado se um
# dos dois for alterado no futuro.

# Tamanho padrao de cada grupo da amostra avaliada. Numeros pequenos de
# proposito: cada transacao avaliada custa 1 chamada real a API Gemini: um
# numero grande deixaria a avaliacao lenta e cara sem agregar muito rigor
# estatistico extra a uma metrica que ja e bem estavel (ver relatorio final).
DEFAULT_N_TRUE_POSITIVES = 15
DEFAULT_N_FALSE_POSITIVES = 15

DEFAULT_RESULTS_PATH = "results/evaluation_report.json"


@dataclass
class TransactionEvaluation:
    """Resultado completo da avaliacao de UMA transacao: dados do modelo,
    explicacao gerada e verificacao de fidelidade. Se algo falhar no meio do
    caminho (ex: erro de rede na API), o erro fica registrado aqui em vez de
    interromper a avaliacao das demais transacoes da amostra."""

    transaction_id: str
    actual_label: int
    predicted_probability: float
    flagged_as: str  # "verdadeiro_positivo" ou "falso_positivo"
    top_shap_features: list[dict] = field(default_factory=list)
    similar_cases: list[dict] = field(default_factory=list)
    generated_explanation: dict | None = None
    faithfulness: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "actual_label": self.actual_label,
            "predicted_probability": self.predicted_probability,
            "flagged_as": self.flagged_as,
            "top_shap_features": self.top_shap_features,
            "similar_cases": self.similar_cases,
            "generated_explanation": self.generated_explanation,
            "faithfulness": self.faithfulness,
            "error": self.error,
        }


def select_flagged_transactions(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DECISION_THRESHOLD,
    n_true_positives: int = DEFAULT_N_TRUE_POSITIVES,
    n_false_positives: int = DEFAULT_N_FALSE_POSITIVES,
    random_state: int = RANDOM_STATE,
) -> tuple[list, list]:
    """Seleciona, dentre as transacoes que o modelo flagrou como suspeitas
    (probabilidade >= threshold), uma amostra aleatoria (com seed fixa) de
    verdadeiros positivos e outra de falsos positivos.

    Retorna dois arrays de indices (compativeis com X_test.loc[...] /
    df_test.loc[...]): (indices_verdadeiros_positivos, indices_falsos_positivos).
    """
    scores = model.predict_proba(X_test)[:, 1]
    flagged_mask = scores >= threshold

    flagged_labels = y_test[flagged_mask]
    true_positive_indices = list(flagged_labels[flagged_labels == 1].index)
    false_positive_indices = list(flagged_labels[flagged_labels == 0].index)

    rng = random.Random(random_state)
    tp_sample = rng.sample(true_positive_indices, min(n_true_positives, len(true_positive_indices)))
    fp_sample = rng.sample(false_positive_indices, min(n_false_positives, len(false_positive_indices)))

    return tp_sample, fp_sample


def evaluate_single_transaction(
    index,
    df_test: pd.DataFrame,
    X_test: pd.DataFrame,
    model,
    explainer,
    collection,
    top_k_shap: int = DEFAULT_TOP_K_SHAP,
    top_k_rag: int = DEFAULT_TOP_K_RAG,
    client=None,
) -> TransactionEvaluation:
    """Roda o pipeline completo (SHAP -> RAG -> LLM -> fidelidade) para uma
    unica transacao e retorna o resultado estruturado.

    Erros na geracao da explicacao (rede, parsing do JSON etc.) sao
    capturados e registrados em `error`, em vez de propagados - uma falha
    isolada numa chamada de API nao deve derrubar a avaliacao da amostra
    inteira.
    """
    x_row = X_test.loc[[index]]
    full_row = df_test.loc[index]

    predicted_probability = float(model.predict_proba(x_row)[0, 1])
    actual_label = int(full_row["isFraud"])
    flagged_as = "verdadeiro_positivo" if actual_label == 1 else "falso_positivo"

    explanation_data = explain_transaction(explainer, x_row, k=top_k_shap)
    top_features = explanation_data["top_features"]
    similar_cases = retrieve_similar_cases_for_transaction(full_row, collection, k=top_k_rag)

    evaluation = TransactionEvaluation(
        transaction_id=str(index),
        actual_label=actual_label,
        predicted_probability=predicted_probability,
        flagged_as=flagged_as,
        top_shap_features=[f.to_dict() for f in top_features],
        similar_cases=[
            {"case_id": case.case_id, "similarity": case.similarity} for case in similar_cases
        ],
    )

    try:
        generated = generate_explanation(
            row=full_row,
            predicted_probability=predicted_probability,
            top_features=top_features,
            similar_cases=similar_cases,
            client=client,
        )
        faithfulness_result = check_faithfulness(generated, top_features)

        evaluation.generated_explanation = generated.to_dict()
        evaluation.faithfulness = faithfulness_result.to_dict()
    except Exception as error:  # noqa: BLE001 - queremos capturar qualquer falha da API/parsing
        evaluation.error = f"{type(error).__name__}: {error}"

    return evaluation


def evaluate_sample(
    indices: list,
    df_test: pd.DataFrame,
    X_test: pd.DataFrame,
    model,
    explainer,
    collection,
    top_k_shap: int = DEFAULT_TOP_K_SHAP,
    top_k_rag: int = DEFAULT_TOP_K_RAG,
    client=None,
) -> list[TransactionEvaluation]:
    """Aplica evaluate_single_transaction a cada indice da amostra."""
    return [
        evaluate_single_transaction(
            index, df_test, X_test, model, explainer, collection, top_k_shap, top_k_rag, client
        )
        for index in indices
    ]


def _make_json_safe(value):
    """Converte tipos numpy (int64/float64), comuns em resultados do pandas/
    sklearn, para tipos nativos do Python antes de serializar em JSON."""
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_report(evaluations: list[TransactionEvaluation], model_metrics: dict) -> dict:
    """Monta o relatorio final: metricas do classificador, fidelidade
    agregada (so sobre as avaliacoes sem erro) e o detalhe de cada transacao."""
    successful = [e for e in evaluations if e.error is None]
    failed = [e for e in evaluations if e.error is not None]

    faithfulness_results = [FaithfulnessResult(**e.faithfulness) for e in successful]
    aggregate = aggregate_faithfulness_results(faithfulness_results)

    report = {
        "model_metrics": model_metrics,
        "faithfulness_summary": aggregate.to_dict(),
        "n_evaluated": len(evaluations),
        "n_failed": len(failed),
        "evaluations": [e.to_dict() for e in evaluations],
    }
    return _make_json_safe(report)


def save_report(report: dict, path: str = DEFAULT_RESULTS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def get_or_train_model(X_train, y_train, model_path: str = DEFAULT_MODEL_PATH):
    """Reaproveita um modelo ja treinado e salvo em disco, se existir;
    caso contrario, treina um novo e salva - evita retreinar a cada
    execucao da avaliacao (o treino sozinho ja leva ~20s no dataset completo)."""
    if os.path.exists(model_path):
        return load_model(model_path)
    model = train_model(X_train, y_train)
    save_model(model, model_path)
    return model


def run_evaluation(
    csv_path: str = "data/paysim.csv",
    n_true_positives: int = DEFAULT_N_TRUE_POSITIVES,
    n_false_positives: int = DEFAULT_N_FALSE_POSITIVES,
    max_rag_cases: int | None = None,
    decision_threshold: float = DECISION_THRESHOLD,
    results_path: str = DEFAULT_RESULTS_PATH,
    client=None,
) -> dict:
    """Orquestra o pipeline completo, do CSV bruto ao relatorio final salvo em disco.

    Fluxo: carrega e prepara os dados -> carrega/treina o modelo -> avalia o
    classificador (AUC-PR) -> indexa a biblioteca de casos do RAG a partir do
    treino -> seleciona a amostra de transacoes flagradas no teste -> roda o
    pipeline completo em cada uma -> agrega a fidelidade -> salva o relatorio.
    """
    raw_df = load_raw_data(csv_path)
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    model = get_or_train_model(X_train, y_train)
    model_metrics = evaluate_model(model, X_test, y_test, threshold=decision_threshold)

    explainer = build_explainer(model)

    print("Indexando biblioteca de casos do RAG a partir do treino...")
    collection = index_case_library(df_train, max_cases=max_rag_cases)

    tp_indices, fp_indices = select_flagged_transactions(
        model, X_test, y_test, threshold=decision_threshold,
        n_true_positives=n_true_positives, n_false_positives=n_false_positives,
    )
    print(
        f"Avaliando {len(tp_indices)} verdadeiros positivos e "
        f"{len(fp_indices)} falsos positivos flagrados pelo modelo..."
    )

    evaluations = evaluate_sample(tp_indices + fp_indices, df_test, X_test, model, explainer, collection, client=client)

    report = build_report(evaluations, model_metrics)
    save_report(report, results_path)

    print(f"\nRelatorio salvo em {results_path}")
    print(f"AUC-PR do modelo: {model_metrics['auc_pr']:.4f}")
    print(f"Explicacoes avaliadas com sucesso: {report['n_evaluated'] - report['n_failed']}/{report['n_evaluated']}")
    print(f"Taxa de fidelidade (F1 >= limiar): {report['faithfulness_summary']['faithful_rate']:.2%}")
    print(f"Taxa de alucinacao (citou feature inexistente): {report['faithfulness_summary']['hallucination_rate']:.2%}")
    print(f"Recall ponderado medio: {report['faithfulness_summary']['mean_weighted_recall']:.4f}")

    return report


if __name__ == "__main__":
    # Execucao manual: `uv run python -m src.evaluate`
    # Requer GEMINI_API_KEY configurada e data/paysim.csv presente.
    run_evaluation()
