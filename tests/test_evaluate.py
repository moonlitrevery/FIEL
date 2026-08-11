"""
Testes do modulo evaluate.

Reaproveita um pequeno pipeline sintetico (dados -> modelo -> explainer ->
biblioteca RAG indexada em diretorio temporario) montado uma vez por modulo
(mais custoso: treina um XGBoost e indexa embeddings), e testa a logica de
selecao de amostra, execucao por transacao (com um cliente Gemini falso) e
agregacao do relatorio final isoladamente.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_prep import prepare_train_test_split
from src.evaluate import (
    TransactionEvaluation,
    _make_json_safe,
    build_report,
    evaluate_sample,
    evaluate_single_transaction,
    save_report,
    select_flagged_transactions,
)
from src.explain import build_explainer
from src.model import train_model
from src.rag.retrieval import index_case_library


class _FakeModels:
    def __init__(self, response_text=None, exception=None):
        self.response_text = response_text
        self.exception = exception

    def generate_content(self, **kwargs):
        if self.exception is not None:
            raise self.exception
        return SimpleNamespace(text=self.response_text)


class FakeGeminiClient:
    """Imita o suficiente da interface google.genai.Client para os testes:
    client.models.generate_content(**kwargs) -> objeto com .text"""

    def __init__(self, response_text=None, exception=None):
        self.models = _FakeModels(response_text=response_text, exception=exception)


def make_raw_paysim_like_df(n: int = 400, fraud_rate: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Dataset sintetico com o schema bruto do PaySim, com um padrao de
    fraude claramente separavel (conta de origem esvaziada + saldo de
    destino inconsistente), suficiente para treinar um modelo de teste."""
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    types = np.where(rng.random(n) < 0.5, "TRANSFER", "CASH_OUT")
    amount = np.where(is_fraud == 1, rng.exponential(50_000, n), rng.exponential(500, n))
    old_orig = amount + rng.exponential(1000, n)
    new_orig = np.where(is_fraud == 1, 0.0, np.maximum(old_orig - amount, 0))
    old_dest = rng.exponential(2000, n)
    new_dest = np.where(is_fraud == 1, old_dest, old_dest + amount)

    return pd.DataFrame(
        {
            "step": rng.integers(1, 745, n),
            "type": types,
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": [f"M{i}" for i in range(n)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": is_fraud,
            "isFlaggedFraud": 0,
        }
    )


@pytest.fixture(scope="module")
def pipeline_fixture(tmp_path_factory):
    """Monta uma vez (por modulo de teste) todo o pipeline nao-LLM: dados,
    modelo, explainer e biblioteca RAG indexada."""
    raw_df = make_raw_paysim_like_df()
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    model = train_model(X_train, y_train)
    explainer = build_explainer(model)

    chroma_dir = tmp_path_factory.mktemp("chroma")
    collection = index_case_library(df_train, persist_directory=str(chroma_dir), max_cases=100)

    return SimpleNamespace(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        df_train=df_train,
        df_test=df_test,
        model=model,
        explainer=explainer,
        collection=collection,
    )


def test_select_flagged_transactions_splits_by_actual_label_and_respects_sample_size():
    class StubModel:
        def predict_proba(self, X):
            # A metade das linhas recebe probabilidade alta (flagrada); a
            # outra metade, baixa (nao flagrada).
            n = len(X)
            high = np.ones(n // 2) * 0.9
            low = np.ones(n - n // 2) * 0.1
            probs_fraud = np.concatenate([high, low])
            return np.column_stack([1 - probs_fraud, probs_fraud])

    n = 20
    X_test = pd.DataFrame({"amount": range(n)})
    # Entre as flagradas (indices 0-9): metade fraude real, metade nao.
    y_test = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] + [0] * 10)

    tp_indices, fp_indices = select_flagged_transactions(
        StubModel(), X_test, y_test, threshold=0.5, n_true_positives=2, n_false_positives=2
    )

    assert len(tp_indices) == 2
    assert len(fp_indices) == 2
    assert all(y_test.loc[i] == 1 for i in tp_indices)
    assert all(y_test.loc[i] == 0 for i in fp_indices)
    # Os indices devem vir do grupo flagrado (0-9), nunca do nao flagrado (10-19).
    assert all(i < 10 for i in tp_indices + fp_indices)


def test_select_flagged_transactions_is_deterministic_given_same_seed():
    class StubModel:
        def predict_proba(self, X):
            probs_fraud = np.ones(len(X)) * 0.9
            return np.column_stack([1 - probs_fraud, probs_fraud])

    X_test = pd.DataFrame({"amount": range(50)})
    y_test = pd.Series([1] * 25 + [0] * 25)

    run_1 = select_flagged_transactions(StubModel(), X_test, y_test, n_true_positives=5, n_false_positives=5, random_state=7)
    run_2 = select_flagged_transactions(StubModel(), X_test, y_test, n_true_positives=5, n_false_positives=5, random_state=7)

    assert run_1 == run_2


def test_evaluate_single_transaction_success_labels_fraud_as_verdadeiro_positivo(pipeline_fixture):
    fixture = pipeline_fixture
    fraud_index = fixture.y_test[fixture.y_test == 1].index[0]

    fake_client = FakeGeminiClient(
        response_text=(
            '{"narrativa": "explicacao de teste", '
            '"features_citadas": ["error_balance_orig"], '
            '"acao_recomendada": "bloquear"}'
        )
    )

    result = evaluate_single_transaction(
        fraud_index, fixture.df_test, fixture.X_test, fixture.model, fixture.explainer, fixture.collection,
        client=fake_client,
    )

    assert isinstance(result, TransactionEvaluation)
    assert result.flagged_as == "verdadeiro_positivo"
    assert result.error is None
    assert result.generated_explanation is not None
    assert result.faithfulness is not None
    assert len(result.top_shap_features) > 0


def test_evaluate_single_transaction_labels_legit_as_falso_positivo(pipeline_fixture):
    fixture = pipeline_fixture
    legit_index = fixture.y_test[fixture.y_test == 0].index[0]

    fake_client = FakeGeminiClient(
        response_text='{"narrativa": "x", "features_citadas": [], "acao_recomendada": "liberar"}'
    )

    result = evaluate_single_transaction(
        legit_index, fixture.df_test, fixture.X_test, fixture.model, fixture.explainer, fixture.collection,
        client=fake_client,
    )

    assert result.flagged_as == "falso_positivo"
    assert result.error is None


def test_evaluate_single_transaction_captures_api_errors_without_raising(pipeline_fixture):
    fixture = pipeline_fixture
    fraud_index = fixture.y_test[fixture.y_test == 1].index[0]

    fake_client = FakeGeminiClient(exception=RuntimeError("falha simulada de rede"))

    result = evaluate_single_transaction(
        fraud_index, fixture.df_test, fixture.X_test, fixture.model, fixture.explainer, fixture.collection,
        client=fake_client,
    )

    assert result.error is not None
    assert "falha simulada de rede" in result.error
    assert result.generated_explanation is None
    assert result.faithfulness is None


def test_evaluate_sample_runs_pipeline_for_multiple_indices(pipeline_fixture):
    fixture = pipeline_fixture
    indices = list(fixture.y_test.index[:3])

    fake_client = FakeGeminiClient(
        response_text='{"narrativa": "x", "features_citadas": [], "acao_recomendada": "revisar"}'
    )

    results = evaluate_sample(
        indices, fixture.df_test, fixture.X_test, fixture.model, fixture.explainer, fixture.collection,
        client=fake_client,
    )

    assert len(results) == 3
    assert all(isinstance(r, TransactionEvaluation) for r in results)


def test_build_report_aggregates_only_successful_evaluations_and_counts_failures():
    successful = TransactionEvaluation(
        transaction_id="1",
        actual_label=1,
        predicted_probability=0.9,
        flagged_as="verdadeiro_positivo",
        generated_explanation={"narrativa": "x", "features_citadas": ["amount"], "acao_recomendada": "bloquear"},
        faithfulness={
            "cited_features": ["amount"],
            "real_top_features": ["amount"],
            "correctly_cited": ["amount"],
            "hallucinated_features": [],
            "irrelevant_features": [],
            "omitted_features": [],
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "weighted_recall": 1.0,
            "is_faithful": True,
        },
    )
    failed = TransactionEvaluation(
        transaction_id="2",
        actual_label=0,
        predicted_probability=0.7,
        flagged_as="falso_positivo",
        error="RuntimeError: falha de API",
    )

    report = build_report([successful, failed], model_metrics={"auc_pr": np.float64(0.95)})

    assert report["n_evaluated"] == 2
    assert report["n_failed"] == 1
    assert report["faithfulness_summary"]["n_explanations"] == 1
    assert report["faithfulness_summary"]["faithful_rate"] == 1.0
    # numpy.float64 no metrics do modelo deve ter sido convertido para float nativo.
    assert isinstance(report["model_metrics"]["auc_pr"], float)
    assert type(report["model_metrics"]["auc_pr"]) is float


def test_make_json_safe_converts_numpy_scalars_recursively():
    payload = {"a": np.int64(5), "b": [np.float64(1.5), {"c": np.int64(2)}]}

    safe = _make_json_safe(payload)

    assert safe == {"a": 5, "b": [1.5, {"c": 2}]}
    assert type(safe["a"]) is int
    assert type(safe["b"][0]) is float


def test_save_report_writes_valid_json_file(tmp_path):
    import json

    report = {"model_metrics": {"auc_pr": 0.9}, "evaluations": []}
    path = str(tmp_path / "subdir" / "report.json")

    save_report(report, path)

    with open(path, encoding="utf-8") as file:
        loaded = json.load(file)

    assert loaded == report
