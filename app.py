"""
Dashboard Streamlit: demonstracao interativa do assistente de explicabilidade de fraude.

Este arquivo e so a camada de apresentacao - toda a logica de negocio (dados,
modelo, SHAP, RAG, geracao via LLM, verificacao de fidelidade) ja existe nos
modulos de src/. O app apenas: (1) carrega tudo uma vez e mantem em cache
(st.cache_resource), (2) deixa o usuario escolher uma transacao suspeita de
exemplo, e (3) chama o pipeline sob demanda, exibindo cada etapa (SHAP, RAG,
explicacao gerada, verificacao de fidelidade) de forma visual.

A chamada ao LLM (que custa dinheiro e tempo) so acontece quando o usuario
clica explicitamente em "Gerar explicacao" - nunca automaticamente ao trocar
de transacao - para manter o controle de custo/latencia durante uma
demonstracao ao vivo.
"""

from __future__ import annotations

import os

import anthropic
import pandas as pd
import streamlit as st

from src.data_prep import load_raw_data, prepare_train_test_split
from src.evaluate import get_or_train_model, select_flagged_transactions
from src.explain import build_explainer, explain_transaction
from src.faithfulness import check_faithfulness
from src.generation import generate_explanation
from src.model import evaluate_model
from src.rag.retrieval import index_case_library, retrieve_similar_cases_for_transaction

st.set_page_config(page_title="Explicabilidade de Fraude", layout="wide")

CSV_PATH = "data/paysim.csv"

# Numero de transacoes de exemplo (por grupo: verdadeiro/falso positivo)
# disponibilizadas no seletor da barra lateral. Nao precisa ser grande: e so
# uma vitrine de casos para a demonstracao, nao uma avaliacao estatistica
# (essa e o papel de src/evaluate.py).
N_SAMPLE_TRANSACTIONS_PER_GROUP = 25

# Numero maximo de casos indexados no RAG quando o "modo rapido" esta ativo,
# para acelerar o carregamento inicial do app numa demonstracao ao vivo. Sao
# sempre fraudes confirmadas reais do treino (nunca ficticias) - o modo
# rapido so reduz a QUANTIDADE de casos, nunca troca por dado inventado.
FAST_MODE_MAX_RAG_CASES = 1500


@st.cache_resource(show_spinner="Carregando dados, modelo e biblioteca de casos (RAG)...")
def load_pipeline(csv_path: str, max_rag_cases: int | None):
    """Carrega e prepara tudo que e caro de recalcular: dados, modelo,
    explainer SHAP e a biblioteca de casos do RAG indexada no ChromaDB.

    Cacheado com st.cache_resource (nao st.cache_data) porque o retorno
    contem objetos com estado interno (modelo XGBoost, TreeExplainer,
    collection do ChromaDB) que nao devem ser copiados/hasheados a cada
    interacao do usuario - so recalculados se os parametros mudarem.
    """
    raw_df = load_raw_data(csv_path)
    X_train, X_test, y_train, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    model = get_or_train_model(X_train, y_train)
    model_metrics = evaluate_model(model, X_test, y_test)
    explainer = build_explainer(model)
    collection = index_case_library(df_train, max_cases=max_rag_cases)

    tp_indices, fp_indices = select_flagged_transactions(
        model, X_test, y_test,
        n_true_positives=N_SAMPLE_TRANSACTIONS_PER_GROUP,
        n_false_positives=N_SAMPLE_TRANSACTIONS_PER_GROUP,
    )

    return {
        "X_test": X_test,
        "df_test": df_test,
        "model": model,
        "model_metrics": model_metrics,
        "explainer": explainer,
        "collection": collection,
        "tp_indices": tp_indices,
        "fp_indices": fp_indices,
    }


def render_sidebar() -> dict:
    """Renderiza os controles da barra lateral e retorna as escolhas do usuario."""
    st.sidebar.title("Configuracao")

    fast_mode = st.sidebar.checkbox(
        "Modo rapido (subconjunto do RAG)",
        value=True,
        help=(
            f"Indexa apenas {FAST_MODE_MAX_RAG_CASES} fraudes confirmadas do treino em vez "
            "de todas (~6.500), para o app carregar mais rapido. Os casos continuam sendo "
            "transacoes reais, nunca ficticias - so em menor quantidade."
        ),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Chave da API Claude")
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.sidebar.text_input(
        "ANTHROPIC_API_KEY",
        value=env_key,
        type="password",
        help="Necessaria apenas para gerar a explicacao em linguagem natural. "
        "As secoes de SHAP e RAG funcionam sem ela.",
    )

    return {"fast_mode": fast_mode, "api_key": api_key}


def render_transaction_selector(pipeline: dict) -> tuple[str, int]:
    """Deixa o usuario escolher o grupo (verdadeiro/falso positivo) e a
    transacao especifica dentro da amostra pre-selecionada."""
    st.sidebar.divider()
    st.sidebar.subheader("Transacao a explicar")

    group_label = st.sidebar.radio(
        "Grupo",
        options=["Verdadeiro positivo (fraude real)", "Falso positivo (modelo errou)"],
    )
    indices = pipeline["tp_indices"] if group_label.startswith("Verdadeiro") else pipeline["fp_indices"]

    df_test = pipeline["df_test"]
    options_labels = [
        f"#{i} - {df_test.loc[i, 'type']} - {df_test.loc[i, 'amount']:,.2f}" for i in indices
    ]
    selected_label = st.sidebar.selectbox("Transacao", options=options_labels)
    selected_index = indices[options_labels.index(selected_label)]

    return group_label, selected_index


def render_model_metrics(model_metrics: dict) -> None:
    st.subheader("Desempenho do classificador (conjunto de teste)")
    cols = st.columns(4)
    cols[0].metric("AUC-PR", f"{model_metrics['auc_pr']:.4f}")
    cols[1].metric("Precision", f"{model_metrics['precision']:.4f}")
    cols[2].metric("Recall", f"{model_metrics['recall']:.4f}")
    cols[3].metric("F1", f"{model_metrics['f1']:.4f}")
    st.caption(
        f"{model_metrics['n_test']:,} transacoes no teste, "
        f"{model_metrics['n_fraud_test']:,} fraudes reais "
        f"(limiar de decisao: {model_metrics['threshold']})."
    )


def render_transaction_details(full_row: pd.Series, predicted_probability: float, actual_label: int) -> None:
    st.subheader("Transacao selecionada")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Tipo", full_row["type"])
        st.metric("Valor", f"{full_row['amount']:,.2f}")
    with cols[1]:
        st.metric("Saldo origem: antes -> depois", f"{full_row['oldbalanceOrg']:,.2f} -> {full_row['newbalanceOrig']:,.2f}")
        st.metric("Saldo destino: antes -> depois", f"{full_row['oldbalanceDest']:,.2f} -> {full_row['newbalanceDest']:,.2f}")
    with cols[2]:
        st.metric("Probabilidade de fraude (modelo)", f"{predicted_probability:.2%}")
        st.metric("Rotulo real", "FRAUDE" if actual_label == 1 else "Legitima")


def render_shap_features(top_features) -> None:
    st.subheader("Por que o modelo suspeitou? (valores SHAP reais)")
    chart_data = pd.DataFrame(
        {"contribuicao_shap": [f.shap_value for f in top_features]},
        index=[f.feature_name for f in top_features],
    )
    st.bar_chart(chart_data)
    st.caption("Valores positivos empurram a predicao para FRAUDE; negativos, para LEGITIMA.")

    for feature in top_features:
        direction = "empurra para FRAUDE" if feature.shap_value > 0 else "empurra para LEGITIMA"
        st.text(f"{feature.feature_name}: valor={feature.feature_value:,.2f}  shap={feature.shap_value:+.4f}  ({direction})")


def render_similar_cases(similar_cases) -> None:
    st.subheader("Casos de fraude confirmada similares (RAG)")
    if not similar_cases:
        st.info("Nenhum caso similar encontrado na biblioteca.")
        return
    for case in similar_cases:
        with st.expander(f"Similaridade {case.similarity:.4f} - {case.case_id}"):
            st.write(case.text)


def render_faithfulness(result) -> None:
    st.subheader("Verificacao de fidelidade")

    if result.is_faithful:
        st.success(f"Explicacao considerada FIEL (F1 = {result.f1:.2f})")
    else:
        st.warning(f"Explicacao considerada POUCO FIEL (F1 = {result.f1:.2f})")

    cols = st.columns(4)
    cols[0].metric("Precision", f"{result.precision:.2f}")
    cols[1].metric("Recall", f"{result.recall:.2f}")
    cols[2].metric("F1", f"{result.f1:.2f}")
    cols[3].metric("Recall ponderado (SHAP)", f"{result.weighted_recall:.2f}")

    if result.correctly_cited:
        st.markdown(f"**Citadas corretamente:** {', '.join(result.correctly_cited)}")
    if result.omitted_features:
        st.markdown(f"**Features importantes omitidas:** {', '.join(result.omitted_features)}")
    if result.irrelevant_features:
        st.markdown(f"**Citadas mas irrelevantes nesse caso:** {', '.join(result.irrelevant_features)}")
    if result.hallucinated_features:
        st.markdown(f"**⚠️ Alucinadas (nao existem no modelo):** {', '.join(result.hallucinated_features)}")


def main() -> None:
    st.title("Assistente de Explicabilidade de Fraude")
    st.caption("XGBoost + SHAP + RAG + LLM, com verificacao de fidelidade da explicacao gerada.")

    controls = render_sidebar()
    max_rag_cases = FAST_MODE_MAX_RAG_CASES if controls["fast_mode"] else None

    pipeline = load_pipeline(CSV_PATH, max_rag_cases)
    render_model_metrics(pipeline["model_metrics"])
    st.divider()

    group_label, selected_index = render_transaction_selector(pipeline)

    X_test = pipeline["X_test"]
    df_test = pipeline["df_test"]
    model = pipeline["model"]

    x_row = X_test.loc[[selected_index]]
    full_row = df_test.loc[selected_index]
    predicted_probability = float(model.predict_proba(x_row)[0, 1])
    actual_label = int(full_row["isFraud"])

    render_transaction_details(full_row, predicted_probability, actual_label)

    explanation_data = explain_transaction(pipeline["explainer"], x_row)
    top_features = explanation_data["top_features"]
    render_shap_features(top_features)

    similar_cases = retrieve_similar_cases_for_transaction(full_row, pipeline["collection"])
    render_similar_cases(similar_cases)

    st.divider()
    st.subheader("Explicacao em linguagem natural (LLM)")

    if "explanations" not in st.session_state:
        st.session_state["explanations"] = {}

    generate_clicked = st.button(
        "Gerar explicacao",
        disabled=not controls["api_key"],
        help=None if controls["api_key"] else "Informe a ANTHROPIC_API_KEY na barra lateral para habilitar.",
    )

    if generate_clicked:
        client = anthropic.Anthropic(api_key=controls["api_key"])
        with st.spinner("Chamando a API Claude..."):
            try:
                generated = generate_explanation(
                    row=full_row,
                    predicted_probability=predicted_probability,
                    top_features=top_features,
                    similar_cases=similar_cases,
                    client=client,
                )
                faithfulness_result = check_faithfulness(generated, top_features)
                st.session_state["explanations"][selected_index] = (generated, faithfulness_result)
            except Exception as error:  # noqa: BLE001
                st.error(f"Falha ao gerar explicacao: {error}")

    if selected_index in st.session_state["explanations"]:
        generated, faithfulness_result = st.session_state["explanations"][selected_index]

        st.markdown(f"**Narrativa:** {generated.narrativa}")
        st.markdown(f"**Acao recomendada:** {generated.acao_recomendada}")
        st.markdown(f"**Features citadas:** {', '.join(generated.features_citadas) or '(nenhuma)'}")

        render_faithfulness(faithfulness_result)
    else:
        st.info("Clique em 'Gerar explicacao' para chamar o LLM e ver a explicacao e a verificacao de fidelidade.")


if __name__ == "__main__":
    main()
