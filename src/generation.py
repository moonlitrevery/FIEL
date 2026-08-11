"""
Geracao da explicacao em linguagem natural via LLM (Claude), grounded em SHAP + RAG.

Este modulo monta o prompt que da ao LLM (Claude) exatamente os fatos que ele
pode usar - os valores SHAP reais da transacao (explain.py) e os casos de
fraude confirmada mais similares (rag/retrieval.py) - e nunca deixa o modelo
"inventar" contexto. A saida e SEMPRE um JSON estruturado com 3 campos fixos
(narrativa, features_citadas, acao_recomendada), nunca texto livre: e isso
que torna o verificador de fidelidade (faithfulness.py) viavel sem precisar
de NLP sofisticado para extrair afirmacoes de um texto solto - basta comparar
a lista de strings em "features_citadas" com os nomes das top features do SHAP.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import anthropic
import pandas as pd

from src.data_prep import get_feature_columns
from src.explain import FeatureContribution
from src.rag.retrieval import RetrievedCase

# Claude Sonnet 5: bom equilibrio entre qualidade de raciocinio e custo/latencia
# para gerar explicacoes de texto curto. Pode ser trocado por um modelo mais
# barato (ex: um Haiku) durante testes/desenvolvimento sem mudar nenhuma outra
# parte do pipeline, ja que o nome do modelo esta centralizado aqui.
MODEL_NAME = "claude-sonnet-5"

MAX_TOKENS = 1024

# Temperatura 0: queremos respostas o mais deterministicas possivel. Isso e
# especialmente importante aqui porque o verificador de fidelidade
# (faithfulness.py) mede a taxa de acerto do LLM ao citar features reais do
# SHAP - com temperatura alta, essa taxa variaria a cada execucao so por
# aleatoriedade da amostragem de tokens, nao por uma mudanca real na
# qualidade da explicacao.
TEMPERATURE = 0

SYSTEM_PROMPT = """Voce e um assistente especializado em explicar, para analistas humanos de uma instituicao financeira, por que um modelo de deteccao de fraude (XGBoost) sinalizou uma transacao como suspeita.

Voce recebe, para uma unica transacao:
1. Os dados brutos da transacao.
2. A probabilidade de fraude estimada pelo modelo.
3. As features que, segundo valores SHAP REAIS do modelo (nao uma estimativa sua), mais influenciaram essa classificacao, com o valor de cada feature e a direcao do efeito.
4. Casos de fraude CONFIRMADA, recuperados por similaridade semantica do historico de transacoes fraudulentas reais.

Regras obrigatorias, sem excecao:
- Baseie sua explicacao APENAS nos dados SHAP e nos casos similares fornecidos abaixo. Nunca invente valores, nomes de features, numeros ou casos que nao estejam explicitamente nos dados fornecidos.
- No campo "features_citadas" da sua resposta, inclua APENAS identificadores escolhidos dentre esta lista fixa de nomes de features do modelo: {feature_columns}. Cite ali as features que voce efetivamente usou como justificativa na narrativa.
- Sua resposta deve ser SOMENTE um objeto JSON valido, sem nenhum texto, comentario ou markdown antes ou depois. Use exatamente este formato:
{{
  "narrativa": "explicacao em portugues, clara e objetiva, para um analista nao tecnico, de por que a transacao foi sinalizada",
  "features_citadas": ["nome_da_feature_1", "nome_da_feature_2"],
  "acao_recomendada": "uma recomendacao curta e concreta (ex: bloquear e contatar cliente, revisao manual, liberar com monitoramento)"
}}"""


@dataclass
class GeneratedExplanation:
    """Explicacao estruturada gerada pelo LLM para uma transacao."""

    narrativa: str
    features_citadas: list[str]
    acao_recomendada: str
    raw_response: str

    def to_dict(self) -> dict:
        return {
            "narrativa": self.narrativa,
            "features_citadas": self.features_citadas,
            "acao_recomendada": self.acao_recomendada,
        }


def get_anthropic_client() -> anthropic.Anthropic:
    """Cria o cliente da API Claude a partir da variavel de ambiente ANTHROPIC_API_KEY.

    Falha de forma explicita (em vez de deixar o erro cru da biblioteca
    anthropic aparecer) para deixar claro qual variavel de ambiente precisa
    ser configurada.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variavel de ambiente ANTHROPIC_API_KEY nao configurada. "
            "Defina-a antes de gerar explicacoes, ex. no PowerShell: "
            "$env:ANTHROPIC_API_KEY = 'sua-chave-aqui'"
        )
    return anthropic.Anthropic(api_key=api_key)


def format_transaction_summary(row: pd.Series) -> str:
    """Formata os dados brutos da transacao em texto legivel para o prompt."""
    return (
        f"- Tipo: {row['type']}\n"
        f"- Valor: {row['amount']:,.2f}\n"
        f"- Saldo da conta de origem antes/depois: {row['oldbalanceOrg']:,.2f} / {row['newbalanceOrig']:,.2f}\n"
        f"- Saldo da conta de destino antes/depois: {row['oldbalanceDest']:,.2f} / {row['newbalanceDest']:,.2f}\n"
        f"- Momento da simulacao: step {int(row['step'])} (hora {int(row['hour_of_day'])} de um ciclo de 24h)"
    )


def format_shap_features(top_features: list[FeatureContribution]) -> str:
    """Formata as top features do SHAP em uma lista numerada para o prompt."""
    lines = []
    for i, feature in enumerate(top_features, start=1):
        direction = "empurra para FRAUDE" if feature.shap_value > 0 else "empurra para LEGITIMA"
        lines.append(
            f"{i}. {feature.feature_name} = {feature.feature_value:,.2f} "
            f"(contribuicao SHAP: {feature.shap_value:+.4f}, {direction})"
        )
    return "\n".join(lines)


def format_similar_cases(similar_cases: list[RetrievedCase]) -> str:
    """Formata os casos similares recuperados pelo RAG em uma lista numerada."""
    if not similar_cases:
        return "(nenhum caso similar encontrado na biblioteca)"
    lines = []
    for i, case in enumerate(similar_cases, start=1):
        lines.append(f"Caso {i} (similaridade {case.similarity:.4f}): {case.text}")
    return "\n".join(lines)


def build_user_prompt(
    row: pd.Series,
    predicted_probability: float,
    top_features: list[FeatureContribution],
    similar_cases: list[RetrievedCase],
) -> str:
    """Monta o prompt de usuario com os 3 blocos de contexto factual: dados da
    transacao, features SHAP e casos similares confirmados."""
    return (
        f"DADOS DA TRANSACAO SUSPEITA:\n{format_transaction_summary(row)}\n\n"
        f"PROBABILIDADE DE FRAUDE ESTIMADA PELO MODELO: {predicted_probability:.2%}\n\n"
        f"FEATURES MAIS RELEVANTES PARA ESSA DECISAO (segundo valores SHAP reais do modelo):\n"
        f"{format_shap_features(top_features)}\n\n"
        f"CASOS DE FRAUDE CONFIRMADA SIMILARES (recuperados do historico real de fraudes):\n"
        f"{format_similar_cases(similar_cases)}\n\n"
        "Gere a explicacao seguindo exatamente o formato JSON e as regras definidas."
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove blocos de markdown (```json ... ```) que o modelo eventualmente
    inclua ao redor do JSON, apesar da instrucao para nao faze-lo."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def parse_llm_response(raw_text: str) -> GeneratedExplanation:
    """Converte o texto bruto retornado pela API em um GeneratedExplanation.

    Valida que o JSON tem exatamente as 3 chaves esperadas, com os tipos
    corretos. Levanta ValueError com uma mensagem clara (incluindo o texto
    bruto) se o LLM nao seguir o formato - isso nao deveria acontecer com o
    prefill do turno do assistente (ver generate_explanation), mas e
    importante falhar de forma legivel em vez de um KeyError/JSONDecodeError cru.
    """
    cleaned_text = _strip_markdown_fences(raw_text.strip())

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Resposta do LLM nao e um JSON valido: {error}\nTexto bruto recebido:\n{raw_text}"
        ) from error

    required_keys = {"narrativa", "features_citadas", "acao_recomendada"}
    missing_keys = required_keys - payload.keys()
    if missing_keys:
        raise ValueError(f"JSON do LLM esta sem as chaves obrigatorias: {missing_keys}. Payload: {payload}")

    if not isinstance(payload["narrativa"], str):
        raise ValueError(f"'narrativa' deveria ser string, veio: {type(payload['narrativa'])}")
    if not isinstance(payload["features_citadas"], list) or not all(
        isinstance(f, str) for f in payload["features_citadas"]
    ):
        raise ValueError(f"'features_citadas' deveria ser lista de strings, veio: {payload['features_citadas']}")
    if not isinstance(payload["acao_recomendada"], str):
        raise ValueError(f"'acao_recomendada' deveria ser string, veio: {type(payload['acao_recomendada'])}")

    return GeneratedExplanation(
        narrativa=payload["narrativa"],
        features_citadas=payload["features_citadas"],
        acao_recomendada=payload["acao_recomendada"],
        raw_response=raw_text,
    )


def generate_explanation(
    row: pd.Series,
    predicted_probability: float,
    top_features: list[FeatureContribution],
    similar_cases: list[RetrievedCase],
    client: anthropic.Anthropic | None = None,
) -> GeneratedExplanation:
    """Gera a explicacao em linguagem natural para uma transacao suspeita.

    Usa a tecnica de "prefill" do turno do assistente: comecamos a resposta
    do modelo com o caractere "{" (via uma mensagem com role="assistant"),
    o que praticamente forca a API a continuar direto no JSON, sem preambulo
    ou explicacao textual antes dele. O texto retornado pela API e a
    CONTINUACAO apos o "{" que fornecemos, entao ele e reanexado antes do
    parsing em parse_llm_response.
    """
    client = client or get_anthropic_client()

    system_prompt = SYSTEM_PROMPT.format(feature_columns=get_feature_columns())
    user_prompt = build_user_prompt(row, predicted_probability, top_features, similar_cases)

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "{"},
        ],
    )

    completion = response.content[0].text
    full_json_text = "{" + completion

    return parse_llm_response(full_json_text)


if __name__ == "__main__":
    # Execucao manual: `uv run python -m src.generation`
    # Requer ANTHROPIC_API_KEY configurada no ambiente.
    from src.data_prep import load_raw_data, prepare_train_test_split
    from src.explain import build_explainer, explain_transaction
    from src.model import load_model
    from src.rag.retrieval import index_case_library, retrieve_similar_cases_for_transaction

    raw_df = load_raw_data("data/paysim.csv")
    _, X_test, _, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    model = load_model()
    explainer = build_explainer(model)

    fraud_index = y_test[y_test == 1].index[0]
    x_row = X_test.loc[[fraud_index]]
    full_row = df_test.loc[fraud_index]

    explanation_data = explain_transaction(explainer, x_row)
    predicted_probability = float(model.predict_proba(x_row)[0, 1])

    print("Indexando biblioteca de casos (amostra rapida para teste manual)...")
    collection = index_case_library(df_train, max_cases=500)
    similar_cases = retrieve_similar_cases_for_transaction(full_row, collection, k=3)

    print("Chamando a API Claude para gerar a explicacao...")
    result = generate_explanation(
        row=full_row,
        predicted_probability=predicted_probability,
        top_features=explanation_data["top_features"],
        similar_cases=similar_cases,
    )

    print("\nExplicacao gerada:")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
