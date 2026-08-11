# fraud-explainability

Assistente de explicabilidade de fraude (TCC / disciplina de IA Generativa).

Dado uma transação flagrada como suspeita por um classificador XGBoost,
o sistema gera uma explicação em linguagem natural (via LLM) fundamentada em:

1. Valores SHAP reais do modelo (por que o modelo classificou como fraude).
2. Casos de fraude confirmada similares, recuperados via RAG a partir do
   próprio conjunto de treino (PaySim).

Em seguida, um verificador de fidelidade mede se a explicação gerada pela
LLM é consistente com o que o modelo realmente usou para decidir (comparando
as features citadas na explicação com as top features do SHAP).

## Stack

- Python 3.11+, gerenciado via uv (https://docs.astral.sh/uv/)
- xgboost, shap, scikit-learn, pandas
- sentence-transformers + chromadb (RAG)
- anthropic (geração da explicação via Claude)
- streamlit (dashboard de demonstração)
- pytest (testes)

## Dataset

PaySim - Synthetic Financial Datasets For Fraud Detection (Kaggle).
Baixe o CSV e coloque em data/paysim.csv (não versionado no git).

## Estrutura

    src/
    |-- data_prep.py       # carrega e filtra o PaySim, feature engineering
    |-- model.py            # treino do XGBoost + avaliacao AUC-PR
    |-- explain.py          # SHAP: calculo e extracao de top-k features
    |-- rag/
    |   |-- case_library.py # serializa fraudes confirmadas do treino em docs
    |   \-- retrieval.py    # embeddings + chromadb + busca por similaridade
    |-- generation.py        # prompt + chamada a API Claude, saida em JSON
    |-- faithfulness.py      # comparacao features citadas vs. SHAP real
    \-- evaluate.py           # roda o pipeline completo num conjunto de teste
    app.py                    # dashboard Streamlit

## Modelo

O classificador (src/model.py) e um XGBoost treinado com scale_pos_weight
para compensar o desbalanceamento de classes (fraude e menos de 1% dos
casos). A metrica de avaliacao e AUC-PR (area sob a curva Precision-Recall),
nao acuracia, por ser a metrica correta para datasets desbalanceados.

## Uso

    uv sync
    uv run python -m src.data_prep
    uv run python -m src.model
    uv run python -m pytest
