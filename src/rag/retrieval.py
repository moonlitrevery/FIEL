"""
Embeddings e busca por similaridade na biblioteca de casos (RAG).

Este modulo cuida da parte "R" (retrieval) do RAG: transforma os textos
estruturados de rag/case_library.py em vetores (embeddings) usando
sentence-transformers, indexa esses vetores no ChromaDB (banco vetorial
local, persistido em disco) e, dado o texto de uma transacao suspeita nova,
busca os casos de fraude confirmada mais semanticamente parecidos.

Calculamos os embeddings explicitamente com sentence-transformers (em vez de
deixar o ChromaDB usar sua funcao de embedding padrao) para termos controle
e clareza total sobre qual modelo esta sendo usado - importante para poder
explicar essa escolha na apresentacao.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.case_library import FraudCase, build_case_library, serialize_transaction_to_text

# Modelo multilingue: os textos dos casos (case_library.py) sao gerados em
# portugues, entao precisamos de um modelo de embeddings treinado para
# capturar semantica em portugues - o "all-MiniLM-L6-v2" (default de muitos
# tutoriais) e treinado majoritariamente em ingles e teria qualidade pior
# aqui. O "paraphrase-multilingual-MiniLM-L12-v2" e compacto (~118MB, 384
# dimensoes) e cobre portugues com boa qualidade, o que mantem a busca
# rapida o suficiente para rodar em CPU numa demonstracao ao vivo.
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Diretorio onde o ChromaDB persiste o indice vetorial em disco (nao
# versionado no git - e reconstruido a partir do CSV do PaySim sempre que
# necessario, via index_case_library()).
CHROMA_PERSIST_DIR = "chroma_db"

COLLECTION_NAME = "fraud_cases"

# Numero padrao de casos similares retornados por consulta. Poucos (3) o
# suficiente para caber no prompt da LLM sem estourar contexto, mas dando
# mais de um precedente para a explicacao nao depender de um unico caso.
DEFAULT_TOP_K = 3

# Tamanho de lote para insercao no ChromaDB. O ChromaDB tem um limite maximo
# de itens por chamada de add(); inserir em lotes menores evita esse limite
# mesmo com uma biblioteca de milhares de casos.
INSERT_BATCH_SIZE = 500


@dataclass
class RetrievedCase:
    """Um caso recuperado da biblioteca, junto da sua similaridade com a
    transacao consultada."""

    case_id: str
    text: str
    metadata: dict
    similarity: float


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Carrega (e cacheia em memoria) o modelo de embeddings.

    O cache evita recarregar o modelo (custoso: le pesos do disco/HF hub) a
    cada chamada - importante porque generation.py e evaluate.py podem
    chamar a busca varias vezes na mesma execucao.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_client(persist_directory: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    """Cria um cliente ChromaDB persistido em disco no diretorio informado."""
    return chromadb.PersistentClient(path=persist_directory)


def reset_collection(client: chromadb.ClientAPI, collection_name: str = COLLECTION_NAME):
    """Apaga (se existir) e recria a collection do zero.

    Usado sempre que reindexamos a biblioteca de casos inteira, para nunca
    misturar casos de execucoes antigas (ex: geradas com max_cases menor)
    com a execucao atual. A metrica de distancia e explicitamente configurada
    como cosseno, o padrao recomendado para embeddings de sentence-transformers.
    """
    existing_names = [c.name for c in client.list_collections()]
    if collection_name in existing_names:
        client.delete_collection(collection_name)
    return client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def index_case_library(
    df_train,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    max_cases: int | None = None,
):
    """Monta a biblioteca de casos a partir do treino e a indexa no ChromaDB.

    Fluxo: df_train -> build_case_library() [texto+metadata] -> embeddings
    (sentence-transformers) -> collection.add() [ChromaDB], em lotes de
    INSERT_BATCH_SIZE para respeitar o limite de itens por insercao.

    Retorna a collection do ChromaDB, pronta para ser consultada por
    retrieve_similar_cases().
    """
    cases = build_case_library(df_train, max_cases=max_cases)
    return index_cases(cases, persist_directory=persist_directory, collection_name=collection_name)


def index_cases(
    cases: list[FraudCase],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
):
    """Indexa uma lista de FraudCase ja construida (ver index_case_library)."""
    model = get_embedding_model()
    client = get_chroma_client(persist_directory)
    collection = reset_collection(client, collection_name)

    texts = [case.text for case in cases]
    ids = [case.case_id for case in cases]
    metadatas = [case.metadata for case in cases]
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64).tolist()

    for start in range(0, len(cases), INSERT_BATCH_SIZE):
        end = start + INSERT_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    return collection


def retrieve_similar_cases(query_text: str, collection, k: int = DEFAULT_TOP_K) -> list[RetrievedCase]:
    """Busca os k casos de fraude confirmada mais similares a um texto de consulta.

    A consulta e embedada com o MESMO modelo usado para indexar a biblioteca
    (get_embedding_model() e cacheado, entao e literalmente a mesma instancia).
    A similaridade retornada e (1 - distancia de cosseno): 1.0 significa
    textos identicos, valores mais baixos indicam menor semelhanca.
    """
    model = get_embedding_model()
    query_embedding = model.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for case_id, document, metadata, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append(
            RetrievedCase(
                case_id=case_id,
                text=document,
                metadata=metadata,
                similarity=1.0 - distance,
            )
        )
    return retrieved


def retrieve_similar_cases_for_transaction(row, collection, k: int = DEFAULT_TOP_K) -> list[RetrievedCase]:
    """Atalho: serializa a transacao (mesmo formato da biblioteca de casos) e
    busca os k casos de fraude confirmada mais parecidos com ela."""
    query_text = serialize_transaction_to_text(row)
    return retrieve_similar_cases(query_text, collection, k=k)


if __name__ == "__main__":
    # Execucao manual: `uv run python -m src.rag.retrieval`
    # Usa max_cases para deixar a indexacao rapida numa checagem manual;
    # o pipeline final (evaluate.py) indexa a biblioteca completa.
    from src.data_prep import load_raw_data, prepare_train_test_split

    raw_df = load_raw_data("data/paysim.csv")
    _, X_test, _, y_test, df_train, df_test = prepare_train_test_split(raw_df)

    print("Indexando biblioteca de casos no ChromaDB (isso baixa o modelo de "
          "embeddings na primeira execucao)...")
    collection = index_case_library(df_train, max_cases=500)
    print(f"Collection '{COLLECTION_NAME}' criada com {collection.count()} casos.")

    fraud_index = y_test[y_test == 1].index[0]
    query_row = df_test.loc[fraud_index]
    print(f"\nTransacao consultada (indice {fraud_index}):")
    print(f"  {serialize_transaction_to_text(query_row)}")

    similar_cases = retrieve_similar_cases_for_transaction(query_row, collection, k=3)
    print(f"\nTop {len(similar_cases)} casos similares na biblioteca de fraudes confirmadas:")
    for case in similar_cases:
        print(f"\n  case_id: {case.case_id} (similaridade: {case.similarity:.4f})")
        print(f"  {case.text}")
