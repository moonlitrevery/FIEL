"""
Verificador de fidelidade: a explicacao gerada pelo LLM realmente reflete o
que o modelo (XGBoost) usou para decidir?

Este e o modulo mais importante do projeto. A ideia central e simples e
deliberadamente NAO depende de NLP sofisticado: como generation.py forca a
saida do LLM a ser um JSON com uma lista explicita de "features_citadas"
(nomes de colunas, nao frases livres), verificar fidelidade vira um problema
de COMPARACAO DE CONJUNTOS entre:
  (a) as features que o LLM disse que usou, e
  (b) as features que o SHAP (fato matematico sobre o modelo) mostra que
      realmente mais pesaram na decisao.

Isso transforma "a IA generativa esta blefando sobre por que o modelo decidiu
isso?" - uma pergunta qualitativa dificil - numa metrica quantitativa
(precision/recall/F1 de citacao), reprodutivel e defensavel na apresentacao.

Distinguimos dois tipos de erro de citacao, que tem gravidade bem diferente
para a analise critica do relatorio:
- "alucinacao": o LLM cita um nome de feature que NEM EXISTE no modelo (ele
  inventou um nome). E o erro mais grave: mostra que o LLM nao esta de fato
  ancorado nos dados fornecidos no prompt.
- "irrelevancia": o LLM cita uma feature que existe no modelo, mas que nao
  estava entre as mais importantes (segundo o SHAP) para ESSA transacao
  especifica. E um erro mais brando: a feature e real, so nao foi a razao
  principal da decisao nesse caso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data_prep import get_feature_columns
from src.explain import FeatureContribution

# Limiar de F1 acima do qual consideramos uma explicacao individual "fiel".
# E um numero didatico (nao ha um padrao universal na literatura para esse
# limiar) - o que importa mais para o relatorio e a distribuicao continua de
# F1 no conjunto de teste, nao so a contagem binaria acima/abaixo do corte.
FAITHFULNESS_THRESHOLD = 0.5


@dataclass
class FaithfulnessResult:
    """Resultado da comparacao entre features citadas pelo LLM e o SHAP real,
    para uma unica explicacao gerada."""

    cited_features: list[str]
    real_top_features: list[str]
    correctly_cited: list[str]
    hallucinated_features: list[str]
    irrelevant_features: list[str]
    omitted_features: list[str]
    precision: float
    recall: float
    f1: float
    weighted_recall: float
    is_faithful: bool

    def to_dict(self) -> dict:
        return {
            "cited_features": self.cited_features,
            "real_top_features": self.real_top_features,
            "correctly_cited": self.correctly_cited,
            "hallucinated_features": self.hallucinated_features,
            "irrelevant_features": self.irrelevant_features,
            "omitted_features": self.omitted_features,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "weighted_recall": self.weighted_recall,
            "is_faithful": self.is_faithful,
        }


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """Remove duplicatas mantendo a ordem original (dict preserva ordem de
    insercao em Python 3.7+); usado para nao inflar metricas se o LLM citar
    a mesma feature duas vezes."""
    return list(dict.fromkeys(items))


def compute_faithfulness(
    cited_features: list[str],
    real_top_features: list[FeatureContribution],
    valid_feature_names: list[str] | None = None,
    faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
) -> FaithfulnessResult:
    """Compara as features citadas pelo LLM com as top features reais do SHAP.

    Parametros
    ----------
    cited_features: lista de nomes de features vinda de
        GeneratedExplanation.features_citadas (generation.py).
    real_top_features: lista de FeatureContribution vinda de
        explain.explain_transaction()["top_features"] - o "gabarito".
    valid_feature_names: conjunto de nomes de features que existem de fato no
        modelo (por padrao, get_feature_columns()) - usado para distinguir
        alucinacao (nome inexistente) de irrelevancia (nome real, mas nao
        top-k nessa transacao).
    faithfulness_threshold: ponto de corte de F1 para o rotulo binario
        is_faithful.

    Retorna
    -------
    FaithfulnessResult com as metricas de precision, recall, F1 e recall
    ponderado pela magnitude do SHAP (weighted_recall), alem das listas
    detalhadas de acertos e cada tipo de erro.
    """
    valid_set = set(valid_feature_names or get_feature_columns())

    cited = _dedupe_preserving_order(cited_features)
    cited_set = set(cited)

    real_names = [feature.feature_name for feature in real_top_features]
    real_set = set(real_names)

    correctly_cited = [name for name in cited if name in real_set]
    hallucinated = [name for name in cited if name not in valid_set]
    irrelevant = [name for name in cited if name in valid_set and name not in real_set]
    omitted = [name for name in real_names if name not in cited_set]

    correctly_cited_set = set(correctly_cited)

    # Citar zero features nunca deveria contar como "explicacao precisa" -
    # tratamos precision e recall como 0.0 nesse caso (em vez de indefinido),
    # porque uma explicacao vazia nao e util nem fiel.
    precision = len(correctly_cited_set) / len(cited_set) if cited_set else 0.0
    recall = len(correctly_cited_set) / len(real_set) if real_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    weighted_recall = _compute_weighted_recall(correctly_cited_set, real_top_features)

    return FaithfulnessResult(
        cited_features=cited,
        real_top_features=real_names,
        correctly_cited=correctly_cited,
        hallucinated_features=hallucinated,
        irrelevant_features=irrelevant,
        omitted_features=omitted,
        precision=precision,
        recall=recall,
        f1=f1,
        weighted_recall=weighted_recall,
        is_faithful=f1 >= faithfulness_threshold,
    )


def _compute_weighted_recall(
    correctly_cited_set: set[str], real_top_features: list[FeatureContribution]
) -> float:
    """Recall ponderado pela magnitude da contribuicao SHAP de cada feature.

    O recall comum trata todas as top-k features como igualmente importantes,
    mas na pratica a 1a feature do ranking SHAP costuma pesar muito mais que
    a 5a. Este recall ponderado da mais credito por citar a feature dominante
    do que por citar uma feature secundaria - uma leitura mais rica para a
    analise critica do relatorio do que o recall simples.

    Calculado como: soma(|shap| das features corretamente citadas) dividido
    por soma(|shap| de todas as top-k reais). Resulta em 0.0 se nenhuma
    feature real tiver contribuicao (caso degenerado).
    """
    total_importance = sum(abs(feature.shap_value) for feature in real_top_features)
    if total_importance == 0:
        return 0.0

    captured_importance = sum(
        abs(feature.shap_value) for feature in real_top_features if feature.feature_name in correctly_cited_set
    )
    return captured_importance / total_importance


def check_faithfulness(generated_explanation, real_top_features: list[FeatureContribution]) -> FaithfulnessResult:
    """Atalho ergonomico: recebe diretamente um GeneratedExplanation
    (generation.py) em vez da lista crua de strings citadas."""
    return compute_faithfulness(generated_explanation.features_citadas, real_top_features)


@dataclass
class AggregateFaithfulnessReport:
    """Resumo agregado de fidelidade sobre varias explicacoes (ex: todo o
    conjunto de teste), usado por evaluate.py para o relatorio final."""

    n_explanations: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_weighted_recall: float
    faithful_rate: float
    hallucination_rate: float
    individual_results: list[FaithfulnessResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_explanations": self.n_explanations,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_weighted_recall": self.mean_weighted_recall,
            "faithful_rate": self.faithful_rate,
            "hallucination_rate": self.hallucination_rate,
        }


def aggregate_faithfulness_results(results: list[FaithfulnessResult]) -> AggregateFaithfulnessReport:
    """Agrega uma lista de FaithfulnessResult (uma por transacao avaliada) em
    um relatorio unico com medias e taxas.

    hallucination_rate: fracao das explicacoes que citaram PELO MENOS UMA
    feature inexistente no modelo - a metrica mais critica para discutir os
    limites do sistema no relatorio (com que frequencia o LLM "inventa" um
    nome de feature, apesar do prompt fornecer a lista exata de nomes validos).
    """
    n = len(results)
    if n == 0:
        return AggregateFaithfulnessReport(
            n_explanations=0,
            mean_precision=0.0,
            mean_recall=0.0,
            mean_f1=0.0,
            mean_weighted_recall=0.0,
            faithful_rate=0.0,
            hallucination_rate=0.0,
            individual_results=[],
        )

    return AggregateFaithfulnessReport(
        n_explanations=n,
        mean_precision=sum(r.precision for r in results) / n,
        mean_recall=sum(r.recall for r in results) / n,
        mean_f1=sum(r.f1 for r in results) / n,
        mean_weighted_recall=sum(r.weighted_recall for r in results) / n,
        faithful_rate=sum(1 for r in results if r.is_faithful) / n,
        hallucination_rate=sum(1 for r in results if r.hallucinated_features) / n,
        individual_results=results,
    )


if __name__ == "__main__":
    # Execucao manual com um exemplo ilustrativo (sem depender da API Claude):
    # `uv run python -m src.faithfulness`
    real_top = [
        FeatureContribution(feature_name="error_balance_orig", feature_value=0.0, shap_value=5.6),
        FeatureContribution(feature_name="newbalanceOrig", feature_value=0.0, shap_value=0.9),
        FeatureContribution(feature_name="oldbalanceOrg", feature_value=2674794.22, shap_value=0.75),
        FeatureContribution(feature_name="amount", feature_value=2674794.22, shap_value=0.72),
        FeatureContribution(feature_name="oldbalanceDest", feature_value=1380829.87, shap_value=-0.36),
    ]

    print("Cenario 1: explicacao fiel (cita as 2 features mais importantes)")
    result = compute_faithfulness(["error_balance_orig", "newbalanceOrig"], real_top)
    print(result.to_dict())

    print("\nCenario 2: explicacao com alucinacao (cita uma feature inexistente)")
    result = compute_faithfulness(["error_balance_orig", "risco_regional"], real_top)
    print(result.to_dict())

    print("\nCenario 3: explicacao irrelevante (cita features reais, mas nao as top-k)")
    result = compute_faithfulness(["hour_of_day", "step"], real_top)
    print(result.to_dict())
