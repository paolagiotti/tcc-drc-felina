"""
Bloco 2: pipeline de modelagem, sem vazamento.

Regra unica que organiza este modulo: toda operacao que APRENDE alguma coisa
dos dados (imputar, padronizar, codificar, balancear) vive dentro do Pipeline,
e portanto e reajustada do zero em cada particao de treino da validacao
cruzada. Nada aqui e ajustado sobre o conjunto completo.

Este modulo nao executa nada sozinho: e importado por `experimento_iris.py` e
pelos demais scripts que ajustam o modelo de estadiamento.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import ExtraTreeRegressor

import config as cfg

SEED = cfg.SEMENTE


# ---------------------------------------------------------------------------
# Carga da coorte produzida pela etapa 1
# ---------------------------------------------------------------------------

def carregar_coorte() -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """Le a saida da triagem. Falha alto se a etapa 1 nao foi executada."""
    parquet = cfg.SAIDA_DADOS / "coorte.parquet"
    conjuntos_json = cfg.SAIDA_DADOS / "conjuntos_variaveis.json"
    if not parquet.exists() or not conjuntos_json.exists():
        raise SystemExit(
            f"Saida da etapa 1 nao encontrada em {cfg.SAIDA_DADOS}.\n"
            f"Rode antes: python triagem.py"
        )
    df = pd.read_parquet(parquet)
    conjuntos = json.loads(conjuntos_json.read_text(encoding="utf-8"))
    y = df[cfg.COL_IRIS].astype(int)
    return df, y, conjuntos


def separar_tipos(df: pd.DataFrame, colunas: list[str]) -> tuple[list[str], list[str]]:
    """Divide as colunas em numericas e categoricas."""
    colunas = [c for c in colunas if c in df.columns]
    num = [c for c in colunas if pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in colunas if c not in num]
    return num, cat


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def construir_preprocessador(num: list[str], cat: list[str],
                             tratamento: str | None = None) -> ColumnTransformer:
    """Imputacao, padronizacao e codificacao -- tudo ajustado dentro do fold.

    `tratamento` controla o que acontece com os valores ausentes; ver
    cfg.TRATAMENTO_AUSENTES.
    """
    tratamento = tratamento or getattr(cfg, "TRATAMENTO_AUSENTES", "imputar")

    if tratamento == "nativo":
        # Sem imputacao: o ausente chega ao modelo. Padronizar tambem sai, por
        # nao fazer sentido com ausentes e por ser desnecessario em arvores.
        etapa_num = "passthrough"
        etapa_cat = ImbPipeline([
            ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                        min_frequency=0.05, sparse_output=False)),
        ])
    else:
        indicador = tratamento == "indicador"
        etapa_num = ImbPipeline([
            ("imputar", IterativeImputer(
                estimator=ExtraTreeRegressor(random_state=SEED),
                max_iter=10, random_state=SEED, initial_strategy="median",
                add_indicator=indicador)),
            ("escalar", StandardScaler()),
        ])
        etapa_cat = ImbPipeline([
            ("imputar", SimpleImputer(strategy="most_frequent",
                                      add_indicator=indicador)),
            ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                        min_frequency=0.05, sparse_output=False)),
        ])
    blocos = []
    if num:
        blocos.append(("num", etapa_num, num))
    if cat:
        blocos.append(("cat", etapa_cat, cat))
    return ColumnTransformer(blocos, remainder="drop")


def k_vizinhos_seguro(y_treino: pd.Series, teto: int = 3) -> int:
    """O SMOTE exige mais amostras da classe minoritaria do que vizinhos.

    Calculado sobre a particao de TREINO do fold, nunca sobre o total.
    """
    menor = int(y_treino.value_counts().min())
    return max(1, min(teto, menor - 1))


_MEMORIA = None


def ativar_cache(caminho) -> None:
    """Liga o cache de transformadores ajustados do sklearn.

    Para um mesmo conjunto de variaveis e uma mesma particao de treino, o
    preprocessador ajustado e identico entre modelos e entre bracos de
    balanceamento -- o SMOTE vem depois dele. Sem cache, o IterativeImputer e
    reajustado nove vezes para cada par (conjunto, fold).

    Nao introduz vazamento: a chave do cache inclui os dados de entrada, entao
    um transformador ajustado num fold nunca e reaproveitado em outro.
    """
    global _MEMORIA
    from joblib import Memory
    _MEMORIA = Memory(location=str(caminho), verbose=0)


ACEITAM_AUSENTE = {"floresta", "gradiente"}


def construir_pipeline(estimador, num: list[str], cat: list[str],
                       braco: str, y_treino: pd.Series | None = None,
                       tratamento: str | None = None):
    """Monta o pipeline completo para um braco experimental.

    braco='peso'   -> ponderacao de classes no estimador, sem SMOTE
    braco='smote'  -> SMOTE no treino, sem ponderacao
    braco='nenhum' -> sem correcao (referencia da ablacao)

    Ponderacao e SMOTE nunca sao acumulados: sao duas solucoes para o mesmo
    problema, e usar as duas e correcao dupla.
    """
    if braco not in {"peso", "smote", "nenhum"}:
        raise ValueError(f"braco desconhecido: {braco}")

    passos = [("prep", construir_preprocessador(num, cat, tratamento))]
    if braco == "smote":
        k = k_vizinhos_seguro(y_treino) if y_treino is not None else 3
        passos.append(("smote", SMOTE(random_state=SEED, k_neighbors=k)))
    passos.append(("clf", estimador))
    return ImbPipeline(passos, memory=_MEMORIA)


# ---------------------------------------------------------------------------
# Estimadores
# ---------------------------------------------------------------------------

class XGBOrdinal(ClassifierMixin, BaseEstimator):
    """Adaptador do XGBClassifier para rotulos 1..4.

    Resolve duas incompatibilidades com o restante do pipeline:

    1. O XGBoost exige classes indexadas a partir de zero. Em vez de recodificar
       o alvo no experimento inteiro -- o que confundiria todas as metricas e a
       matriz de confusao --, o deslocamento fica contido aqui.

    2. O XGBoost nao aceita `class_weight`; a ponderacao e feita por
       `sample_weight` no fit. Sem esta traducao o braco de ponderacao seria
       silenciosamente identico ao braco sem correcao.
    """

    def __init__(self, class_weight=None, **kwargs):
        self.class_weight = class_weight
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return {"class_weight": self.class_weight, **self.kwargs}

    def set_params(self, **params):
        if "class_weight" in params:
            self.class_weight = params.pop("class_weight")
        self.kwargs.update(params)
        return self

    def fit(self, X, y, **kw):
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier
        y = np.asarray(y)
        self.deslocamento_ = int(y.min())
        self.classes_ = np.unique(y)
        if self.class_weight is not None and "sample_weight" not in kw:
            kw["sample_weight"] = compute_sample_weight(self.class_weight, y)
        self.modelo_ = XGBClassifier(**self.kwargs)
        self.modelo_.fit(X, y - self.deslocamento_, **kw)
        return self

    def predict(self, X):
        return self.modelo_.predict(X) + self.deslocamento_

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)


def estimadores(braco: str) -> dict:
    """Modelos candidatos, com hiperparametros contidos pelo tamanho da amostra.

    Grades maiores nao sao busca, sao ruido: com 123 registros e 18 na menor
    classe, explorar muitas combinacoes encontra variacao amostral.
    """
    peso = "balanced" if braco == "peso" else None
    modelos = {
        "logistica": LogisticRegression(
            max_iter=2000, class_weight=peso, random_state=SEED, C=1.0),
        "floresta": RandomForestClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            class_weight=peso, random_state=SEED, n_jobs=-1),
    }
    try:
        import xgboost  # noqa: F401
        modelos["gradiente"] = XGBOrdinal(
            class_weight=peso,
            max_depth=2, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=SEED, n_jobs=-1, verbosity=0)
        # objetivo e numero de classes NAO sao fixados: o XGBoost os infere do
        # alvo. Fixar num_class=4 quebrava o modo binario.
    except ImportError:
        print("  aviso: xgboost ausente, modelo 'gradiente' indisponivel")
    return modelos


# Sem busca de hiperparametros. Os valores acima foram fixados a priori pelo
# tamanho da amostra: com 123 registros e 18 na menor classe, uma busca
# encontraria variacao amostral, nao configuracao melhor. Declarado na
# metodologia como ausencia de busca, e nao busca reduzida.


# ---------------------------------------------------------------------------
# Baseline: a regra IRIS sobre a creatinina
# ---------------------------------------------------------------------------

def regra_iris(creatinina: float) -> float:
    """Estagio pela diretriz IRIS, em umol/L. Piso de comparacao do escopo 1.

    Faixas para gatos (IRIS, 2026): estagio 1 abaixo de 140; 2 de 140 a 250;
    3 de 251 a 440; 4 acima de 440. O limite superior do estagio 1 e exclusivo:
    140 exatos e estagio 2 (uma versao anterior usava "<= 140" e punha um gato
    com 140,0 no estagio 1; a correcao esta registrada na secao de testes).
    """
    if pd.isna(creatinina):
        return np.nan
    lim = cfg.LIMIARES_IRIS_UMOL
    if creatinina < lim[1][1]:
        return 1.0
    if creatinina <= lim[2][1]:
        return 2.0
    if creatinina <= lim[3][1]:
        return 3.0
    return 4.0


def aplicar_regra_iris(df: pd.DataFrame) -> pd.Series:
    """Regra IRIS aplicada a coluna de creatinina; ausente onde ela falta."""
    return df[cfg.COL_CREATININA].apply(regra_iris)
