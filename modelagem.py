"""
Pipeline de modelagem sem vazamento: tudo que aprende dos dados (imputar e
codificar) vive dentro do Pipeline e e reajustado em cada particao de treino.
Estimadores, adaptador do XGBoost e regra IRIS. Importado pelos scripts que
ajustam o modelo de estadiamento.
----

biblioteca, não roda sozinho 
Regra única: tudo que aprende dos dados (imputar, codificar) vive dentro do Pipeline e é reajustado 
em cada partição de treino, nunca no conjunto inteiro. 
Três pré-processadores: nativo (ausente passa ao modelo; só codificação das categóricas), imputar (IterativeImputer) e indicador (imputação mais uma coluna de ausência). 
Dois braços de desbalanceamento: peso (ponderação das classes) e nenhum. 
Dois estimadores com hiperparâmetros fixados a priori pelo tamanho da amostra, sem busca: floresta e XGBoost. 
O XGBOrdinal é um adaptador: o XGBoost exige classes a partir de zero e não aceita class_weight; 
sem ele, o braço de ponderação seria silenciosamente igual ao sem correção. 
A regra_iris é o piso de comparação (140 exatos é estágio 2).

"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
    """Imputacao e codificacao, ajustadas dentro da particao. Nao ha
    padronizacao: os dois modelos sao de arvores, que nao dependem da escala.
    `tratamento`: imputar, indicador ou nativo (config.TRATAMENTO_AUSENTES)."""
    tratamento = tratamento or cfg.TRATAMENTO_AUSENTES

    if tratamento == "nativo":
        # Sem imputacao: o ausente chega ao modelo.
        etapa_num = "passthrough"
        etapa_cat = Pipeline([
            ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                        min_frequency=0.05, sparse_output=False)),
        ])
    else:
        indicador = tratamento == "indicador"
        etapa_num = Pipeline([
            ("imputar", IterativeImputer(
                estimator=ExtraTreeRegressor(random_state=SEED),
                max_iter=10, random_state=SEED, initial_strategy="median",
                add_indicator=indicador)),
        ])
        etapa_cat = Pipeline([
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


_MEMORIA = None


def ativar_cache(caminho) -> None:
    """Cache dos transformadores ajustados (joblib): o preprocessador de um par
    (conjunto, particao) e o mesmo para todos os modelos e bracos. A chave do
    cache inclui os dados, entao nada ajustado numa particao vale em outra."""
    global _MEMORIA
    from joblib import Memory
    _MEMORIA = Memory(location=str(caminho), verbose=0)


def construir_pipeline(estimador, num: list[str], cat: list[str],
                       braco: str, tratamento: str | None = None) -> Pipeline:
    """Pipeline completo de um braco: 'peso' (ponderacao de classes, dentro do
    estimador) ou 'nenhum'."""
    if braco not in {"peso", "nenhum"}:
        raise ValueError(f"braco desconhecido: {braco}")
    return Pipeline([("prep", construir_preprocessador(num, cat, tratamento)),
                     ("clf", estimador)], memory=_MEMORIA)


# ---------------------------------------------------------------------------
# Estimadores
# ---------------------------------------------------------------------------

class XGBOrdinal(ClassifierMixin, BaseEstimator):
    """Adaptador do XGBClassifier para rotulos 1..4: desloca as classes para
    comecar em zero e traduz `class_weight` em `sample_weight`."""

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
    """Modelos candidatos, os dois de arvores e os dois aceitando valores
    ausentes, com hiperparametros fixados a priori pelo tamanho da amostra
    (sem busca)."""
    peso = "balanced" if braco == "peso" else None
    return {
        "floresta": RandomForestClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=3,
            class_weight=peso, random_state=SEED, n_jobs=-1),
        "gradiente": XGBOrdinal(
            class_weight=peso,
            max_depth=2, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=SEED, n_jobs=-1, verbosity=0),
    }


# ---------------------------------------------------------------------------
# Baseline: a regra IRIS sobre a creatinina
# ---------------------------------------------------------------------------

def regra_iris(creatinina: float) -> float:
    """Estagio pela diretriz IRIS (2026), em umol/L: 1 abaixo de 140; 2 de 140 a
    250; 3 de 251 a 440; 4 acima de 440. O limite superior do estagio 1 e
    exclusivo: 140 exatos e estagio 2."""
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
