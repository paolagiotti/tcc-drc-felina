"""
Bloco 3: avaliacao por validacao cruzada.

Dois pontos de desenho que sustentam tudo o que sai daqui:

1. Os splits sao materializados UMA vez e reaproveitados por todas as
   configuracoes. Sem isso a ablacao compararia conjuntos de variaveis em
   particoes diferentes, e o delta observado seria ruido de particao.

2. As metricas sao escolhidas para alvo ordinal e classes desiguais. Acuracia
   simples nao entra: com 47% de estagio 2, o modelo que chuta sempre 2 marca
   0,47 sem ter aprendido nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold

import config as cfg

SEED = cfg.SEMENTE

# Classes do alvo corrente. Ajustado por `definir_classes` quando o alvo e
# colapsado em binario, para que matriz de confusao e kappa usem os rotulos
# certos.
CLASSES = [1, 2, 3, 4]


def definir_classes(y: pd.Series) -> list[int]:
    """Fixa as classes do alvo corrente (quatro estagios ou binario) para as metricas."""
    global CLASSES
    CLASSES = sorted(int(c) for c in pd.unique(y))
    return CLASSES


# ---------------------------------------------------------------------------
# Splits compartilhados
# ---------------------------------------------------------------------------

def materializar_splits(X: pd.DataFrame, y: pd.Series,
                        n_splits: int | None = None,
                        n_repeats: int | None = None) -> list[tuple]:
    """Gera a lista de particoes uma unica vez, para uso por todas as configuracoes."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits or cfg.N_FOLDS,
        n_repeats=n_repeats or cfg.N_REPETICOES,
        random_state=SEED)
    return list(cv.split(X, y))


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def acuracia_adjacente(y_true, y_pred) -> float:
    """Acerto dentro de um estagio de distancia.

    Num alvo ordinal com fronteiras difusas, confundir 2 com 3 nao e o mesmo
    erro que confundir 1 com 4.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def metricas_probabilidade(y_true, proba, classes) -> dict:
    """Qualidade das probabilidades, nao so da classe eleita.

    Necessario porque o escopo 3 pontua exames pela reducao de entropia das
    distribuicoes preditas: se as probabilidades nao tiverem significado, a
    entropia tambem nao tem, e o recomendador opera sobre numeros vazios.

    - Brier multiclasse: erro quadratico medio entre a probabilidade atribuida
      e o vetor real (1 na classe certa, 0 nas demais). Menor e melhor; 0 e
      perfeito. O piso de referencia e o Brier de prever sempre a prevalencia.
    - Log loss: penaliza confianca alta em predicao errada. Menor e melhor.
    """
    y_true = np.asarray(y_true)
    classes = list(classes)
    real = np.zeros_like(proba, dtype=float)
    for i, c in enumerate(classes):
        real[:, i] = (y_true == c).astype(float)
    brier = float(np.mean(np.sum((proba - real) ** 2, axis=1)))

    # piso: prever sempre a prevalencia observada no proprio conjunto avaliado
    prev = real.mean(axis=0)
    brier_piso = float(np.mean(np.sum((np.tile(prev, (len(y_true), 1)) - real) ** 2,
                                      axis=1)))
    try:
        ll = float(log_loss(y_true, proba, labels=classes))
    except ValueError:
        ll = np.nan

    # RPS -- Ranked Probability Score. O Brier para classes ORDENADAS: compara as
    # distribuicoes acumuladas P(estagio <= k) prevista e real. Errar para o
    # estagio vizinho custa pouco; errar para o oposto custa muito. Como o
    # Brier, e regra de pontuacao propria (so e minimizado pelas probabilidades
    # verdadeiras), mas respeita a ordem do estadiamento.
    k = len(classes)

    def _rps(p):
        acum_p = np.cumsum(p, axis=1)[:, :-1]
        acum_r = np.cumsum(real, axis=1)[:, :-1]
        return float(np.mean(np.sum((acum_p - acum_r) ** 2, axis=1) / (k - 1)))

    rps = _rps(proba) if k > 1 else np.nan
    rps_piso = _rps(np.tile(prev, (len(y_true), 1))) if k > 1 else np.nan
    return {"brier_multiclasse": brier, "brier_piso_prevalencia": brier_piso,
            "log_loss": ll, "rps": rps, "rps_piso_prevalencia": rps_piso}


def metricas(y_true, y_pred) -> dict:
    """Metricas de classificacao do fold: globais e por classe."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    saida = {
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa_quadratico": cohen_kappa_score(y_true, y_pred, weights="quadratic",
                                              labels=CLASSES),
        "acuracia_adjacente": acuracia_adjacente(y_true, y_pred),
    }
    for c in CLASSES:
        presente = bool((y_true == c).any())
        if not presente:
            saida[f"sensibilidade_e{c}"] = np.nan
            saida[f"precisao_e{c}"] = np.nan
            saida[f"f1_e{c}"] = np.nan
            continue
        # Sensibilidade sozinha premia o exagero: um modelo que classifica todos
        # como a classe mais grave alcanca 1,0. Precisao e F1 dao o contrapeso.
        saida[f"sensibilidade_e{c}"] = recall_score(
            y_true, y_pred, labels=[c], average="macro", zero_division=0)
        saida[f"precisao_e{c}"] = precision_score(
            y_true, y_pred, labels=[c], average="macro", zero_division=0)
        saida[f"f1_e{c}"] = f1_score(
            y_true, y_pred, labels=[c], average="macro", zero_division=0)
    return saida


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def avaliar(construir_pipe, X: pd.DataFrame, y: pd.Series,
            splits: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda a validacao cruzada e devolve metricas por fold e predicoes fora de amostra.

    `construir_pipe` e uma funcao que recebe a serie y do TREINO e devolve o
    pipeline. A dependencia existe porque o k_neighbors do SMOTE depende da
    composicao do fold de treino, e nao pode ser fixado de fora.
    """
    linhas = []
    n_rep = len(splits) // cfg.N_FOLDS
    oof = pd.DataFrame(index=X.index,
                       columns=[f"rep{r}" for r in range(n_rep)], dtype="float")

    for i, (tr, te) in enumerate(splits):
        rep = i // cfg.N_FOLDS
        pipe = construir_pipe(y.iloc[tr])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        linha = {"fold": i, "repeticao": rep, **metricas(y.iloc[te], pred)}
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X.iloc[te])
                classes = list(pipe.classes_) if hasattr(pipe, "classes_") \
                    else list(pipe[-1].classes_)
                linha.update(metricas_probabilidade(y.iloc[te], proba, classes))
            except Exception:  # noqa: BLE001
                pass
        linhas.append(linha)
        oof.iloc[te, rep] = pred

    return pd.DataFrame(linhas), oof


def consolidar_oof(oof: pd.DataFrame) -> pd.Series:
    """Uma predicao por registro: a moda entre as repeticoes.

    Necessario antes de qualquer analise por registro (sobrevida, discordancias):
    usar as tres repeticoes separadamente triplicaria a amostra artificialmente.
    """
    return oof.mode(axis=1, dropna=True).iloc[:, 0]


def resumir(por_fold: pd.DataFrame) -> pd.Series:
    """Mediana e faixa entre folds. Valor pontual nao e reportado de proposito."""
    metr = [c for c in por_fold.columns if c not in ("fold", "repeticao")]
    linha = {}
    for m in metr:
        s = por_fold[m].dropna()
        if s.empty:
            linha[m] = np.nan
            continue
        linha[m] = s.median()
        linha[f"{m}_min"] = s.min()
        linha[f"{m}_max"] = s.max()
    return pd.Series(linha)


def matriz_confusao(y_true, y_pred) -> pd.DataFrame:
    """Matriz de confusao com as classes correntes nas linhas (real) e colunas (predito)."""
    m = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return pd.DataFrame(m,
                        index=[f"real_e{c}" for c in CLASSES],
                        columns=[f"pred_e{c}" for c in CLASSES])


def comparar_pareado(a: pd.DataFrame, b: pd.DataFrame, metrica: str) -> dict:
    """Diferenca fold a fold entre duas configuracoes avaliadas nos mesmos splits.

    Comparacao pareada tem mais poder que confrontar dois intervalos lado a
    lado, e so e valida porque os splits sao compartilhados.
    """
    d = (a[metrica] - b[metrica]).dropna()
    return {
        "metrica": metrica,
        "n_folds": int(len(d)),
        "delta_mediano": float(d.median()),
        "delta_min": float(d.min()),
        "delta_max": float(d.max()),
        "folds_a_favor": int((d > 0).sum()),
        "folds_contra": int((d < 0).sum()),
        "inclui_zero": bool(d.min() <= 0 <= d.max()),
    }
