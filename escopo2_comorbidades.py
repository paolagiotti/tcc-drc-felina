"""
Escopo 2: ranqueamento de comorbidades.

Para cada gato, estima a probabilidade de cada uma das quatro comorbidades
modelaveis e as ordena. A saida para o veterinario e uma lista priorizada: "neste
caso, investigar primeiro hipertireoidismo, depois doenca periodontal...".

Desenho, e por que:

- COORTE: os 618 registros que satisfazem os criterios de inclusao, sem exigir
  estagio IRIS. As comorbidades nao dependem do estagio, e exigir o estagio
  descartaria 80% dos dados sem motivo.
- ALVOS: presenca ANTERIOR ou simultanea ao diagnostico de DRC (versao _pre).
  Presenca posterior seria informacao do futuro.
- ENTRADAS: as mesmas 18 variaveis do estadiamento. O mesmo tratamento nativo de
  ausentes e o mesmo braco sem correcao de desbalanceamento, pelo mesmo motivo:
  o ranqueamento usa as PROBABILIDADES, e elas precisam ser calibradas.
- DUAS FORMULACOES, ambas previstas na metodologia:
    independente -- um classificador por comorbidade (Binary Relevance);
    cadeia       -- cada classificador recebe as comorbidades anteriores da
                    cadeia como entrada (Classifier Chain). Testa se a
                    coocorrencia entre comorbidades carrega informacao.
- PISO: ordenar pela prevalencia de cada comorbidade na particao de treino --
  a mesma ordem para todos os gatos. O ranqueamento personalizado so se
  justifica se superar essa ordem fixa.

Uso:
    python escopo2_comorbidades.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

import config as cfg
import modelagem as mdl

warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")
SAIDA = cfg.SAIDA_RELATORIOS / "escopo2"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def carregar() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Coorte das comorbidades, variaveis de entrada do sistema e alvos binarios."""
    arq = cfg.SAIDA_DADOS / "coorte_comorbidades.parquet"
    if not arq.exists():
        raise SystemExit(f"{arq} nao encontrado. Rode antes: python triagem.py")
    df = pd.read_parquet(arq)
    conj = json.loads((cfg.SAIDA_DADOS / "conjuntos_variaveis.json").read_text())
    entradas = [c for c in conj[cfg.CONJUNTO_SISTEMA] if c in df.columns]
    alvos = [f"{c}_pre" for c in cfg.COMORBIDADES_MODELAVEIS
             if f"{c}_pre" in df.columns]
    for a in alvos:
        df[a] = df[a].astype(bool).astype(int)
    # Nenhuma comorbidade-alvo pode estar entre as entradas.
    sobrepostas = set(entradas) & set(alvos)
    if sobrepostas:
        raise SystemExit(f"alvo usado como entrada: {sobrepostas}")
    return df, entradas, alvos


def _estrato(df: pd.DataFrame, alvos: list[str]) -> pd.Series:
    """Estratifica pelo numero de comorbidades presentes (0, 1, 2 ou mais)."""
    return df[alvos].sum(axis=1).clip(upper=2)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def _pipe(num, cat):
    """Pipeline do sistema (gradiente, sem correcao de desbalanceamento, ausentes nativos)."""
    est = mdl.estimadores("nenhum")["gradiente"]
    return mdl.construir_pipeline(est, num, cat, "nenhum", None, "nativo")


def _prob_pos(pipe, X) -> np.ndarray:
    """Probabilidade da classe positiva; zeros se o treino so tinha uma classe."""
    p = pipe.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.zeros(len(X))


def ajustar_independente(Xtr, Ytr, Xte, entradas):
    """Binary Relevance: um classificador por comorbidade, ajustado no treino."""
    num, cat = mdl.separar_tipos(Xtr, entradas)
    saida = {}
    for a in Ytr.columns:
        pipe = _pipe(num, cat)
        pipe.fit(Xtr[num + cat], Ytr[a])
        saida[a] = _prob_pos(pipe, Xte[num + cat])
    return pd.DataFrame(saida, index=Xte.index)


def ajustar_cadeia(Xtr, Ytr, Xte, entradas, ordem):
    """Classifier Chain.

    No treino, cada elo recebe os valores REAIS das comorbidades anteriores; no
    teste, as PROBABILIDADES preditas por elas. E a formulacao padrao, e evita
    que o teste use informacao que nao existiria no uso real.
    """
    Xtr = Xtr.copy(); Xte = Xte.copy()
    saida = {}
    for a in ordem:
        num, cat = mdl.separar_tipos(Xtr, entradas + [c for c in ordem
                                                       if c in saida])
        pipe = _pipe(num, cat)
        pipe.fit(Xtr[num + cat], Ytr[a])
        saida[a] = _prob_pos(pipe, Xte[num + cat])
        Xtr[a] = Ytr[a].astype(float)
        Xte[a] = saida[a]
    return pd.DataFrame(saida, index=Xte.index)[list(Ytr.columns)]


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def metricas_por_alvo(Y, P, prev_treino) -> dict:
    """AUC, precisao media, prevalencia e Brier (contra o piso da prevalencia) por comorbidade."""
    r = {}
    for a in Y.columns:
        y, p = Y[a].values, P[a].values
        if y.min() == y.max():
            continue
        r[f"auc_{a}"] = roc_auc_score(y, p)
        r[f"ap_{a}"] = average_precision_score(y, p)
        r[f"prevalencia_{a}"] = y.mean()
        r[f"brier_{a}"] = brier_score_loss(y, p)
        r[f"brier_piso_{a}"] = brier_score_loss(y, np.full(len(y), prev_treino[a]))
    return r


def metricas_ranking(Y, P) -> dict:
    """Qualidade da ORDEM das comorbidades, por gato.

    Avaliada apenas nos gatos com ao menos uma comorbidade -- sem nenhuma, nao ha
    ordem a acertar.

    - acerto no topo (P@1): a comorbidade ranqueada em primeiro esta presente.
    - MAP: precisao media ao longo da lista; premia colocar as presentes no alto.
    """
    tem = Y.sum(axis=1) > 0
    Yv, Pv = Y[tem].values, P[tem].values
    if len(Yv) == 0:
        return {}
    topo, ap, ndcg = [], [], []
    rec = {k: [] for k in (1, 2, 3)}
    prec = {k: [] for k in (1, 2, 3)}
    for y, p in zip(Yv, Pv):
        ordem = np.argsort(-p, kind="stable")
        topo.append(y[ordem[0]])
        acertos, soma = 0, 0.0
        for k, idx in enumerate(ordem, 1):
            if y[idx]:
                acertos += 1
                soma += acertos / k
        ap.append(soma / y.sum())
        # NDCG: ganho descontado pela posicao, normalizado pelo ideal
        dcg = sum(y[idx] / np.log2(k + 1) for k, idx in enumerate(ordem, 1))
        ideal = sum(1 / np.log2(k + 1) for k in range(1, int(y.sum()) + 1))
        ndcg.append(dcg / ideal)
        # Recall@K e Precision@K, com K ate 3: com quatro rotulos, K maior
        # seria trivial (K=4 recupera tudo)
        for k in (1, 2, 3):
            topk = y[ordem[:k]]
            rec[k].append(topk.sum() / y.sum())
            prec[k].append(topk.sum() / k)
    r = {"p_at_1": float(np.mean(topo)), "map": float(np.mean(ap)),
         "ndcg": float(np.mean(ndcg)), "n_com_alguma": int(tem.sum())}
    for k in (1, 2, 3):
        r[f"recall_at_{k}"] = float(np.mean(rec[k]))
        r[f"precision_at_{k}"] = float(np.mean(prec[k]))
    return r


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Escopo 2: comorbidades.")
    ap.add_argument("--repeticoes", type=int, default=cfg.N_REPETICOES)
    args = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)

    print("[1] carregando coorte do escopo 2")
    df, entradas, alvos = carregar()
    print(f"  {len(df)} registros | {len(entradas)} variaveis de entrada")
    for a in alvos:
        print(f"    {a:<28} {int(df[a].sum()):>4} positivos ({df[a].mean():.1%})")
    n_com = int((df[alvos].sum(axis=1) > 0).sum())
    print(f"  com ao menos uma comorbidade: {n_com} ({n_com/len(df):.0%})")

    ordem_prev = list(df[alvos].mean().sort_values(ascending=False).index)
    print(f"  ordem da cadeia (por prevalencia): {[a.replace('_pre','') for a in ordem_prev]}")

    cv = RepeatedStratifiedKFold(n_splits=cfg.N_FOLDS, n_repeats=args.repeticoes,
                                 random_state=cfg.SEMENTE)
    splits = list(cv.split(df, _estrato(df, alvos)))
    print(f"\n[2] {len(splits)} particoes ({cfg.N_FOLDS} x {args.repeticoes})")

    linhas = []
    oof = {k: [] for k in ("independente", "cadeia", "prevalencia")}
    for i, (tr, te) in enumerate(splits, 1):
        Xtr, Xte = df.iloc[tr], df.iloc[te]
        Ytr, Yte = df.iloc[tr][alvos], df.iloc[te][alvos]
        prev = Ytr.mean()

        preds = {
            "independente": ajustar_independente(Xtr, Ytr, Xte, entradas),
            "cadeia": ajustar_cadeia(Xtr, Ytr, Xte, entradas, ordem_prev),
            "prevalencia": pd.DataFrame({a: np.full(len(te), prev[a])
                                         for a in alvos}, index=Xte.index),
        }
        for nome, P in preds.items():
            linhas.append({"particao": i, "formulacao": nome,
                           **metricas_por_alvo(Yte, P, prev),
                           **metricas_ranking(Yte, P)})
            oof[nome].append(P.assign(_rep=(i - 1) // cfg.N_FOLDS))
        print(f"  particao {i:>2}/{len(splits)}", end="\r")
    print(" " * 30, end="\r")

    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "metricas_por_particao.csv", index=False)

    # ---------------- resumo ----------------
    def med(col, form):
        s = t.loc[t.formulacao == form, col].dropna()
        return (s.median(), s.min(), s.max()) if len(s) else (np.nan,) * 3

    print("[3] discriminacao por comorbidade (AUC-ROC; 0,5 = acaso)")
    print(f"    {'comorbidade':<22}{'independente':>20}{'cadeia':>20}")
    resumo = []
    for a in alvos:
        mi, ci = med(f"auc_{a}", "independente"), med(f"auc_{a}", "cadeia")
        nome = a.replace("_pre", "")
        print(f"    {nome:<22}{mi[0]:>8.3f} [{mi[1]:.2f}-{mi[2]:.2f}]"
              f"{ci[0]:>8.3f} [{ci[1]:.2f}-{ci[2]:.2f}]")
        for form in ("independente", "cadeia"):
            m_auc, m_ap = med(f"auc_{a}", form), med(f"ap_{a}", form)
            m_b, m_bp = med(f"brier_{a}", form), med(f"brier_piso_{a}", form)
            resumo.append({"comorbidade": nome, "formulacao": form,
                           "auc": m_auc[0], "auc_min": m_auc[1], "auc_max": m_auc[2],
                           "ap": m_ap[0], "prevalencia": med(f"prevalencia_{a}", form)[0],
                           "brier": m_b[0], "brier_piso": m_bp[0]})

    print("\n[4] calibracao por comorbidade (Brier; deve ser menor que o piso)")
    r = pd.DataFrame(resumo)
    for _, x in r[r.formulacao == "independente"].iterrows():
        marca = "" if x.brier < x.brier_piso else "   <-- NAO supera o piso"
        print(f"    {x.comorbidade:<22} {x.brier:.4f}  piso {x.brier_piso:.4f}"
              f"  (AP {x.ap:.3f} vs prevalencia {x.prevalencia:.3f}){marca}")

    print("\n[5] qualidade do RANQUEAMENTO (gatos com ao menos uma comorbidade)")
    metr_rk = ["p_at_1", "map", "ndcg", "recall_at_1", "recall_at_2",
               "recall_at_3", "precision_at_2", "precision_at_3"]
    print(f"    {'formulacao':<14}" + "".join(f"{m:>15}" for m in metr_rk))
    rk = []
    for form in ("prevalencia", "independente", "cadeia"):
        lin = {"formulacao": form}
        txt = f"    {form:<14}"
        for m in metr_rk:
            v = med(m, form)
            lin[m], lin[f"{m}_min"], lin[f"{m}_max"] = v
            txt += f"{v[0]:>15.3f}"
        rk.append(lin)
        print(txt)

    # comparacao pareada contra o piso
    print("\n[6] ganho sobre o piso de prevalencia (pareado, mesmas particoes)")
    base = t[t.formulacao == "prevalencia"].set_index("particao")
    for form in ("independente", "cadeia"):
        f = t[t.formulacao == form].set_index("particao")
        for m in ("map", "ndcg", "p_at_1", "recall_at_2"):
            d = (f[m] - base[m]).dropna()
            zero = " (faixa inclui zero)" if d.min() <= 0 <= d.max() else ""
            print(f"    {form:<14}{m:<8} delta={d.median():+.3f} "
                  f"[{d.min():+.3f}, {d.max():+.3f}]  "
                  f"particoes a favor: {int((d > 0).sum())}/{len(d)}{zero}")

    r.to_csv(SAIDA / "resumo_por_comorbidade.csv", index=False)
    pd.DataFrame(rk).to_csv(SAIDA / "resumo_ranqueamento.csv", index=False)

    # probabilidades fora da amostra, media das repeticoes -- insumo da camada
    # interpretativa: a lista priorizada de cada gato
    P = (pd.concat(oof["independente"]).groupby(level=0).mean()
         .drop(columns="_rep").reindex(df.index))
    P.columns = [c.replace("_pre", "") for c in P.columns]
    ranking = P.apply(lambda l: " > ".join(l.sort_values(ascending=False).index),
                      axis=1)
    out = pd.concat([df[[cfg.COL_ID]] if cfg.COL_ID in df.columns else
                     pd.DataFrame(index=df.index), P.round(3)], axis=1)
    out["ranking"] = ranking
    out.to_csv(SAIDA / "probabilidades_por_gato.csv", index=False)

    (SAIDA / "execucao.json").write_text(json.dumps({
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "n": len(df), "alvos": alvos, "entradas": cfg.CONJUNTO_SISTEMA,
        "tratamento_ausentes": "nativo", "braco": "nenhum",
        "particoes": len(splits), "ordem_cadeia": ordem_prev,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaidas em {SAIDA}")


if __name__ == "__main__":
    main()
