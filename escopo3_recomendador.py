"""
Escopo 3: recomendacao priorizada de exames complementares.

Para um gato com exames faltando, qual exame pedir primeiro? O que mais
esclareceria o estagio.

MECANISMO -- reducao esperada de entropia.
  A entropia mede a duvida do modelo: maxima quando ele divide a probabilidade
  igualmente entre os quatro estagios, zero quando tem certeza de um. Para cada
  exame faltante, o sistema pergunta: se eu tivesse o resultado, quanto a
  duvida cairia?
  Como o resultado ainda nao existe, ele e ESPERADO: o sistema simula valores
  plausiveis do exame, tirados de gatos reais de cada estagio no treino, e
  pondera pela probabilidade que o proprio modelo atribui a cada estagio.
  Decisao de desenho: considera-se apenas a duvida sobre o ESTAGIO, e nao sobre
  as comorbidades, porque os exames candidatos sao renais e pouco informam sobre elas.

AVALIACAO -- simulacao retrospectiva por mascaramento.
  Nos gatos que tem dois ou mais exames, todos sao escondidos; a estrategia
  escolhe a ordem de revelacao; revela-se o valor REAL; mede-se a duvida e o
  acerto a cada passo. Comparam-se:
    recomendador -- a ordem pela reducao esperada de entropia, em TURNO UNICO
                    (lista produzida uma vez, como previsto na metodologia)
    com custo    -- a mesma, penalizada por custo, invasividade e acesso
    ordem fixa   -- sempre a mesma sequencia (bioquimica, urinalise, pressao, UPC)
    aleatoria    -- media exata sobre todas as ordens possiveis
    oraculo      -- a melhor ordem em retrospecto: o teto possivel

  Todos os modelos sao ajustados na particao de treino e aplicados a gatos da
  particao de teste. As estrategias sao comparadas PAREADAS, no mesmo gato.

LIMITACAO DE DESENHO:
  O modelo aprendeu que exame ausente significa, em parte, gravidade -- porque
  nos dados, faltar creatinina era "nunca foi pedida", muitas vezes porque o gato
  morreu antes. Na simulacao, faltar e "ainda nao foi pedida". O modelo le as duas
  coisas do mesmo jeito. Isso afeta os valores ABSOLUTOS de duvida, mas nao a
  comparacao entre estrategias, que enfrentam o mesmo vies.

Uso:
    python escopo3_recomendador.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import avaliacao as av
import config as cfg
import modelagem as mdl

warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")
SAIDA = cfg.SAIDA_RELATORIOS / "escopo3"
CLASSES = [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def entropia(p: np.ndarray) -> np.ndarray:
    """Entropia em bits. 0 = certeza; 2 = duvida maxima entre quatro estagios."""
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log2(p)).sum(axis=-1)


def rps(p: np.ndarray, y: int) -> float:
    """Ranked Probability Score de uma predicao (menor e melhor)."""
    real = np.array([1.0 if c == y else 0.0 for c in CLASSES])
    return float(np.sum((np.cumsum(p)[:-1] - np.cumsum(real)[:-1]) ** 2) / 3)


def mascarar(x: pd.Series, exames: dict, manter: set) -> pd.Series:
    """Esconde todas as variaveis dos exames que nao estao em `manter`."""
    x = x.copy()
    for e, vs in exames.items():
        if e not in manter:
            x[vs] = np.nan
    return x


# ---------------------------------------------------------------------------
# Reducao esperada de entropia
# ---------------------------------------------------------------------------

class Recomendador:
    """Reducao esperada de entropia de cada exame faltante, com valores amostrados do treino."""
    def __init__(self, pipe, colunas, Xtr, ytr, exames, amostras=15, semente=0):
        self.pipe, self.colunas, self.exames = pipe, colunas, exames
        rng = np.random.default_rng(semente)
        # Para cada exame e cada estagio: valores reais observados no treino.
        # Linhas inteiras do exame (ex.: creatinina E ureia juntas), para
        # preservar a correlacao entre as variaveis de um mesmo exame.
        self.banco = {}
        todos = {}
        for e, vs in exames.items():
            com = Xtr[vs].notna().any(axis=1)
            todos[e] = Xtr.loc[com, vs]
            for s in CLASSES:
                pool = Xtr.loc[com & (ytr == s), vs]
                if len(pool) > amostras:
                    pool = pool.sample(amostras, random_state=int(rng.integers(1e9)))
                self.banco[(e, s)] = pool
        # fallback: estagio sem nenhum gato com o exame usa o conjunto geral
        for e in exames:
            for s in CLASSES:
                if self.banco[(e, s)].empty:
                    geral = todos[e]
                    self.banco[(e, s)] = geral.sample(min(amostras, len(geral)),
                                                     random_state=0)

    def prob(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidades dos quatro estagios, na ordem de CLASSES, para cada linha."""
        p = self.pipe.predict_proba(X[self.colunas])
        cls = list(self.pipe.classes_) if hasattr(self.pipe, "classes_") \
            else list(self.pipe[-1].classes_)
        out = np.zeros((len(X), 4))
        for j, c in enumerate(cls):
            out[:, CLASSES.index(int(c))] = p[:, j]
        return out

    def ganho_esperado(self, x: pd.Series, candidatos: list[str]) -> dict:
        """Reducao esperada de entropia de cada exame candidato."""
        p0 = self.prob(x.to_frame().T)[0]
        h0 = float(entropia(p0))
        ganhos = {}
        for e in candidatos:
            vs = self.exames[e]
            blocos, pesos = [], []
            for k, s in enumerate(CLASSES):
                pool = self.banco[(e, s)]
                if pool.empty or p0[k] < 1e-6:
                    continue
                bloco = pd.DataFrame([x] * len(pool)).reset_index(drop=True)
                bloco[vs] = pool.values
                blocos.append(bloco)
                pesos.append(np.full(len(pool), p0[k] / len(pool)))
            if not blocos:
                ganhos[e] = 0.0
                continue
            h = entropia(self.prob(pd.concat(blocos, ignore_index=True)))
            w = np.concatenate(pesos)
            ganhos[e] = h0 - float(np.sum(w * h) / w.sum())
        return ganhos


def penalidade(e: str) -> float:
    """Penalidade de custo, invasividade e acesso de um exame (config.CUSTO_EXAMES)."""
    c = cfg.CUSTO_EXAMES[e]
    a, b, g = cfg.PESOS_CUSTO
    return a * c["custo"] + b * c["invasividade"] + g * c["indisponibilidade"]


def ordenar(ganhos: dict, com_custo: bool) -> list[str]:
    """Exames do maior para o menor ganho; com custo, ganho normalizado menos a penalidade."""
    if not com_custo:
        return sorted(ganhos, key=lambda e: -ganhos[e])
    # ganho normalizado pelo maior ganho da decisao, para ficar na mesma escala
    # da penalidade (0 a 1); o lambda regula quanto o custo pesa
    m = max(max(ganhos.values()), 1e-9)
    esc = {e: ganhos[e] / m - cfg.LAMBDA_CUSTO * penalidade(e) for e in ganhos}
    return sorted(esc, key=lambda e: -esc[e])


# ---------------------------------------------------------------------------
# Simulacao
# ---------------------------------------------------------------------------

def trajetoria(rec, x_real, y, ordem, exames) -> list[dict]:
    """Revela os exames na ordem dada e mede duvida e erro a cada passo."""
    out = []
    reveladas = set()
    for passo in range(len(ordem) + 1):
        x = mascarar(x_real, exames, reveladas)
        p = rec.prob(x.to_frame().T)[0]
        out.append({"passo": passo, "entropia": float(entropia(p)),
                    "rps": rps(p, y), "acerto": int(CLASSES[int(np.argmax(p))] == y),
                    "adjacente": int(abs(CLASSES[int(np.argmax(p))] - y) <= 1)})
        if passo < len(ordem):
            reveladas.add(ordem[passo])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Escopo 3: recomendacao de exames.")
    ap.add_argument("--repeticoes", type=int, default=1)
    ap.add_argument("--amostras", type=int, default=15)
    args = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)
    exames = cfg.EXAMES_RECOMENDAVEIS

    print("[1] carregando coorte e configuracao do sistema")
    df, y, conj = mdl.carregar_coorte()
    cs = cfg.CONFIGURACAO_SISTEMA
    colunas = conj[cs["conjunto"]]
    num, cat = mdl.separar_tipos(df, colunas)
    colunas = num + cat
    tem = pd.DataFrame({e: df[vs].notna().any(axis=1) for e, vs in exames.items()})
    aval = tem.sum(axis=1) >= 2
    print(f"  sistema: {cs['conjunto']} | {cs['modelo']} | {cs['braco']} | nativo")
    print(f"  exames candidatos: {list(exames)}")
    print(f"  gatos avaliaveis (2+ exames): {int(aval.sum())} de {len(df)} | "
          f"com os 4: {int((tem.sum(axis=1) == 4).sum())}")

    cv = RepeatedStratifiedKFold(n_splits=cfg.N_FOLDS, n_repeats=args.repeticoes,
                                 random_state=cfg.SEMENTE)
    fixa = list(cfg.ORDEM_FIXA_EXAMES)
    linhas, primeiras, recs_reais, parciais = [], [], [], []

    print(f"\n[2] simulacao em {cfg.N_FOLDS * args.repeticoes} particoes")
    for i, (tr, te) in enumerate(cv.split(df, y), 1):
        est = mdl.estimadores(cs["braco"])[cs["modelo"]]
        pipe = mdl.construir_pipeline(est, num, cat, cs["braco"], y.iloc[tr], "nativo")
        pipe.fit(df.iloc[tr][colunas], y.iloc[tr])
        rec = Recomendador(pipe, colunas, df.iloc[tr][colunas], y.iloc[tr], exames,
                           args.amostras, semente=i)

        for idx in df.index[te]:
            x_real = df.loc[idx, colunas]
            disp = [e for e in exames if tem.loc[idx, e]]

            # saida de uso real: recomendacao para o estado verdadeiro do gato
            if i <= cfg.N_FOLDS:
                faltam = [e for e in exames if not tem.loc[idx, e]]
                if faltam:
                    g = rec.ganho_esperado(x_real, faltam)
                    recs_reais.append({
                        "indice": idx, "iris_registrado": int(y[idx]),
                        "sem_creatinina": bool(pd.isna(df.loc[idx, cfg.COL_CREATININA])),
                        "recomendacao": " > ".join(ordenar(g, False)),
                        "recomendacao_com_custo": " > ".join(ordenar(g, True)),
                        **{f"ganho_{e}": round(v, 4) for e, v in g.items()}})

            if not aval[idx]:
                continue
            yi = int(y[idx])
            # TURNO UNICO, como na metodologia: a lista e produzida uma vez, a
            # partir do estado inicial, e seguida. Nao se recalcula a cada exame
            # revelado -- isso usaria valores reais que a lista nao conhece.
            x_zero = mascarar(x_real, exames, set())
            g0 = rec.ganho_esperado(x_zero, disp)
            estrategias = {
                "recomendador": ordenar(g0, False),
                "com custo": ordenar(g0, True),
                "ordem fixa": [e for e in fixa if e in disp],
            }
            trajs = {k: trajetoria(rec, x_real, yi, o, exames)
                     for k, o in estrategias.items()}
            # aleatoria (media exata) e oraculo (melhor em retrospecto)
            todas = [trajetoria(rec, x_real, yi, list(o), exames)
                     for o in itertools.permutations(disp)]
            trajs["aleatoria"] = [
                {m: float(np.mean([t[k][m] for t in todas])) if m != "passo" else k
                 for m in todas[0][k]} for k in range(len(disp) + 1)]
            trajs["oraculo"] = min(todas, key=lambda t: np.mean([p["rps"] for p in t[1:-1]])
                                   if len(t) > 2 else t[1]["rps"])
            for nome, t in trajs.items():
                for p in t:
                    linhas.append({"particao": i, "indice": idx, "estagio": yi,
                                   "n_exames": len(disp), "estrategia": nome, **p})
            # --- partida PARCIAL: o veterinario ja tem alguns exames ---
            # Partir do zero favorece a coincidencia com a ordem fixa (a
            # bioquimica e sempre a melhor primeira escolha). O teste justo da
            # adaptacao do recomendador e partir de subconjuntos ja conhecidos
            # e perguntar qual o PROXIMO, com ao menos duas opcoes restantes.
            for tam in range(1, len(disp) - 1):
                for conhecidos in itertools.combinations(disp, tam):
                    resto = [e for e in disp if e not in conhecidos]
                    x0 = mascarar(x_real, exames, set(conhecidos))
                    g = rec.ganho_esperado(x0, resto)
                    escolhas = {
                        "recomendador": ordenar(g, False)[0],
                        "com custo": ordenar(g, True)[0],
                        "ordem fixa": [e for e in fixa if e in resto][0],
                    }
                    res_prox = {}
                    for e in resto:
                        x1 = mascarar(x_real, exames, set(conhecidos) | {e})
                        p = rec.prob(x1.to_frame().T)[0]
                        res_prox[e] = rps(p, yi)
                    for nome, e in escolhas.items():
                        parciais.append({"particao": i, "indice": idx, "estagio": yi,
                                         "conhecidos": " + ".join(conhecidos),
                                         "estrategia": nome, "escolha": e,
                                         "rps": res_prox[e]})
                    parciais.append({"particao": i, "indice": idx, "estagio": yi,
                                     "conhecidos": " + ".join(conhecidos),
                                     "estrategia": "aleatoria", "escolha": "-",
                                     "rps": float(np.mean(list(res_prox.values())))})
                    parciais.append({"particao": i, "indice": idx, "estagio": yi,
                                     "conhecidos": " + ".join(conhecidos),
                                     "estrategia": "oraculo",
                                     "escolha": min(res_prox, key=res_prox.get),
                                     "rps": min(res_prox.values())})

            primeiras.append({"indice": idx, "estagio": yi,
                              "primeiro_recomendador": estrategias["recomendador"][0],
                              "primeiro_com_custo": estrategias["com custo"][0],
                              "primeiro_fixa": estrategias["ordem fixa"][0]})
        print(f"  particao {i}", end="\r")
    print(" " * 30, end="\r")

    t = pd.DataFrame(linhas)
    t.to_csv(SAIDA / "trajetorias.csv", index=False)

    # ------------------------------------------------------------------
    print("[3] duvida e erro apos o PRIMEIRO exame revelado")
    print("    (o passo em que a escolha da estrategia mais importa)")
    p1 = t[t.passo == 1]
    ordem_est = ["oraculo", "recomendador", "com custo", "ordem fixa", "aleatoria"]
    res = []
    for nome in ordem_est:
        s = p1[p1.estrategia == nome]
        res.append({"estrategia": nome, "entropia": s.entropia.mean(),
                    "rps": s.rps.mean(), "acerto": s.acerto.mean(),
                    "adjacente": s.adjacente.mean(), "n": len(s)})
    r = pd.DataFrame(res)
    base0 = t[t.passo == 0].groupby("indice").first()
    print(f"    ponto de partida (nenhum exame): entropia {base0.entropia.mean():.3f} "
          f"bits | RPS {base0.rps.mean():.3f}")
    print(f"    {'estrategia':<14}{'entropia':>10}{'RPS':>8}{'acerto':>9}{'adjac.':>9}")
    for _, x in r.iterrows():
        print(f"    {x.estrategia:<14}{x.entropia:>10.3f}{x.rps:>8.3f}"
              f"{x.acerto:>9.3f}{x.adjacente:>9.3f}")
    r.to_csv(SAIDA / "resumo_primeiro_exame.csv", index=False)

    print("\n[4] comparacao pareada no primeiro exame (mesmo gato)")
    piv = p1.pivot_table(index="indice", columns="estrategia", values="rps")
    for alvo in ("ordem fixa", "aleatoria"):
        for nome in ("recomendador", "com custo"):
            d = piv[alvo] - piv[nome]   # positivo = estrategia melhor que o alvo
            print(f"    {nome:<13} vs {alvo:<11} RPS menor em {int((d > 1e-9).sum()):>2}, "
                  f"igual em {int((d.abs() <= 1e-9).sum()):>2}, maior em "
                  f"{int((d < -1e-9).sum()):>2} gatos | ganho medio {d.mean():+.4f}")
    cap = ((piv["aleatoria"] - piv["recomendador"]) /
           (piv["aleatoria"] - piv["oraculo"]).replace(0, np.nan)).dropna()
    print(f"    fracao do ganho possivel capturada pelo recomendador "
          f"(0 = aleatorio, 1 = oraculo): mediana {cap.median():.2f}")

    print("\n[5] curva ao longo dos passos (media de RPS, gatos com 4 exames)")
    q = t[t.n_exames == 4]
    if len(q):
        c = q.pivot_table(index="passo", columns="estrategia", values="rps", aggfunc="mean")
        print("    " + c[[e for e in ordem_est if e in c]].round(3)
              .to_string().replace("\n", "\n    "))
        c.to_csv(SAIDA / "curva_rps_4exames.csv")

    if parciais:
        pp = pd.DataFrame(parciais)
        pp.to_csv(SAIDA / "partida_parcial.csv", index=False)
        print("\n[5b] partida PARCIAL: ja ha exames; qual o proximo?")
        n_sit = pp[pp.estrategia == "recomendador"].shape[0]
        print(f"    situacoes avaliadas: {n_sit}")
        rp = pp.groupby("estrategia").rps.mean()
        for e in ("oraculo", "recomendador", "com custo", "ordem fixa", "aleatoria"):
            print(f"    {e:<14} RPS medio {rp[e]:.4f}")
        pv = pp.pivot_table(index=["indice", "conhecidos", "particao"],
                            columns="estrategia", values="rps")
        ch = pp.pivot_table(index=["indice", "conhecidos", "particao"],
                            columns="estrategia", values="escolha", aggfunc="first")
        difere = ch["recomendador"] != ch["ordem fixa"]
        d = (pv["ordem fixa"] - pv["recomendador"])[difere]
        print(f"    recomendador escolhe diferente da ordem fixa em "
              f"{int(difere.sum())} de {len(difere)} situacoes ({difere.mean():.0%})")
        if difere.any():
            print(f"      nessas, RPS menor em {int((d > 1e-9).sum())}, maior em "
                  f"{int((d < -1e-9).sum())} | ganho medio {d.mean():+.4f}")
        d2 = pv["aleatoria"] - pv["recomendador"]
        print(f"    contra a aleatoria: ganho medio {d2.mean():+.4f}, "
              f"melhor em {int((d2 > 1e-9).sum())} de {len(d2)}")

    pr = pd.DataFrame(primeiras)
    pr.to_csv(SAIDA / "primeiro_exame_por_gato.csv", index=False)
    print("\n[6] qual exame e recomendado primeiro")
    for col in ("primeiro_recomendador", "primeiro_com_custo"):
        print(f"    {col.replace('primeiro_', ''):<13} "
              f"{pr[col].value_counts().to_dict()}")
    conc = (pr.primeiro_recomendador == pr.primeiro_com_custo).mean()
    print(f"    concordancia com e sem custo no primeiro exame: {conc:.0%}")

    rr = pd.DataFrame(recs_reais)
    if len(rr):
        rr.to_csv(SAIDA / "recomendacoes_estado_real.csv", index=False)
        print("\n[7] uso real: recomendacao para o estado verdadeiro de cada gato")
        print(f"    gatos com algum exame faltando: {len(rr)}")
        pr1 = rr.recomendacao.str.split(" > ").str[0]
        print(f"    primeiro exame recomendado: {pr1.value_counts().to_dict()}")
        sc = rr[rr.sem_creatinina]
        if len(sc):
            print(f"    nos {len(sc)} sem creatinina, bioquimica em primeiro: "
                  f"{(sc.recomendacao.str.startswith('Bioquímica')).mean():.0%}")

    (SAIDA / "execucao.json").write_text(json.dumps({
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "sistema": cs, "exames": exames, "ordem_fixa": fixa,
        "custos": cfg.CUSTO_EXAMES, "pesos_custo": cfg.PESOS_CUSTO,
        "lambda_custo": cfg.LAMBDA_CUSTO, "amostras": args.amostras,
        "particoes": cfg.N_FOLDS * args.repeticoes,
        "avaliaveis": int(aval.sum()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaidas em {SAIDA}")
    print("\nLIMITACAO: o modelo le exame ausente como sinal de gravidade. Na")
    print("simulacao, ausente e 'ainda nao pedido'. Afeta valores absolutos de")
    print("duvida, nao a comparacao entre estrategias.")


if __name__ == "__main__":
    main()
