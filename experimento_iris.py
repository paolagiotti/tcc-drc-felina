"""
Componente de estadiamento: pisos, modelos, ablacoes e validacao aninhada.

Para um tratamento de ausentes (--ausentes), avalia todas as combinacoes de
conjunto de variaveis, modelo e braco de desbalanceamento nas mesmas particoes,
compara os conjuntos por ablacao pareada, seleciona modelo e braco dentro do
conjunto do sistema pelo RPS e grava as predicoes da configuracao do sistema.
Com --aninhada, estima o desempenho de todo o processo de selecao.

Uso:
    python experimento_iris.py --ausentes nativo
    python experimento_iris.py --ausentes nativo --alvo binario --restrito
    python experimento_iris.py --ausentes nativo --aninhada --sufixo _final_aninhada
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import avaliacao as av
import config as cfg
import modelagem as mdl

SAIDA = cfg.SAIDA_RELATORIOS / "escopo1"   # redefinido em main() conforme as opcoes
BRACOS = ["peso", "nenhum"]
CRITERIO = "rps"                           # criterio de selecao: regra de pontuacao ordinal

# O IterativeImputer avisa que nao convergiu em 10 iteracoes. Com este numero de
# registros isso e esperado e nao afeta o resultado; o aviso so polui a saida.
warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")


# ---------------------------------------------------------------------------

def _versoes() -> dict:
    """Versoes das bibliotecas que afetam o resultado numerico.

    Mesma semente e mesmos dados produzem numeros diferentes entre versoes de
    xgboost e scikit-learn. Sem este registro, um resultado nao e reproduzivel.
    """
    import platform
    import sys
    out = {"python": sys.version.split()[0], "plataforma": platform.platform()}
    for m in ("numpy", "pandas", "sklearn", "xgboost"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = "ausente"
    return out


def sensibilidade_por_creatinina(df: pd.DataFrame, y: pd.Series,
                                 pred: pd.Series) -> pd.DataFrame:
    """Acerto por classe, separado pela presenca de creatinina."""
    tem = df[cfg.COL_CREATININA].notna()
    linhas = []
    for c in av.CLASSES:
        for presente, rotulo in ((True, "com creatinina"), (False, "sem creatinina")):
            m = (y == c) & (tem == presente)
            n = int(m.sum())
            acertos = int((pred[m] == c).sum()) if n else 0
            linhas.append({"classe": c, "grupo": rotulo, "n": n,
                           "acertos": acertos,
                           "sensibilidade": round(acertos / n, 3) if n else None})
    return pd.DataFrame(linhas)


def decisao_clinica(ps: pd.DataFrame, destino) -> dict:
    """Leitura clinica binaria (avancado, III-IV, ou nao) das predicoes do
    modelo de quatro estagios."""
    real = ps.iris_registrado.isin([3, 4])
    pred = ps.predito_modelo.isin([3, 4])
    tem = ps.predito_regra.notna()
    vp = int((real & pred).sum()); fp = int((~real & pred).sum())
    fn = int((real & ~pred).sum()); vn = int((~real & ~pred).sum())
    dif = (ps.iris_registrado - ps.predito_modelo).abs()
    q = ~tem
    r = {
        "n": int(len(ps)), "vp": vp, "fp": fp, "fn": fn, "vn": vn,
        "acuracia": round((vp + vn) / len(ps), 3),
        "sensibilidade": round(vp / (vp + fn), 3) if vp + fn else None,
        "especificidade": round(vn / (vn + fp), 3) if vn + fp else None,
        "vpp": round(vp / (vp + fp), 3) if vp + fp else None,
        "vpn": round(vn / (vn + fn), 3) if vn + fn else None,
        "erro_grave_2_ou_mais_estagios": round(float((dif >= 2).mean()), 3),
        "acuracia_adjacente": round(float((dif <= 1).mean()), 3),
        "abrangencia_modelo": 1.0,
        "abrangencia_regra_iris": round(float(tem.mean()), 3),
        "sem_creatinina_sensibilidade": round(
            float((real[q] & pred[q]).sum() / max(real[q].sum(), 1)), 3),
        # contagens por presenca de creatinina (tabela por tratamento do texto)
        "avancados_detectados_sem_creatinina": [int((real[q] & pred[q]).sum()), int(real[q].sum())],
        "avancados_detectados_com_creatinina": [int((real[tem] & pred[tem]).sum()), int(real[tem].sum())],
        "alarmes_falsos_sem_creatinina": int((~real[q] & pred[q]).sum()),
        "alarmes_falsos_com_creatinina": int((~real[tem] & pred[tem]).sum()),
        "estagio4_detectado_sem_creatinina": [
            int(((ps.iris_registrado == 4) & (ps.predito_modelo == 4) & q).sum()),
            int(((ps.iris_registrado == 4) & q).sum())],
        "estagio4_detectado_com_creatinina": [
            int(((ps.iris_registrado == 4) & (ps.predito_modelo == 4) & tem).sum()),
            int(((ps.iris_registrado == 4) & tem).sum())],
    }
    (destino / "decisao_clinica_sistema.json").write_text(
        json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[9] decisao clinica (avancado III-IV), configuracao do sistema")
    print(f"    acuracia {r['acuracia']:.1%} | sens {r['sensibilidade']:.1%} | "
          f"espec {r['especificidade']:.1%} | VPP {r['vpp']:.1%} | VPN {r['vpn']:.1%}")
    print(f"    erro de 2+ estagios {r['erro_grave_2_ou_mais_estagios']:.1%} | "
          f"abrangencia: modelo 100%, regra {r['abrangencia_regra_iris']:.0%}")
    return r


def _colapsar(s: pd.Series, binario: bool) -> pd.Series:
    """Quatro estagios em dois (1 = precoce, 2 = avancado) quando o alvo e binario."""
    return s.map({1: 1, 2: 1, 3: 2, 4: 2}) if binario else s


def rodar_baselines(df: pd.DataFrame, y: pd.Series, binario: bool = False) -> pd.DataFrame:
    """Pisos de comparacao: classe majoritaria e regra IRIS onde ha creatinina."""
    linhas = []

    # 1. classe majoritaria: o modelo preguicoso
    maj = int(y.value_counts().idxmax())
    linhas.append({"baseline": f"classe majoritaria (estagio {maj})",
                   "n_avaliado": int(len(y)),
                   **av.metricas(y, np.full(len(y), maj))})

    # 2. regra IRIS sobre a creatinina, onde ela existe
    regra = _colapsar(mdl.aplicar_regra_iris(df), binario)
    tem = regra.notna()
    linhas.append({"baseline": "regra IRIS sobre creatinina",
                   "n_avaliado": int(tem.sum()),
                   **av.metricas(y[tem], regra[tem])})
    return pd.DataFrame(linhas)


def concordancia_regra(df: pd.DataFrame, y: pd.Series,
                       binario: bool = False) -> tuple[pd.DataFrame, dict]:
    """Contingencia entre o estagio registrado e a regra IRIS aplicada a creatinina."""
    regra = _colapsar(mdl.aplicar_regra_iris(df), binario)
    tem = regra.notna()
    tab = pd.crosstab(y[tem], regra[tem].astype(int),
                      rownames=["IRIS registrado"], colnames=["regra"])
    resumo = {
        "n_com_creatinina": int(tem.sum()),
        "concordancia_exata": float((y[tem] == regra[tem]).mean()),
        "concordancia_adjacente": float((abs(y[tem] - regra[tem]) <= 1).mean()),
        "n_discordancias": int((y[tem] != regra[tem]).sum()),
    }
    return tab, resumo


def rodar_configuracao(df, y, colunas, braco, nome_modelo, splits, tratamento):
    """Avalia uma configuracao (conjunto, modelo, braco, tratamento) nos splits compartilhados."""
    num, cat = mdl.separar_tipos(df, colunas)

    def construir():
        est = mdl.estimadores(braco)[nome_modelo]
        return mdl.construir_pipeline(est, num, cat, braco, tratamento)

    return av.avaliar(construir, df[num + cat], y, splits)


# ---------------------------------------------------------------------------

# Metricas em que MENOR e melhor. Todas as demais sao maximizadas.
MINIMIZAR = {"rps", "brier_multiclasse"}


def validacao_aninhada(df, y, colunas, nomes_mod, bracos, tratamento,
                       criterio: str) -> tuple[pd.DataFrame, pd.Series]:
    """Validacao aninhada: em cada particao externa, uma validacao interna no
    treino escolhe modelo e braco; o escolhido e avaliado na particao externa."""
    from sklearn.model_selection import StratifiedKFold

    externo = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True,
                              random_state=cfg.SEMENTE)
    num, cat = mdl.separar_tipos(df, colunas)
    X = df[num + cat]
    linhas, escolhidas = [], []
    pred_externa = pd.Series(index=df.index, dtype="float")

    for i, (tr, te) in enumerate(externo.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        interno = list(StratifiedKFold(
            n_splits=cfg.N_FOLDS, shuffle=True,
            random_state=cfg.SEMENTE + i).split(X_tr, y_tr))

        sinal = -1.0 if criterio in MINIMIZAR else 1.0
        melhor, melhor_valor = None, -np.inf
        for nome_mod in nomes_mod:
            for braco in bracos:
                def construir(_m=nome_mod, _b=braco):
                    est = mdl.estimadores(_b)[_m]
                    return mdl.construir_pipeline(est, num, cat, _b, tratamento)
                por_fold, _ = av.avaliar(construir, X_tr, y_tr, interno)
                valor = sinal * por_fold[criterio].median()
                if valor > melhor_valor:
                    melhor_valor, melhor = valor, (nome_mod, braco)

        nome_mod, braco = melhor
        pipe = mdl.construir_pipeline(mdl.estimadores(braco)[nome_mod], num, cat,
                                      braco, tratamento)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X.iloc[te])
        pred_externa.iloc[te] = pred
        escolhidas.append(f"{nome_mod}|{braco}")
        linhas.append({"particao_externa": i, "escolhida": f"{nome_mod}|{braco}",
                       "criterio_interno": round(float(sinal * melhor_valor), 4),
                       **av.metricas(y.iloc[te], pred),
                       **av.metricas_probabilidade(
                           y.iloc[te], pipe.predict_proba(X.iloc[te]),
                           list(pipe.classes_))})

    por_particao = pd.DataFrame(linhas)
    return por_particao, pred_externa


def main() -> None:
    ap = argparse.ArgumentParser(description="Componente de estadiamento.")
    ap.add_argument("--alvo", choices=["quatro", "binario"], default="quatro",
                    help="quatro = estagios I a IV; binario = precoce (I-II) "
                         "contra avancado (III-IV)")
    ap.add_argument("--restrito", action="store_true",
                    help="avalia apenas os registros que possuem creatinina, "
                         "tornando o modelo comparavel a regra IRIS")
    ap.add_argument("--ausentes", choices=["imputar", "indicador", "nativo"],
                    default=None,
                    help="tratamento dos valores ausentes; ver config.TRATAMENTO_AUSENTES")
    ap.add_argument("--aninhada", action="store_true",
                    help="validacao aninhada: estima o desempenho de todo o "
                         "processo de selecao, nao apenas do modelo vencedor")
    ap.add_argument("--sufixo", default=None,
                    help="sufixo da pasta de saida; por padrao derivado das opcoes")
    args = ap.parse_args()

    mdl.ativar_cache(cfg.SAIDA / "cache")
    t0 = time.time()

    print("[1] carregando coorte da etapa 1")
    df, y, conjuntos = mdl.carregar_coorte()
    print(f"  {len(df)} registros | distribuicao: {y.value_counts().sort_index().to_dict()}")

    if args.restrito:
        # So os registros com creatinina, para comparar o modelo com a regra IRIS
        # na mesma populacao.
        tem = df[cfg.COL_CREATININA].notna()
        df, y = df[tem].copy(), y[tem].copy()
        print(f"  RESTRITO aos registros com creatinina: {len(df)} | "
              f"{y.value_counts().sort_index().to_dict()}")

    if args.alvo == "binario":
        # Precoce (I-II) contra avancado (III-IV): analise diagnostica.
        y = y.map({1: 1, 2: 1, 3: 2, 4: 2})
        print(f"  ALVO BINARIO: 1 = precoce (I-II), 2 = avancado (III-IV) | "
              f"{y.value_counts().sort_index().to_dict()}")

    av.definir_classes(y)
    tratamento = args.ausentes or cfg.TRATAMENTO_AUSENTES
    if tratamento != "imputar":
        print(f"  TRATAMENTO DE AUSENTES: {tratamento}")
    global SAIDA
    sufixo = args.sufixo
    if sufixo is None:
        sufixo = "" if args.alvo == "quatro" else f"_{args.alvo}"
        if args.restrito:
            sufixo += "_restrito"
        if tratamento != "imputar":
            sufixo += f"_{tratamento}"
    SAIDA = cfg.SAIDA_RELATORIOS / f"escopo1{sufixo}"
    SAIDA.mkdir(parents=True, exist_ok=True)

    print("\n[2] baselines")
    base = rodar_baselines(df, y, binario=args.alvo == "binario")
    for _, r in base.iterrows():
        print(f"  {r['baseline']:<36} n={r['n_avaliado']:>4}  "
              f"acc_bal={r['acuracia_balanceada']:.3f}  kappa_q={r['kappa_quadratico']:.3f}")
    base.to_csv(SAIDA / "baselines.csv", index=False)

    tab, resumo_conc = concordancia_regra(df, y, binario=args.alvo == "binario")
    print(f"\n  concordancia regra x rotulo: {resumo_conc['concordancia_exata']:.1%} exata, "
          f"{resumo_conc['concordancia_adjacente']:.1%} adjacente "
          f"({resumo_conc['n_discordancias']} discordancias em {resumo_conc['n_com_creatinina']})")
    tab.to_csv(SAIDA / "contingencia_regra_rotulo.csv")

    print("\n[3] materializando splits (compartilhados por todas as configuracoes)")
    splits = av.materializar_splits(df, y)
    print(f"  {len(splits)} folds = {cfg.N_FOLDS} particoes x {cfg.N_REPETICOES} repeticoes")

    nomes_conj = [c for c in conjuntos if len(conjuntos[c]) >= 3]
    nomes_mod = list(mdl.estimadores("peso").keys())

    total = len(nomes_conj) * len(nomes_mod) * len(BRACOS)
    print(f"\n[4] avaliando {len(nomes_conj)} conjuntos x {len(nomes_mod)} modelos "
          f"x {len(BRACOS)} bracos = {total} configuracoes x {len(splits)} folds")
    feitas = 0
    resumos, todos_folds, oofs = [], {}, {}

    for conj in nomes_conj:
        colunas = conjuntos[conj]
        for nome_mod in nomes_mod:
            for braco in BRACOS:
                chave = f"{conj}|{nome_mod}|{braco}"
                feitas += 1
                marcha = time.time()
                por_fold, oof = rodar_configuracao(
                    df, y, colunas, braco, nome_mod, splits, tratamento)
                r = av.resumir(por_fold)
                resumos.append({"conjunto": conj, "modelo": nome_mod, "braco": braco,
                                "n_variaveis": len(colunas), **r.to_dict()})
                todos_folds[chave] = por_fold
                oofs[chave] = oof
                brier = r.get("brier_multiclasse")
                piso = r.get("brier_piso_prevalencia")
                rps_v = r.get("rps")
                cal = f"  brier={brier:.3f}/{piso:.3f}" if brier == brier else ""
                if rps_v == rps_v and rps_v is not None:
                    cal += f"  rps={rps_v:.3f}/{r.get('rps_piso_prevalencia'):.3f}"
                decorrido = time.time() - marcha
                restante = (time.time() - t0) / feitas * (total - feitas)
                print(f"  [{feitas:>2}/{total}] {chave:<52} "
                      f"acc_bal={r['acuracia_balanceada']:.3f} "
                      f"[{r['acuracia_balanceada_min']:.2f}-{r['acuracia_balanceada_max']:.2f}]  "
                      f"kappa_q={r['kappa_quadratico']:.3f}  "
                      f"sens_e{max(av.CLASSES)}={r[f'sensibilidade_e{max(av.CLASSES)}']:.2f}{cal}  "
                      f"({decorrido:.0f}s, restam ~{restante/60:.0f}min)")

    tabela = pd.DataFrame(resumos).sort_values("acuracia_balanceada", ascending=False)
    tabela.to_csv(SAIDA / "resumo_configuracoes.csv", index=False)

    print("\n[5] ablacao da creatinina (comparacao pareada, mesmos folds)")
    ablacoes = []
    for nome_mod in nomes_mod:
        for braco in BRACOS:
            k_full = f"FULL|{nome_mod}|{braco}"
            for alvo in ("SEM_CREATININA", "SEM_SUBESTADIAMENTO",
                         "SEM_CREATININA_NEM_SUBESTADIAMENTO", "SEM_INDICADORES_EXAME"):
                k_alt = f"{alvo}|{nome_mod}|{braco}"
                if k_full not in todos_folds or k_alt not in todos_folds:
                    continue
                for m in ("acuracia_balanceada", "kappa_quadratico",
                          f"sensibilidade_e{max(av.CLASSES)}",
                          f"f1_e{max(av.CLASSES)}"):
                    d = av.comparar_pareado(todos_folds[k_full], todos_folds[k_alt], m)
                    ablacoes.append({"modelo": nome_mod, "braco": braco,
                                     "referencia": "FULL", "comparado": alvo, **d})
    if ablacoes:
        abl = pd.DataFrame(ablacoes)
        abl.to_csv(SAIDA / "ablacoes.csv", index=False)
        vis = abl[(abl.metrica == "acuracia_balanceada")]
        for _, r in vis.iterrows():
            marca = "  (faixa inclui zero)" if r["inclui_zero"] else ""
            print(f"  {r['modelo']:<11} {r['braco']:<7} FULL - {r['comparado']:<36} "
                  f"delta={r['delta_mediano']:+.3f} "
                  f"[{r['delta_min']:+.2f},{r['delta_max']:+.2f}]{marca}")

    g = max(av.CLASSES)
    grave, f1_grave = f"sensibilidade_e{g}", f"f1_e{g}"

    # Elegiveis: so o conjunto do sistema (os controles medem contribuicao, nunca
    # sao escolhidos) e so configuracoes cujas probabilidades superam o piso da
    # prevalencia. O conjunto de variaveis e fixado em config.CONJUNTO_SISTEMA;
    # a selecao escolhe apenas modelo e braco.
    CONTROLES = {"SEM_CREATININA", "SEM_CREATININA_NEM_SUBESTADIAMENTO",
                 "SO_AUSENCIAS", "DEFINIDOR_ESTAGIO", "SUBESTADIAMENTO"}
    conj_sistema = cfg.CONJUNTO_SISTEMA
    if conj_sistema in set(tabela.conjunto):
        elegiveis = tabela[tabela.conjunto == conj_sistema]
        print(f"\n  selecao restrita ao conjunto do sistema: {conj_sistema}")
    else:
        elegiveis = tabela[~tabela.conjunto.isin(CONTROLES)]
        print(f"\n  conjunto do sistema ({conj_sistema}) ausente desta execucao; "
              f"selecionando entre os candidatos avaliados")
    if "rps" in elegiveis.columns:
        calibradas = elegiveis[elegiveis.rps < elegiveis.rps_piso_prevalencia]
        if not calibradas.empty:
            elegiveis = calibradas
    print(f"\n  elegiveis para selecao: {len(elegiveis)} de {len(tabela)} "
          f"(excluidos controles e configuracoes abaixo do piso de calibracao)")
    lin = elegiveis.sort_values(["rps", "acuracia_balanceada"],
                                ascending=[True, False]).iloc[0]
    chave_c = f"{lin['conjunto']}|{lin['modelo']}|{lin['braco']}"
    print(f"\n[6] configuracao selecionada pelo RPS: {chave_c}")
    pred_c = av.consolidar_oof(oofs[chave_c])
    mc_c = av.matriz_confusao(y, pred_c)
    mc_c.to_csv(SAIDA / "matriz_confusao_rps.csv")
    print(f"    acc_bal={lin['acuracia_balanceada']:.3f}  "
          f"sens_e{g}={lin[grave]:.2f}  prec_e{g}={lin[f'precisao_e{g}']:.2f}  "
          f"F1_e{g}={lin[f1_grave]:.2f}")
    print("    " + mc_c.to_string().replace("\n", "\n    "))
    if not args.restrito:
        spc = sensibilidade_por_creatinina(df, y, pred_c)
        spc.to_csv(SAIDA / "sensibilidade_por_creatinina_rps.csv", index=False)
        recorte = spc[spc.classe == max(av.CLASSES)]
        print("    classe mais grave, por presenca de creatinina:")
        for _, r in recorte.iterrows():
            if r["n"]:
                print(f"      {r['grupo']:<16} {r['acertos']:>2}/{r['n']:<2} "
                      f"({r['sensibilidade']:.0%})")

    # Predicoes da configuracao do sistema (config.CONFIGURACAO_SISTEMA): e delas
    # que saem as analises descritivas.
    cs = cfg.CONFIGURACAO_SISTEMA
    chave_sis = f"{cs['conjunto']}|{cs['modelo']}|{cs['braco']}"
    if chave_sis in oofs:
        pred_sis = av.consolidar_oof(oofs[chave_sis])
        ps = pd.DataFrame({
            cfg.COL_ID: df[cfg.COL_ID] if cfg.COL_ID in df.columns else df.index,
            "iris_registrado": y,
            "predito_modelo": pred_sis.astype("Int64"),
            "predito_regra": mdl.aplicar_regra_iris(df),
        })
        ps.to_csv(SAIDA / "predicoes_sistema.csv", index=False)
        if not args.restrito and args.alvo == "quatro":
            decisao_clinica(ps, SAIDA)
        print(f"\n  predicoes do sistema gravadas: {chave_sis}")
    else:
        print(f"\n  AVISO: a configuracao do sistema ({chave_sis}) nao foi")
        print("  avaliada nesta execucao; predicoes_sistema.csv nao gerado.")

    (SAIDA / "execucao.json").write_text(json.dumps({
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "semente": cfg.SEMENTE,
        "n_folds": cfg.N_FOLDS, "n_repeticoes": cfg.N_REPETICOES,
        "alvo": args.alvo, "restrito_a_creatinina": bool(args.restrito),
        "tratamento_ausentes": tratamento,
        "n_avaliado": int(len(df)),
        "conjuntos": nomes_conj, "modelos": nomes_mod, "bracos": BRACOS,
        "selecionada_rps": chave_c,
        "versoes": _versoes(),
        "concordancia_regra": resumo_conc,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.aninhada:
        conj_final = cfg.CONJUNTO_SISTEMA
        print(f"\n[7] validacao aninhada sobre {conj_final} (criterio interno: {CRITERIO})")
        por_part, pred_an = validacao_aninhada(
            df, y, conjuntos[conj_final], nomes_mod, BRACOS, tratamento, CRITERIO)
        if por_part.empty:
            print("  nenhuma particao concluida.")
        else:
            por_part.to_csv(SAIDA / "validacao_aninhada_por_particao.csv",
                            index=False)
            for _, r in por_part.iterrows():
                print(f"    particao {int(r['particao_externa'])}: "
                      f"escolhida {r['escolhida']:<18} "
                      f"acc_bal={r['acuracia_balanceada']:.3f}  "
                      f"kappa_q={r['kappa_quadratico']:.3f}")
            cols = ["acuracia_balanceada", "kappa_quadratico",
                    f"sensibilidade_e{max(av.CLASSES)}",
                    f"precisao_e{max(av.CLASSES)}", "rps", "brier_multiclasse"]
            est = por_part[[c for c in cols if c in por_part.columns]]
            print("\n  estimativa aninhada (mediana entre particoes externas):")
            for c in est.columns:
                print(f"    {c:<26} {est[c].median():.3f} "
                      f"[{est[c].min():.2f}-{est[c].max():.2f}]")
            esc = por_part["escolhida"].value_counts()
            print(f"    configuracoes escolhidas nas particoes: {esc.to_dict()}")

    print(f"\nsaidas em {SAIDA}  |  tempo total: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
