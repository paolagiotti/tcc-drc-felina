"""
Escopo 1: estadiamento IRIS. Executa baselines, modelos e ablacoes.

Ordem das saidas, a mesma do capitulo de resultados:

  1. baselines  -- piso de comparacao, sem o qual nenhuma metrica significa nada
  2. modelos    -- desempenho por estimador e por braco de balanceamento
  3. ablacoes   -- o experimento central: o que se perde sem a creatinina

Uso:
    python experimento_iris.py
    python experimento_iris.py --conjuntos FULL SEM_CREATININA --bracos peso
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
    for m in ("numpy", "pandas", "sklearn", "imblearn", "xgboost"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = "ausente"
    return out


def sensibilidade_por_creatinina(df: pd.DataFrame, y: pd.Series,
                                 pred: pd.Series) -> pd.DataFrame:
    """Acerto por classe, separado pela presenca de creatinina.

    E o recorte que sustenta o achado central do escopo 1: os casos graves sao
    os que mais frequentemente nao tem creatinina, e e neles que o modelo
    falha. Gerado automaticamente para que o numero saia da rodada final, e nao
    de um calculo manual sobre a configuracao que calhou de ser salva.
    """
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
    """Desempenho do modelo de QUATRO estagios na decisao clinica binaria.

    Nao e o alvo binario: o modelo e o mesmo, treinado nos quatro estagios. A
    saida e apenas lida na decisao que o veterinario toma -- manejo de doenca
    avancada (III-IV) ou nao --, que e onde a conduta muda.
    """
    import json
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
    """Pisos de comparacao. Nenhuma metrica de modelo significa nada sem isto."""
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
    """Tabela de contingencia entre o estagio registrado e a regra aplicada.

    Sustenta a reformulacao do argumento de circularidade: se a concordancia
    fosse total, o alvo seria funcao deterministica da creatinina e a tarefa
    seria trivial.
    """
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
    if tratamento == "nativo":
        if nome_modelo not in mdl.ACEITAM_AUSENTE:
            raise ValueError(
                f"o modelo '{nome_modelo}' nao aceita valores ausentes; "
                f"use --ausentes imputar ou indicador")
        if braco == "smote":
            # O SMOTE interpola entre registros e nao opera com ausentes. A
            # incompatibilidade e do metodo, nao um defeito da configuracao.
            raise ValueError(
                "SMOTE e incompativel com --ausentes nativo: a interpolacao "
                "exige valores completos. Use ponderacao ou nenhum.")
    num, cat = mdl.separar_tipos(df, colunas)
    estimador = mdl.estimadores(braco)[nome_modelo]

    def construir(y_treino):
        return mdl.construir_pipeline(estimador, num, cat, braco, y_treino,
                                      tratamento)

    return av.avaliar(construir, df[num + cat], y, splits)


# ---------------------------------------------------------------------------

# Metricas em que MENOR e melhor. Todas as demais sao maximizadas.
MINIMIZAR = {"rps", "brier_multiclasse", "log_loss"}


def validacao_aninhada(df, y, colunas, nomes_mod, bracos, tratamento,
                       criterio: str) -> tuple[pd.DataFrame, pd.Series]:
    """Estimativa honesta de desempenho, com a selecao dentro do treino.

    O problema que isto corrige: nas execucoes principais, todas as
    configuracoes sao avaliadas nas mesmas particoes e a melhor e escolhida
    olhando esses resultados. O numero da vencedora fica otimista, porque a
    propria validacao participou da selecao.

    Aqui, cada particao externa estima o desempenho de TODO o processo: dentro
    de sua porcao de treino, uma validacao interna escolhe modelo e braco; o
    escolhido e reajustado e aplicado a particao externa, nunca observada
    durante a escolha. Espera-se numero MENOR que o das execucoes principais --
    essa reducao e o vies que estava embutido.
    """
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
                if tratamento == "nativo" and (
                        nome_mod not in mdl.ACEITAM_AUSENTE or braco == "smote"):
                    continue
                est = mdl.estimadores(braco)[nome_mod]

                def construir(y_sub, _e=est, _b=braco):
                    return mdl.construir_pipeline(_e, num, cat, _b, y_sub,
                                                  tratamento)
                try:
                    por_fold, _ = av.avaliar(construir, X_tr, y_tr, interno)
                except Exception:  # noqa: BLE001
                    continue
                if criterio not in por_fold.columns:
                    continue
                valor = sinal * por_fold[criterio].median()
                if valor > melhor_valor:
                    melhor_valor, melhor = valor, (nome_mod, braco, est)

        if melhor is None:
            continue
        nome_mod, braco, est = melhor
        pipe = mdl.construir_pipeline(est, num, cat, braco, y_tr, tratamento)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X.iloc[te])
        pred_externa.iloc[te] = pred
        escolhidas.append(f"{nome_mod}|{braco}")
        linha = {"particao_externa": i, "escolhida": f"{nome_mod}|{braco}",
                 "criterio_interno": round(float(sinal * melhor_valor), 4),
                 **av.metricas(y.iloc[te], pred)}
        try:
            linha.update(av.metricas_probabilidade(
                y.iloc[te], pipe.predict_proba(X.iloc[te]), list(pipe.classes_)))
        except Exception:  # noqa: BLE001
            pass
        linhas.append(linha)

    por_particao = pd.DataFrame(linhas)
    return por_particao, pred_externa


def main() -> None:
    ap = argparse.ArgumentParser(description="Escopo 1: estadiamento IRIS.")
    ap.add_argument("--conjuntos", nargs="*", default=None,
                    help="subconjunto de conjuntos de variaveis a avaliar")
    ap.add_argument("--bracos", nargs="*", default=["peso", "smote", "nenhum"])
    ap.add_argument("--modelos", nargs="*", default=None)
    ap.add_argument("--sem-cache", action="store_true",
                    help="desliga o cache de transformadores ajustados")
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
    ap.add_argument("--criterio-selecao", default="rps",
                    help="criterio de selecao da validacao aninhada (padrao: rps, "
                         "regra de pontuacao propria e ordinal)")
    ap.add_argument("--conjunto-final", default=None,
                    help="conjunto de variaveis usado na validacao aninhada "
                         "(padrao: SEM_INDICADORES_EXAME)")
    ap.add_argument("--sufixo", default=None,
                    help="sufixo da pasta de saida; por padrao derivado das opcoes")
    args = ap.parse_args()

    if not args.sem_cache:
        mdl.ativar_cache(cfg.SAIDA / "cache")
    t0 = time.time()

    print("[1] carregando coorte da etapa 1")
    df, y, conjuntos = mdl.carregar_coorte()
    print(f"  {len(df)} registros | distribuicao: {y.value_counts().sort_index().to_dict()}")

    if args.restrito:
        # A regra IRIS so opera onde ha creatinina. Avaliar o modelo nos 123 e a
        # regra nos 74 compara populacoes diferentes: o modelo carrega os casos
        # em que a creatinina foi imputada, que a regra nao teria como estadiar.
        tem = df[cfg.COL_CREATININA].notna()
        df, y = df[tem].copy(), y[tem].copy()
        print(f"  RESTRITO aos registros com creatinina: {len(df)} | "
              f"{y.value_counts().sort_index().to_dict()}")

    if args.alvo == "binario":
        # A fronteira I/II concentra a maior parte dos erros e e borrada no
        # proprio rotulo. Colapsar testa se o desempenho baixo vem do modelo ou
        # da granularidade excessiva para o n disponivel.
        y = y.map({1: 1, 2: 1, 3: 2, 4: 2})
        print(f"  ALVO BINARIO: 1 = precoce (I-II), 2 = avancado (III-IV) | "
              f"{y.value_counts().sort_index().to_dict()}")

    av.definir_classes(y)
    tratamento = args.ausentes or getattr(cfg, "TRATAMENTO_AUSENTES", "imputar")
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

    nomes_conj = args.conjuntos or [c for c in conjuntos
                                    if len(conjuntos[c]) >= 3]
    nomes_mod = args.modelos or list(mdl.estimadores("peso").keys())

    total = len(nomes_conj) * len(nomes_mod) * len(args.bracos)
    print(f"\n[4] avaliando {len(nomes_conj)} conjuntos x {len(nomes_mod)} modelos "
          f"x {len(args.bracos)} bracos = {total} configuracoes x {len(splits)} folds")
    feitas = 0
    resumos, todos_folds, oofs = [], {}, {}

    for conj in nomes_conj:
        colunas = conjuntos[conj]
        for nome_mod in nomes_mod:
            for braco in args.bracos:
                chave = f"{conj}|{nome_mod}|{braco}"
                feitas += 1
                marcha = time.time()
                try:
                    por_fold, oof = rodar_configuracao(
                        df, y, colunas, braco, nome_mod, splits, tratamento)
                except ValueError as e:
                    if "incompativel" in str(e) or "nao aceita" in str(e):
                        print(f"  [{feitas:>2}/{total}] {chave:<52} PULADA: {e}")
                    else:
                        print(f"  [{feitas:>2}/{total}] {chave:<52} FALHOU: {e}")
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"  [{feitas:>2}/{total}] {chave:<52} "
                          f"FALHOU: {type(e).__name__}: {e}")
                    continue
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
        for braco in args.bracos:
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

    # Dois criterios de selecao. A acuracia balanceada pesa igualmente errar um
    # caso leve e um grave; clinicamente, deixar passar um caso grave e o erro
    # de maior custo. Quando os dois criterios divergem, a diferenca de acuracia
    # balanceada costuma estar dentro do ruido entre folds, e a de sensibilidade
    # na classe grave, nao.
    g = max(av.CLASSES)
    grave, f1_grave = f"sensibilidade_e{g}", f"f1_e{g}"

    # Elegiveis: apenas conjuntos CANDIDATOS a sistema (os controles existem para
    # medir contribuicao, nunca para serem escolhidos) e apenas configuracoes
    # cujas probabilidades superam o piso de prever a prevalencia.
    # O conjunto de variaveis NAO se decide por metrica: foi fixado por argumento
    # conceitual e arquitetural (config.CONJUNTO_SISTEMA). A selecao aqui escolhe
    # apenas modelo e braco, dentro do conjunto do sistema.
    CONTROLES = {"SEM_CREATININA", "SEM_CREATININA_NEM_SUBESTADIAMENTO",
                 "SO_AUSENCIAS", "DEFINIDOR_ESTAGIO", "SUBESTADIAMENTO"}
    conj_sistema = args.conjunto_final or getattr(
        cfg, "CONJUNTO_SISTEMA", "SEM_INDICADORES_EXAME")
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
    # O criterio clinico usa F1 da classe mais grave, nao sensibilidade pura.
    # Sensibilidade sozinha e maximizada por classificar tudo como grave; o F1
    # pondera com a precisao e impede essa degeneracao.
    criterios = {
        "acuracia_balanceada": elegiveis.sort_values(
            ["acuracia_balanceada", f1_grave], ascending=False).iloc[0],
        "rps": elegiveis.sort_values(
            ["rps", "acuracia_balanceada"], ascending=[True, False]).iloc[0],
    }
    print("\n[6] melhor configuracao por criterio: matriz de confusao")
    escolhas = {}
    for nome_crit, lin in criterios.items():
        chave_c = f"{lin['conjunto']}|{lin['modelo']}|{lin['braco']}"
        escolhas[nome_crit] = chave_c
        pred_c = av.consolidar_oof(oofs[chave_c])
        mc_c = av.matriz_confusao(y, pred_c)
        mc_c.to_csv(SAIDA / f"matriz_confusao_{nome_crit}.csv")
        print(f"\n  por {nome_crit}: {chave_c}")
        print(f"    acc_bal={lin['acuracia_balanceada']:.3f}  "
              f"sens_e{g}={lin[grave]:.2f}  prec_e{g}={lin[f'precisao_e{g}']:.2f}  "
              f"F1_e{g}={lin[f1_grave]:.2f}")
        print("    " + mc_c.to_string().replace("\n", "\n    "))

        if not args.restrito:
            spc = sensibilidade_por_creatinina(df, y, pred_c)
            spc.to_csv(SAIDA / f"sensibilidade_por_creatinina_{nome_crit}.csv", index=False)
            recorte = spc[spc.classe == max(av.CLASSES)]
            print(f"    classe mais grave, por presenca de creatinina:")
            for _, r in recorte.iterrows():
                if r["n"]:
                    print(f"      {r['grupo']:<16} {r['acertos']:>2}/{r['n']:<2} "
                          f"({r['sensibilidade']:.0%})")

    if escolhas["acuracia_balanceada"] != escolhas["rps"]:
        print("\n  OS CRITERIOS DIVERGEM. O criterio declarado e o RPS; a acuracia")
        print("  balanceada e reportada como referencia.")

    melhor = criterios["acuracia_balanceada"]
    chave = escolhas["acuracia_balanceada"]
    pred = av.consolidar_oof(oofs[chave])
    mc = av.matriz_confusao(y, pred)
    mc.to_csv(SAIDA / "matriz_confusao_melhor.csv")

    # --- predicoes da configuracao DO SISTEMA, explicitamente ---
    cs = getattr(cfg, "CONFIGURACAO_SISTEMA", None)
    if cs:
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
            print(f"\n  predicoes do SISTEMA gravadas: {chave_sis}")
        else:
            print(f"\n  AVISO: a configuracao do sistema ({chave_sis}) nao foi")
            print("  avaliada nesta execucao; predicoes_sistema.csv nao gerado.")

    saida_pred = pd.DataFrame({
        cfg.COL_ID: df[cfg.COL_ID] if cfg.COL_ID in df.columns else df.index,
        "iris_registrado": y,
        "predito_modelo": pred.astype("Int64"),
        "predito_regra": mdl.aplicar_regra_iris(df),
    })
    saida_pred.to_csv(SAIDA / "predicoes_fora_de_amostra.csv", index=False)

    (SAIDA / "execucao.json").write_text(json.dumps({
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "semente": cfg.SEMENTE,
        "n_folds": cfg.N_FOLDS, "n_repeticoes": cfg.N_REPETICOES,
        "alvo": args.alvo, "restrito_a_creatinina": bool(args.restrito),
        "tratamento_ausentes": tratamento,
        "n_avaliado": int(len(df)),
        "conjuntos": nomes_conj, "modelos": nomes_mod, "bracos": args.bracos,
        "melhor_configuracao": chave,
        "melhor_por_criterio": escolhas,
        "versoes": _versoes(),
        "concordancia_regra": resumo_conc,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Calibracao: o escopo 3 pontua exames pela reducao de entropia das
    # distribuicoes preditas. Se as probabilidades nao superarem o piso de
    # prever a prevalencia, a entropia nao tem significado utilizavel.
    if "brier_multiclasse" in tabela.columns:
        print("\n[7] calibracao das probabilidades (Brier multiclasse, menor e melhor)")
        for nome_crit, chave_c in escolhas.items():
            achado = tabela[(tabela.conjunto + "|" + tabela.modelo + "|"
                             + tabela.braco) == chave_c]
            if achado.empty:
                continue
            lin = achado.iloc[0]
            b, p = lin["brier_multiclasse"], lin["brier_piso_prevalencia"]
            print(f"  {chave_c}")
            print(f"    Brier={b:.3f}  piso(prevalencia)={p:.3f}  "
                  f"reducao={(p - b) / p:+.1%}  log_loss={lin.get('log_loss'):.3f}")
            if b >= p:
                print("    ATENCAO: nao supera o piso. As probabilidades nao tem")
                print("    significado utilizavel; o escopo 3 nao pode apoiar-se nelas.")

    if args.aninhada:
        conj_final = args.conjunto_final or getattr(
            cfg, "CONJUNTO_SISTEMA", "SEM_INDICADORES_EXAME")
        if conj_final not in conjuntos:
            conj_final = "FULL"
        crit = args.criterio_selecao
        print(f"\n[8] validacao aninhada sobre {conj_final}")
        print(f"  criterio de selecao interna: {crit}")
        por_part, pred_an = validacao_aninhada(
            df, y, conjuntos[conj_final], nomes_mod, args.bracos, tratamento, crit)
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
            print("\n  ESTIMATIVA HONESTA (mediana entre particoes externas):")
            for c in est.columns:
                print(f"    {c:<26} {est[c].median():.3f} "
                      f"[{est[c].min():.2f}-{est[c].max():.2f}]")
            # Referencia correta: a MESMA configuracao que a selecao escolheu,
            # avaliada sem aninhamento. Comparar com outra configuracao mede
            # diferenca entre modelos, nao vies de selecao.
            mais_escolhida = por_part["escolhida"].value_counts().idxmax()
            chave_ref = f"{conj_final}|{mais_escolhida}"
            ref = tabela[(tabela.conjunto + "|" + tabela.modelo + "|" + tabela.braco)
                         == chave_ref]
            if not ref.empty:
                o = ref.iloc[0]["acuracia_balanceada"]
                d = o - est["acuracia_balanceada"].median()
                print(f"\n    {chave_ref} sem aninhamento: {o:.3f}")
                print(f"    diferenca (sem aninhamento - aninhada): {d:+.3f}")
                print("    Ressalva: a aninhada usa 5 particoes externas unicas; a")
                print("    avaliacao principal usa 15 (5 x 3 repeticoes). Com n=123,")
                print("    diferencas desta ordem estao dentro da variacao entre")
                print("    particoes e nao devem ser lidas como vies.")
            esc = por_part["escolhida"].value_counts()
            print(f"\n    configuracoes escolhidas nas particoes: {esc.to_dict()}")
            if len(esc) > 1:
                print("    A selecao NAO e estavel entre particoes: a escolha de")
                print("    modelo/braco depende dos dados de treino, o que reforca")
                print("    que a diferenca entre candidatos esta dentro do ruido.")

    print(f"\nsaidas em {SAIDA}  |  tempo total: {(time.time() - t0)/60:.1f} min")
    print("\nLEITURA: nenhuma metrica isolada significa desempenho. Compare sempre")
    print("contra os baselines do passo [2], e reporte a faixa entre folds, nao a mediana.")


if __name__ == "__main__":
    main()
