"""
Analise de sobrevida das discordancias entre o modelo e a regra IRIS.

A pergunta: quando o modelo e a regra discordam, qual dos dois se alinha melhor
ao desfecho do animal?

Nao ha gabarito melhor que o rotulo para arbitrar a discordancia. Ha, porem, o
desfecho: se os gatos que o modelo estadia acima da regra morrem mais cedo, a
discordancia carrega informacao prognostica. Se as curvas se sobrepoem, nao
carrega.

ADVERTENCIA DE PODER, a declarar junto do resultado: a discordancia so pode ser
apurada nos registros que possuem creatinina, e sao poucos. Com essa contagem de
eventos, o teste de log-rank nao detecta diferenca a menos que ela seja enorme.
A analise e EXPLORATORIA; a ausencia de significancia nao e evidencia de
ausencia de efeito.

Uso:
    python sobrevida_discordancias.py
    python sobrevida_discordancias.py --pasta escopo1_nativo
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import modelagem as mdl

SAIDA = cfg.SAIDA_RELATORIOS / "escopo1_sobrevida"

# Forma das curvas, a mesma das demais figuras do texto: sem titulo dentro da
# imagem, acentos, virgula decimal, estagios em romanos, 300 dpi, largura final
# de 16 cm com letra de 10 pt, tracos distintos em tons de cinza e tabela de
# numero em risco sob o eixo. Os nomes dos grupos nas tabelas (CSV) nao mudam;
# so o rotulo mostrado na figura.
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})
_ROTULOS = {
    "estagio 1": "Estágio I", "estagio 2": "Estágio II",
    "estagio 3": "Estágio III", "estagio 4": "Estágio IV",
    "concordante": "Concordante", "modelo menos grave": "Modelo menos grave",
    "modelo mais grave": "Modelo mais grave",
    "predito avancado (III-IV)": "Predito avançado (III–IV)",
    "predito precoce (I-II)": "Predito precoce (I–II)",
}
_ESTILOS = [("#1f4e79", "-"), ("#7a7a7a", "--"), ("#000000", ":"), ("#5b8db8", "-.")]


def _rotulo(nome) -> str:
    """Rotulo do grupo como aparece na figura."""
    return _ROTULOS.get(str(nome), str(nome))


def _tabela_em_risco(ax, series: list, rotulos: list) -> None:
    """Numero em risco nos marcos do eixo x, abaixo do grafico."""
    # Distancias em pontos, fixas, abaixo do eixo x (1 pol = 72 pt).
    marcos = [x for x in ax.get_xticks() if ax.get_xlim()[0] <= x <= ax.get_xlim()[1]]
    trans = ax.get_xaxis_transform()
    def escreve(texto, x, coords, desloc, **kw):
        ax.annotate(texto, xy=(x, 0), xycoords=coords, xytext=(0, desloc),
                    textcoords="offset points", va="center", fontsize=10, **kw)
    escreve("Em risco", -0.06, "axes fraction", -40, ha="right", fontweight="bold")
    for k, (tempos, rot) in enumerate(zip(series, rotulos)):
        d = -40 - 14.4 * (k + 1)
        escreve(rot, -0.06, "axes fraction", d, ha="right")
        for x in marcos:
            escreve(str(int((tempos >= x).sum())), x, trans, d, ha="center")


def montar_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """Tempo em dias do diagnostico ate o obito ou a censura.

    A data de diagnostico e classificada como coluna a excluir e nao sobrevive a
    triagem; e relida do arquivo de origem e pareada pelo identificador. As
    datas sao convertidas com dia primeiro: sem isso a comparacao e alfabetica e
    silenciosamente errada.
    """
    if cfg.COL_DATA_DIAGNOSTICO in df.columns:
        diag = pd.to_datetime(df[cfg.COL_DATA_DIAGNOSTICO], dayfirst=True,
                              errors="coerce")
    else:
        bruto = pd.read_csv(cfg.resolver_bruto(), low_memory=False)
        bruto.columns = [c.strip() for c in bruto.columns]
        bruto = bruto.set_index(cfg.COL_ID)
        diag = pd.to_datetime(
            df[cfg.COL_ID].map(bruto[cfg.COL_DATA_DIAGNOSTICO]),
            dayfirst=True, errors="coerce")
        diag.index = df.index
    fim = pd.to_datetime(df["Censored date"], dayfirst=True, errors="coerce")
    obito = df["Death"]
    if obito.dtype == object:
        obito = obito.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])
    return pd.DataFrame({
        "tempo": (fim - diag).dt.days,
        "evento": obito.astype(bool).astype(int),
    }, index=df.index)


def classificar_discordancia(pred_modelo, pred_regra) -> pd.Series:
    """Concordante, modelo mais grave, modelo menos grave, ou sem regra (sem creatinina)."""
    d = pred_modelo - pred_regra
    return pd.Series(np.select(
        [d.isna(), d == 0, d > 0, d < 0],
        ["sem regra", "concordante", "modelo mais grave", "modelo menos grave"],
        default="sem regra"), index=pred_modelo.index)


def curvas(sob: pd.DataFrame, grupos: pd.Series, titulo: str, arquivo: str,
           minimo: int = 3) -> pd.DataFrame:
    """Curvas de Kaplan-Meier por grupo, com o teste de log-rank; grupos pequenos sao omitidos."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    # Tamanho final fixo (16 cm de largura); a margem esquerda e a de baixo
    # abrigam os rotulos e a tabela de numero em risco.
    n_g = sum(len(sob.loc[idx].dropna(subset=["tempo"])) >= minimo
              for idx in grupos.groupby(grupos).groups.values())
    linhas_leg = n_g if n_g <= 2 else -(-n_g // 2)
    baixo = 0.62 + 0.2 * (n_g + 1)            # rotulo do eixo e tabela, em pol
    cima = 0.22 * linhas_leg + 0.12           # legenda
    alto = baixo + 2.3 + cima
    fig = plt.figure(figsize=(16 / 2.54, alto))
    ax = fig.add_axes([0.33, baixo / alto, 0.64, 2.3 / alto])
    linhas = []
    usados = []
    tempos_grupos, rotulos_grupos = [], []
    for nome, idx in grupos.groupby(grupos).groups.items():
        s = sob.loc[idx].dropna(subset=["tempo"])
        if len(s) < minimo:
            print(f"    grupo omitido por ter menos de {minimo} registros: "
                  f"{nome} (n={len(s)})")
            continue
        km = KaplanMeierFitter()
        km.fit(s.tempo, s.evento,
               label=f"{_rotulo(nome)} (n = {len(s)}; óbitos: {int(s.evento.sum())})")
        cor, traco = _ESTILOS[len(usados) % len(_ESTILOS)]
        km.plot_survival_function(ax=ax, ci_show=len(s) >= 10, ci_alpha=.12,
                                  color=cor, linestyle=traco, linewidth=1.6)
        usados.append(nome)
        tempos_grupos.append(s.tempo.to_numpy())
        rotulos_grupos.append(_rotulo(nome))
        try:
            mediana = km.median_survival_time_
        except Exception:  # noqa: BLE001
            mediana = np.nan
        linhas.append({"grupo": nome, "n": len(s), "obitos": int(s.evento.sum()),
                       "mediana_sobrevida_dias": mediana,
                       "acompanhamento_mediano_dias": float(s.tempo.median())})

    from matplotlib.ticker import FuncFormatter
    ax.set_xlabel("Dias desde o diagnóstico")
    ax.set_ylabel("Proporção viva estimada")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.grid(alpha=.25)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.5, 1.0),
              ncol=1 if len(usados) <= 2 else 2, handlelength=2.6)
    _tabela_em_risco(ax, tempos_grupos, rotulos_grupos)
    print(f"    figura: {arquivo} ({titulo}; o titulo vai para a legenda no texto)")
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / arquivo, dpi=300)
    plt.close(fig)

    tab = pd.DataFrame(linhas)
    # O p do log-rank e gravado NA TABELA, nao apenas em atributos: atributos se
    # perdem ao salvar em CSV, e este e o numero que vai para o texto.
    tab["p_logrank"] = np.nan
    tab["eventos_no_teste"] = np.nan
    tab["grupos_omitidos"] = ", ".join(
        f"{g} (n={int((grupos == g).sum())})"
        for g in grupos.unique() if g not in usados) or "nenhum"
    if len(usados) >= 2:
        m = grupos.isin(usados) & sob.tempo.notna()
        r = multivariate_logrank_test(sob.tempo[m], grupos[m], sob.evento[m])
        tab["p_logrank"] = round(float(r.p_value), 5)
        tab["eventos_no_teste"] = int(sob.evento[m].sum())
        tab.attrs["p_logrank"] = float(r.p_value)
        tab.attrs["eventos_totais"] = int(sob.evento[m].sum())
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(description="Sobrevida das discordancias.")
    ap.add_argument("--pasta", default="escopo1_nativo",
                    help="pasta em saida/relatorios de onde ler as predicoes")
    args = ap.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    df, y, _ = mdl.carregar_coorte()
    # Le as predicoes da CONFIGURACAO DO SISTEMA. O arquivo generico
    # predicoes_fora_de_amostra.csv contem a configuracao que o passo [6]
    # elegeu, que pode ser outra.
    arq = cfg.SAIDA_RELATORIOS / args.pasta / "predicoes_sistema.csv"
    if not arq.exists():
        raise SystemExit(f"{arq} nao encontrado.\n"
                         f"Rode antes: python experimento_iris.py --ausentes nativo "
                         f"--modelos floresta gradiente --bracos peso nenhum")
    print(f"  predicoes: {arq.name} ({cfg.CONFIGURACAO_SISTEMA})")
    pred = pd.read_csv(arq)
    pred.index = df.index

    print("[1] montando tempo ate o desfecho")
    sob = montar_tempo(df)
    validos = sob.tempo.notna() & (sob.tempo >= 0)
    print(f"  registros com tempo calculavel: {int(validos.sum())} de {len(df)}")
    print(f"  obitos: {int(sob.evento[validos].sum())} "
          f"({sob.evento[validos].mean():.0%})")
    print(f"  acompanhamento mediano: {sob.tempo[validos].median():.0f} dias")

    print("\n[2] sobrevida por estagio registrado (verificacao de sanidade)")
    t = curvas(sob[validos], y[validos].map(lambda v: f"estagio {v}"),
               "Sobrevida por estagio IRIS registrado", "km_por_estagio.png")
    print(t.to_string(index=False))
    if "p_logrank" in t.attrs:
        print(f"  log-rank p = {t.attrs['p_logrank']:.4f}")
    print("  Se o estagio nao separar as curvas, o desfecho nao serve de arbitro")
    print("  para as discordancias, e o resto desta analise perde sentido.")

    print("\n[3] classificando discordancias entre modelo e regra")
    g = classificar_discordancia(pred.predito_modelo, pred.predito_regra)
    print(g.value_counts().to_string())
    m = validos & (g != "sem regra")
    print(f"  avaliaveis: {int(m.sum())} | obitos entre eles: "
          f"{int(sob.evento[m].sum())}")

    print("\n[4] sobrevida por tipo de discordancia")
    t2 = curvas(sob[m], g[m], "Sobrevida por concordancia entre modelo e regra IRIS",
                "km_discordancias.png")
    print(t2.to_string(index=False))
    if "p_logrank" in t2.attrs:
        p = t2.attrs["p_logrank"]
        ev = t2.attrs["eventos_totais"]
        print(f"  log-rank p = {p:.4f}  (eventos = {ev})")
        if ev < 20:
            print(f"  PODER INSUFICIENTE: com {ev} eventos, apenas diferencas muito")
            print("  grandes seriam detectaveis. Reportar como exploratorio e NAO")
            print("  interpretar ausencia de significancia como ausencia de efeito.")

    t2.to_csv(SAIDA / "sobrevida_discordancias.csv", index=False)
    t.to_csv(SAIDA / "sobrevida_por_estagio.csv", index=False)

    print("\n[5] o modelo separa sobrevida entre os registros SEM creatinina?")
    print("    (grupo em que a regra IRIS nao opera)")
    sem = validos & pred.predito_regra.isna()
    gm = pred.predito_modelo[sem].map(
        lambda v: "predito avancado (III-IV)" if v in (3, 4) else "predito precoce (I-II)")
    t3 = curvas(sob[sem], gm, "Sobrevida por predicao do modelo, sem creatinina",
                "km_sem_creatinina.png")
    print(t3.to_string(index=False))
    if "p_logrank" in t3.attrs:
        print(f"  log-rank p = {t3.attrs['p_logrank']:.4f} "
              f"(eventos = {t3.attrs['eventos_totais']})")
    print("    Esta e a analise de maior interesse pratico: mede se a predicao tem")
    print("    valor prognostico justamente onde nao existe alternativa.")
    t3.to_csv(SAIDA / "sobrevida_sem_creatinina.csv", index=False)

    import json
    resumo = {
        "n_com_tempo": int(validos.sum()),
        "obitos": int(sob.evento[validos].sum()),
        "acompanhamento_mediano_dias": float(sob.tempo[validos].median()),
        "por_estagio": {"p_logrank": t.attrs.get("p_logrank"),
                        "eventos": t.attrs.get("eventos_totais")},
        "discordancias": {"p_logrank": t2.attrs.get("p_logrank"),
                          "eventos": t2.attrs.get("eventos_totais"),
                          "distribuicao": g.value_counts().to_dict()},
        "sem_creatinina": {"p_logrank": t3.attrs.get("p_logrank"),
                           "eventos": t3.attrs.get("eventos_totais")},
    }
    (SAIDA / "resumo_sobrevida.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaidas em {SAIDA}")


if __name__ == "__main__":
    main()
