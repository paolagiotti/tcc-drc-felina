"""
Figuras do capitulo de resultados, geradas das saidas da execucao final (nunca
de numeros digitados). Padrao unico: 16 cm de largura (MEIA para pares lado a
lado), letra de 10 pt, 300 dpi, virgula decimal, estagios em romanos, sem
titulo dentro da imagem.

Uso:
    python figuras.py
    python figuras.py --relatorios entregas/rodada_final_XXXX/relatorios
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

CINZA = "#7a7a7a"
DESTAQUE = "#1f4e79"
CLARO = "#c9d6e3"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 300,
    # tamanho final = figsize, sem recorte "tight"; o layout restrito acomoda
    # os rotulos
    "savefig.bbox": "standard", "figure.constrained_layout.use": True,
})
LARGURA = 16 / 2.54          # 16 cm, largura do texto
MEIA = 7.7 / 2.54            # metade, com espaco entre as duas
ROMANOS = {1: "I", 2: "II", 3: "III", 4: "IV"}

# Nomes das variaveis em portugues; usados tambem pelas fichas e pela camada.
NOMES_PT = {
    "Creatinine at diagnosis": "Creatinina",
    "USG at diagnosis": "Densidade urinária",
    "Reason for presentation": "Motivo da apresentação",
    "Age": "Idade",
    "BUN at diagnosis": "Ureia",
    "Blood pressure at first measurement": "Pressão arterial",
    "Neuter": "Estado reprodutivo",
    "Goitre": "Bócio",
    "Sex": "Sexo",
    "Heart Murmur_pre": "Sopro cardíaco",
    "Previous exposure to nephrotoxic drugs": "Fármacos nefrotóxicos",
    "Previous anaesthetic": "Anestesia prévia",
    "UPC at diagnosis": "UPC",
    "Phosphorus at diagnosis": "Fósforo",
    "Breed": "Raça",
    "Dipstick protein at diagnosis": "Proteinúria (fita)",
    "Gallop rhythm_pre": "Ritmo de galope",
    "Arrhythmia_pre": "Arritmia",
}


def _br(v: float, casas: int = 3) -> str:
    """Numero com virgula decimal e sinal de menos tipografico."""
    s = f"{v:+.{casas}f}" if v < 0 else f"{v:.{casas}f}"
    return s.replace(".", ",").replace("-", "−")


def _eixo_br(ax, eixo: str = "y", casas: int | None = None) -> None:
    """Rotulos de eixo com virgula decimal."""
    from matplotlib.ticker import FuncFormatter
    def fmt(x, _):
        t = f"{x:g}" if casas is None else f"{x:.{casas}f}"
        return t.replace(".", ",")
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(FuncFormatter(fmt))


def _tira_eixo(ax):
    """Remove as bordas superior e direita."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


def _salvar(fig, destino: Path, nome: str) -> None:
    """Grava a figura em PNG e fecha."""
    fig.savefig(destino / f"{nome}.png")
    plt.close(fig)
    print(f"  {nome}.png")


# ---------------------------------------------------------------- dados

def fig_ausencia(destino: Path) -> None:
    """Presenca de creatinina por estagio registrado."""
    df, y, _ = __import__("modelagem").carregar_coorte()
    tem = df[cfg.COL_CREATININA].notna()
    est = [1, 2, 3, 4]
    pct = [100 * tem[y == e].mean() for e in est]
    n = [int((y == e).sum()) for e in est]
    k = [int(tem[y == e].sum()) for e in est]
    fig, ax = plt.subplots(figsize=(LARGURA, 1.7))
    cores = [CLARO, CLARO, CLARO, DESTAQUE]
    barras = ax.barh([f"Estágio {ROMANOS[e]}" for e in est][::-1], pct[::-1],
                     color=cores[::-1], edgecolor="black", linewidth=.6, height=.62)
    for b, p, kk, nn in zip(barras, pct[::-1], k[::-1], n[::-1]):
        ax.text(p + 1.5, b.get_y() + b.get_height() / 2,
                f"{p:.0f}% ({kk} de {nn})", va="center", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Gatos com creatinina medida (%)")
    _tira_eixo(ax)
    _salvar(fig, destino, "fig_ausencia_creatinina_por_estagio")


# --------------------------------------------------------- estadiamento

def _linha_sistema(pasta: Path) -> pd.Series | None:
    """Linha da configuracao do sistema no resumo de configuracoes de uma pasta."""
    cs = cfg.CONFIGURACAO_SISTEMA
    arq = pasta / "resumo_configuracoes.csv"
    if not arq.exists():
        return None
    t = pd.read_csv(arq)
    m = ((t.conjunto == cs["conjunto"]) & (t.modelo == cs["modelo"])
         & (t.braco == cs["braco"]))
    return t[m].iloc[0] if m.any() else None


def fig_tratamentos(rel: Path, pastas: dict, destino: Path) -> None:
    """Configuracao do sistema sob imputacao, indicador e tratamento nativo."""
    metricas = [("kappa_quadratico", "Kappa quadrático", False),
                ("acuracia_balanceada", "Acurácia balanceada", False),
                ("rps", "RPS (menor é melhor)", True)]
    linhas = {nome: _linha_sistema(rel / p) for nome, p in pastas.items()}
    if any(v is None for v in linhas.values()):
        print("  (tratamentos: pasta ausente ou sem a configuracao do sistema; pulada)")
        return
    fig, eixos = plt.subplots(1, 3, figsize=(LARGURA, 3.0))
    rotulos = list(pastas.keys())
    for ax, (col, titulo, menor) in zip(eixos, metricas):
        med = [linhas[r][col] for r in rotulos]
        lo = [linhas[r].get(f"{col}_min", np.nan) for r in rotulos]
        hi = [linhas[r].get(f"{col}_max", np.nan) for r in rotulos]
        err = np.array([np.subtract(med, lo), np.subtract(hi, med)])
        cores = [CLARO, CLARO, DESTAQUE]
        ax.bar(rotulos, med, color=cores, edgecolor="black", linewidth=.6,
               yerr=None if np.isnan(err).any() else err, capsize=4,
               error_kw={"elinewidth": .8, "ecolor": CINZA})
        topo = [h if not np.isnan(h) else v for h, v in zip(hi, med)]
        for i, (v, tp) in enumerate(zip(med, topo)):
            ax.text(i, tp + .02 * max(topo), _br(v), ha="center", va="bottom",
                    fontsize=10)
        ax.set_ylim(0, max(topo) * 1.18)
        ax.set_title(titulo)            # nome do painel, nao titulo da figura
        _eixo_br(ax, "y", 2 if menor else 1)
        ax.tick_params(axis="x", labelrotation=30)
        for r in ax.get_xticklabels():
            r.set_ha("right")
    _salvar(fig, destino, "fig_tratamentos_ausentes")


def _matriz(m: pd.DataFrame, xlabel: str, ylabel: str, destino: Path, nome: str) -> None:
    """Matriz 4 x 4 em cinza, com a fronteira I/II destacada."""
    fig, ax = plt.subplots(figsize=(MEIA, MEIA * .95))
    ax.imshow(m.values, cmap="Greys", vmin=0, vmax=max(m.values.max(), 1) * 1.15)
    for i in range(4):
        for j in range(4):
            v = m.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=10,
                    color="white" if v > m.values.max() * .55 else "black")
    ax.add_patch(plt.Rectangle((-.5, -.5), 2, 2, fill=False, lw=1.6,
                               ls="--", edgecolor="black"))
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(list(ROMANOS.values())); ax.set_yticklabels(list(ROMANOS.values()))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines[:].set_visible(False)
    _salvar(fig, destino, nome)


def fig_matriz(rel: Path, pasta: str, destino: Path) -> None:
    """Matriz de confusao da configuracao do sistema."""
    arq = rel / pasta / "predicoes_sistema.csv"
    if not arq.exists():
        print("  (matriz: predicoes_sistema.csv ausente; pulada)")
        return
    ps = pd.read_csv(arq)
    m = pd.crosstab(ps.iris_registrado, ps.predito_modelo).reindex(
        index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)
    _matriz(m, "Estágio predito pelo sistema", "Estágio registrado", destino,
            "fig_matriz_confusao_sistema")


def fig_teto_rotulo(rel: Path, pasta: str, destino: Path) -> None:
    """Regra IRIS aplicada a creatinina contra o estagio registrado."""
    arq = rel / pasta / "contingencia_regra_rotulo.csv"
    if not arq.exists():
        print("  (teto do rotulo: contingencia ausente; pulada)")
        return
    c = pd.read_csv(arq, index_col=0)
    c.columns = [int(float(x)) for x in c.columns]
    c = c.reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)
    print(f"    regra x registro: n = {c.values.sum()}, "
          f"concordancia = {np.trace(c.values)}")
    _matriz(c, "Estágio pela regra IRIS (creatinina)", "Estágio registrado", destino,
            "fig_teto_rotulo")


def fig_shap(rel: Path, destino: Path) -> None:
    """Importancia global (SHAP fora da amostra), com as variaveis de Bradley destacadas."""
    g = rel / "escopo1_shap" / "shap_global.csv"
    if not g.exists():
        print("  (shap: shap_global.csv ausente; pulada)")
        return
    t = pd.read_csv(g).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(LARGURA, 3.6))
    nomes = [NOMES_PT.get(v, v) for v in t.variavel]
    cores = [DESTAQUE if v in cfg.VARIAVEIS_BRADLEY else CLARO
             for v in t.variavel]
    ax.barh(nomes, t.proporcao, color=cores, edgecolor="black", linewidth=.6)
    for i, v in enumerate(t.proporcao):
        ax.text(v + .004, i, _br(v), va="center", fontsize=10)
    ax.set_xlabel("Proporção da explicação (SHAP, fora da amostra)")
    _eixo_br(ax, "x", 2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=DESTAQUE, edgecolor="black", linewidth=.6,
                             label="variáveis de Bradley et al. (2019)")],
              frameon=False, loc="lower right")
    _salvar(fig, destino, "fig_shap_global")


# (arquivo, comparado, metrica, grupo, rotulo da metrica, destaque)
ABLACOES = [
    ("escopo1_restrito_nativo", "SEM_CREATININA", "acuracia_balanceada",
     "Creatinina, quatro estágios (74)", "Acurácia balanceada", True),
    ("escopo1_restrito_nativo", "SEM_CREATININA", "kappa_quadratico",
     "Creatinina, quatro estágios (74)", "Kappa quadrático", True),
    ("escopo1_binario_restrito_nativo", "SEM_CREATININA", "acuracia_balanceada",
     "Creatinina, alvo binário (74)", "Acurácia balanceada", True),
    ("escopo1_nativo", "SEM_SUBESTADIAMENTO", "acuracia_balanceada",
     "UPC e pressão arterial (123)", "Acurácia balanceada", False),
    ("escopo1_nativo", "SEM_SUBESTADIAMENTO", "kappa_quadratico",
     "UPC e pressão arterial (123)", "Kappa quadrático", False),
]


def fig_ablacoes(rel: Path, destino: Path) -> None:
    """Diferenca pareada (conjunto completo menos sem as variaveis) por metrica e grupo."""
    fig, ax = plt.subplots(figsize=(LARGURA, 2.4))
    n = len(ABLACOES)
    ys = np.arange(n)[::-1]
    for yi, (pasta, comp, met, grupo, rot, dest) in zip(ys, ABLACOES):
        t = pd.read_csv(rel / pasta / "ablacoes.csv")
        r = t[(t.modelo == "gradiente") & (t.braco == "nenhum")
              & (t.comparado == comp) & (t.metrica == met)]
        assert len(r) == 1, (pasta, comp, met)
        r = r.iloc[0]
        lo, hi, med = r.delta_min, r.delta_max, r.delta_mediano
        favor, contra = int(r.folds_a_favor), int(r.folds_contra)
        empate = int(r.n_folds) - favor - contra
        ax.plot([lo, hi], [yi, yi], color=CINZA, lw=1.2, zorder=1)
        for x in (lo, hi):
            ax.plot([x, x], [yi - .12, yi + .12], color=CINZA, lw=1.2)
        ax.plot(med, yi, "o", ms=7, zorder=3, markeredgecolor="black",
                markeredgewidth=.5, color=DESTAQUE if dest else CLARO)
        ax.text(hi + .015, yi, _br(med), va="center", fontsize=10)
        ax.text(.90, yi, f"{favor} / {contra} / {empate}", va="center",
                ha="center", fontsize=10)
    ax.text(.90, ys.max() + .72, "a favor / contra / empate", va="center",
            ha="center", fontsize=9, color=CINZA)
    ax.axvline(0, color="black", lw=.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g}\n{m}" for _, _, _, g, m, _ in ABLACOES])
    ax.set_ylim(-.6, n - .1)
    ax.set_xlim(-.2, 1.02)
    xt = [-.2, 0, .2, .4, .6]
    ax.set_xticks(xt)
    ax.set_xticklabels([_br(t, 1) for t in xt])
    ax.spines["bottom"].set_bounds(-.2, .6)
    ax.set_xlabel("Diferença (conjunto completo − sem as variáveis)", loc="left")
    _tira_eixo(ax)
    _salvar(fig, destino, "fig_ablacoes")


# --------------------------------------------------------- recomendacao

COMORB = [  # ordem por AUC decrescente; nome da coluna e nome no texto
    ("Hyperthyroidism", "Hipertireoidismo"),
    ("Arthritis", "Artrite"),
    ("Periodontal disease", "Doença periodontal"),
    ("Cystitis/FLUTD", "Cistite/DTUIF"),
]
METRICAS_COMORB = [
    ("map", "MAP"),
    ("ndcg", "NDCG"),
    ("p_at_1", "Acerto na primeira posição"),
]


def fig_comorbidades(rel: Path, destino: Path) -> None:
    """(a) AUC por comorbidade; (b) diferenca pareada entre o modelo e a ordem
    por prevalencia, por particao."""
    m = pd.read_csv(rel / "escopo2" / "metricas_por_particao.csv")
    ind = m[m.formulacao == "independente"].set_index("particao").sort_index()
    prev = m[m.formulacao == "prevalencia"].set_index("particao").sort_index()
    assert len(ind) == 15 and len(prev) == 15

    fig, (a, b) = plt.subplots(2, 1, figsize=(LARGURA, 3.2),
                               gridspec_kw={"height_ratios": [1.05, 0.95]})

    y = np.arange(len(COMORB))[::-1]
    for yi, (col, nome) in zip(y, COMORB):
        v = ind[f"auc_{col}_pre"].values
        a.plot([v.min(), v.max()], [yi, yi], color=CINZA, lw=1.2, zorder=1)
        for x in (v.min(), v.max()):
            a.plot([x, x], [yi - .14, yi + .14], color=CINZA, lw=1.2)
        a.plot(np.median(v), yi, "o", color=DESTAQUE, ms=7, zorder=3)
        a.text(v.max() + .012, yi, _br(np.median(v)), va="center", fontsize=10)
    a.axvline(.5, color="black", lw=.8, ls="--")
    a.text(.508, y.max() + .95, "acaso (0,5)", fontsize=9, va="center")
    a.set_yticks(y)
    a.set_yticklabels([n for _, n in COMORB])
    a.set_xlim(.45, 1.02)
    a.set_ylim(-.6, len(COMORB) - .3 + .7)
    xt = [.5, .6, .7, .8, .9, 1.0]
    a.set_xticks(xt)
    a.set_xticklabels([_br(t, 1) for t in xt])
    a.set_xlabel("AUC (mediana e faixa entre as 15 partições)", loc="left")
    a.set_title("(a) Discriminação de cada classificador", loc="left")
    _tira_eixo(a)

    rng = np.random.default_rng(7)
    ym = np.arange(len(METRICAS_COMORB))[::-1]
    for yi, (col, nome) in zip(ym, METRICAS_COMORB):
        d = ind[col].values - prev[col].values
        jit = rng.uniform(-.17, .17, size=d.size)
        b.scatter(d, yi + jit, s=22, facecolor=CLARO, edgecolor="black",
                  linewidth=.5, zorder=2)
        med = float(np.median(d))
        b.plot([med, med], [yi - .3, yi + .3], color=DESTAQUE, lw=2.4, zorder=3)
        mp, mm = np.median(prev[col].values), np.median(ind[col].values)
        b.text(.235, yi, f"{_br(mp)} → {_br(mm)}  (+{_br(med)})",
               va="center", ha="center", fontsize=10)
    b.text(.235, ym.max() + .95, "prevalência → modelo (ganho)", va="center",
           ha="center", fontsize=9, color=CINZA)
    b.axvline(0, color="black", lw=.8, ls="--")
    b.set_yticks(ym)
    b.set_yticklabels([n for _, n in METRICAS_COMORB])
    b.set_ylim(-.6, len(METRICAS_COMORB) - .3 + .7)
    b.set_xlim(-.05, .33)
    xt = [-.05, 0, .05, .10, .15]
    b.set_xticks(xt)
    b.set_xticklabels([_br(t, 2) for t in xt])
    b.spines["bottom"].set_bounds(-.05, .15)
    b.set_xlabel("Modelo − ordem por prevalência, por partição\n(positivo: modelo melhor)",
                 loc="left")
    b.set_title("(b) Ranqueamento contra a ordem por prevalência", loc="left")
    _tira_eixo(b)
    _salvar(fig, destino, "fig_comorbidades_resultados")


def fig_exames(rel: Path, destino: Path) -> None:
    """Ganho esperado do melhor exame no estado real, gatos com e sem a bioquimica."""
    arq = rel / "escopo3" / "recomendacoes_estado_real.csv"
    if not arq.exists():
        print("  (exames: recomendacoes_estado_real.csv ausente; pulada)")
        return
    r = pd.read_csv(arq)
    g = [c for c in r.columns if c.startswith("ganho_")]
    col_bioq = "ganho_Bioquímica renal"
    if col_bioq not in r.columns:
        raise SystemExit(f"{arq.name}: coluna '{col_bioq}' ausente; "
                         f"colunas de ganho: {g}")
    r["melhor"] = r[g].max(axis=1)
    sem_bioq = r[col_bioq].notna()      # ha ganho calculado = bioquimica faltando
    grupos = [("Com bioquímica\njá feita", r[~sem_bioq]),
              ("Sem bioquímica", r[sem_bioq])]
    rng = np.random.default_rng(42)     # so o espalhamento vertical dos pontos
    fig, ax = plt.subplots(figsize=(LARGURA, 2.8))
    for i, (_, d) in enumerate(grupos):
        y = i + rng.uniform(-.18, .18, len(d))
        ax.scatter(d.melhor, y, s=16, color=DESTAQUE if i else CLARO,
                   edgecolor="black", linewidth=.4, zorder=3)
        med = d.melhor.median()
        ax.plot([med, med], [i - .3, i + .3], color="black", lw=2, zorder=4)
        ax.text(med, i + .34, f"mediana {_br(med, 2)}", ha="center", va="bottom",
                fontsize=10)
    ax.axvline(0, color=CINZA, lw=.8, ls="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"{n}\n(n = {len(d)})" for n, d in grupos])
    ax.set_ylim(-.5, 1.8)
    ax.set_xlabel("Ganho esperado do melhor exame (bits)")
    _eixo_br(ax, "x", 1)
    _salvar(fig, destino, "fig_exames_ganho")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Figuras do capitulo de resultados.")
    ap.add_argument("--relatorios", default=None,
                    help="pasta relatorios da execucao (padrao: saida/relatorios)")
    ap.add_argument("--pastas", nargs=3, default=None,
                    metavar=("IMPUTAR", "INDICADOR", "NATIVO"),
                    help="pastas dos tres tratamentos, dentro de --relatorios")
    args = ap.parse_args()

    rel = Path(args.relatorios) if args.relatorios else cfg.SAIDA_RELATORIOS
    pastas = dict(zip(("Imputação", "Indicador", "Nativo"),
                      args.pastas or ("escopo1", "escopo1_indicador", "escopo1_nativo")))
    destino = cfg.SAIDA / "figuras" / "tcc"
    destino.mkdir(parents=True, exist_ok=True)

    print(f"lendo {rel}")
    print(f"gravando em {destino}\n")
    fig_ausencia(destino)
    fig_tratamentos(rel, pastas, destino)
    fig_matriz(rel, pastas["Nativo"], destino)
    fig_teto_rotulo(rel, pastas["Nativo"], destino)
    fig_shap(rel, destino)
    fig_ablacoes(rel, destino)
    fig_comorbidades(rel, destino)
    fig_exames(rel, destino)


if __name__ == "__main__":
    main()
