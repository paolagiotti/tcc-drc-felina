"""Figuras complementares do capitulo de resultados: ranqueamento de comorbidades,
ablacoes do estadiamento e presenca de creatinina por estagio.

So le os CSV da execucao final (a pasta relatorios/ da rodada) e a coorte
gravada ao lado dela (dados/coorte.parquet), e desenha; nenhum modelo roda.
Mesmo padrao de figuras.py: 16 cm de largura, letra de 10 pt, 300 dpi, cores
CLARO/DESTAQUE/CINZA, virgula decimal, estagios em romanos.

Uso:
    python figuras_extra.py --relatorios entregas/rodada_final_XXXX/relatorios \\
                            --destino saida/figuras/tcc
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "standard", "figure.constrained_layout.use": True,
})
CINZA = "#7a7a7a"
DESTAQUE = "#1f4e79"
CLARO = "#c9d6e3"
LARGURA = 16 / 2.54


def _br(v, casas=3):
    """Numero com virgula decimal e sinal de menos tipografico."""
    s = f"{v:+.{casas}f}" if v < 0 else f"{v:.{casas}f}"
    return s.replace(".", ",").replace("-", "−")


def _tira_eixo(ax):
    """Remove as bordas superior e direita."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)


# ---------------------------------------------------------------- comorbidades
COMORB = [  # ordem por AUC decrescente; nome da coluna e nome no texto
    ("Hyperthyroidism", "Hipertireoidismo"),
    ("Arthritis", "Artrite"),
    ("Periodontal disease", "Doença periodontal"),
    ("Cystitis/FLUTD", "Cistite/DTUIF"),
]
METRICAS = [  # as tres metricas de ordenacao reportadas no texto
    ("map", "MAP"),
    ("ndcg", "NDCG"),
    ("p_at_1", "Acerto na primeira posição"),
]


def fig_comorbidades(rel: Path, destino: Path):
    """Dois paineis, um sobre o outro, na largura do texto, com a mesma gramatica
    da figura das ablacoes: ponto ou barra na mediana, faixa ou pontos das 15
    particoes, zero ou acaso tracejado, numeros a direita."""
    m = pd.read_csv(rel / "escopo2" / "metricas_por_particao.csv")
    ind = m[m.formulacao == "independente"].set_index("particao").sort_index()
    prev = m[m.formulacao == "prevalencia"].set_index("particao").sort_index()
    assert len(ind) == 15 and len(prev) == 15

    fig, (a, b) = plt.subplots(2, 1, figsize=(LARGURA, 3.2),
                               gridspec_kw={"height_ratios": [1.05, 0.95]})

    # (a) AUC por comorbidade: ponto na mediana, linha na faixa; Brier a direita
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

    # (b) diferenca pareada modelo - prevalencia, por particao; medianas a direita
    rng = np.random.default_rng(7)
    ym = np.arange(len(METRICAS))[::-1]
    for yi, (col, nome) in zip(ym, METRICAS):
        d = ind[col].values - prev[col].values
        jit = rng.uniform(-.17, .17, size=d.size)
        b.scatter(d, yi + jit, s=22, facecolor=CLARO, edgecolor="black",
                  linewidth=.5, zorder=2)
        med = float(np.median(d))
        b.plot([med, med], [yi - .3, yi + .3], color=DESTAQUE, lw=2.4, zorder=3)
        mp, mm = np.median(prev[col].values), np.median(ind[col].values)
        b.text(.235, yi, f"{_br(mp)} \u2192 {_br(mm)}  (+{_br(med)})",
               va="center", ha="center", fontsize=10)
    b.text(.235, ym.max() + .95, "prevalência \u2192 modelo (ganho)", va="center",
           ha="center", fontsize=9, color=CINZA)
    b.axvline(0, color="black", lw=.8, ls="--")
    b.set_yticks(ym)
    b.set_yticklabels([n for _, n in METRICAS])
    b.set_ylim(-.6, len(METRICAS) - .3 + .7)
    b.set_xlim(-.05, .33)
    xt = [-.05, 0, .05, .10, .15]
    b.set_xticks(xt)
    b.set_xticklabels([_br(t, 2) for t in xt])
    b.spines["bottom"].set_bounds(-.05, .15)
    b.set_xlabel("Modelo \u2212 ordem por prevalência, por partição\n(positivo: modelo melhor)",
                 loc="left")
    b.set_title("(b) Ranqueamento contra a ordem por prevalência", loc="left")
    _tira_eixo(b)

    fig.savefig(destino / "fig_comorbidades_resultados.png")
    plt.close(fig)
    # conferencia, para o texto
    for col, nome in METRICAS:
        d = ind[col].values - prev[col].values
        print(f"{col}: mediana {np.median(d):+.3f}; a favor {(d > 1e-9).sum()}, "
              f"igual {(np.abs(d) <= 1e-9).sum()}, contra {(d < -1e-9).sum()}")


# ------------------------------------------------------------------- ablacoes
LINHAS = [  # (arquivo, comparado, metrica, grupo, rotulo da metrica, destaque)
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


def fig_ablacoes(rel: Path, destino: Path):
    """Diferenca pareada (conjunto completo menos sem as variaveis) por metrica e grupo."""
    fig, ax = plt.subplots(figsize=(LARGURA, 2.4))
    n = len(LINHAS)
    ys = np.arange(n)[::-1]
    for yi, (pasta, comp, met, grupo, rot, dest) in zip(ys, LINHAS):
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
    ax.set_yticklabels([f"{g}\n{m}" for _, _, _, g, m, _ in LINHAS])
    ax.set_ylim(-.6, n - .1)
    ax.set_xlim(-.2, 1.02)
    xt = [-.2, 0, .2, .4, .6]
    ax.set_xticks(xt)
    ax.set_xticklabels([_br(t, 1) for t in xt])
    ax.spines["bottom"].set_bounds(-.2, .6)
    ax.set_xlabel("Diferença (conjunto completo − sem as variáveis)", loc="left")
    _tira_eixo(ax)
    fig.savefig(destino / "fig_ablacoes.png")
    plt.close(fig)


# ------------------------------------------------- creatinina por estagio (4.1)
ROMANOS = {1: "I", 2: "II", 3: "III", 4: "IV"}


def fig_ausencia(rel: Path, destino: Path):
    """Presenca de creatinina por estagio registrado, na largura do texto e baixa
    (a versao de figuras.py tem 16 x 7,4 cm; esta tem 16 x 4,3 cm, para caber na
    4.1 sem reduzir a letra abaixo de 10 pt). Le a coorte da execucao final."""
    df = pd.read_parquet(rel.parent / "dados" / "coorte.parquet")
    y = df["Iris stage at diagnosis"].astype(int)
    tem = df["Creatinine at diagnosis"].notna()
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
    fig.savefig(destino / "fig_ausencia_creatinina_por_estagio.png")
    plt.close(fig)
    print("creatinina por estagio:", list(zip(est, k, n)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--relatorios", required=True)
    p.add_argument("--destino", required=True)
    a = p.parse_args()
    rel, destino = Path(a.relatorios), Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    fig_comorbidades(rel, destino)
    fig_ablacoes(rel, destino)
    fig_ausencia(rel, destino)
    print("ok:", destino)


if __name__ == "__main__":
    main()
