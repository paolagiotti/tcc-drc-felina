"""
Figuras do capitulo de resultados, geradas a partir das saidas do experimento.

Cada figura sai de arquivos produzidos pela rodada final, e nunca de numeros
digitados a mao: se a rodada mudar, as figuras mudam junto. Estilo sobrio,
rotulos em portugues, 300 dpi, legivel em impressao em tons de cinza.

Regras de forma: sem titulo dentro da imagem (a legenda vai no texto, acima da
figura); virgula decimal; estagios em
algarismos romanos (I a IV), como no texto; desenhadas na largura final
(LARGURA = 16 cm; MEIA = metade, para pares lado a lado), com letra de 10 pt
em tudo, para incluir com width=\linewidth (ou 0.49\linewidth) sem reduzir.

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
    # Tamanho final = figsize: sem recorte "tight", que mudava a largura e,
    # com ela, o tamanho da letra no texto. O layout restrito acomoda rotulos.
    "savefig.bbox": "standard", "figure.constrained_layout.use": True,
})
LARGURA = 16 / 2.54          # 16 cm, largura do texto
MEIA = 7.7 / 2.54            # metade, com espaco entre as duas
ROMANOS = {1: "I", 2: "II", 3: "III", 4: "IV"}

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
    """Numero com virgula decimal, padrao do texto em portugues."""
    return f"{v:.{casas}f}".replace(".", ",")


def _eixo_br(ax, eixo: str = "y", casas: int | None = None) -> None:
    """Rotulos de eixo com virgula decimal."""
    from matplotlib.ticker import FuncFormatter
    def fmt(x, _):
        if casas is None:
            t = f"{x:g}"
        else:
            t = f"{x:.{casas}f}"
        return t.replace(".", ",")
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(FuncFormatter(fmt))


def _salvar(fig, destino: Path, nome: str) -> None:
    """Grava a figura em PNG e fecha."""
    fig.savefig(destino / f"{nome}.png")
    plt.close(fig)
    print(f"  {nome}.png")


# ---------------------------------------------------------------------------

def fig_ausencia_por_estagio(destino: Path) -> None:
    """Presenca de creatinina por estagio registrado (falta sobretudo no IV)."""
    df, y, _ = __import__("modelagem").carregar_coorte()
    tem = df[cfg.COL_CREATININA].notna()
    est = [1, 2, 3, 4]
    pct = [100 * tem[y == e].mean() for e in est]
    n = [int((y == e).sum()) for e in est]
    fig, ax = plt.subplots(figsize=(LARGURA, 2.9))
    cores = [CLARO, CLARO, CLARO, DESTAQUE]
    barras = ax.bar([f"Estágio {ROMANOS[e]}" for e in est], pct, color=cores,
                    edgecolor="black", linewidth=.6, width=.6)
    for b, p, k in zip(barras, pct, n):
        ax.text(b.get_x() + b.get_width() / 2, p + 2, f"{p:.0f}%\n(n = {k})",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Gatos com creatinina (%)")
    ax.set_xlabel("Estágio IRIS registrado")
    _salvar(fig, destino, "fig_ausencia_creatinina_por_estagio")


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
    """A escada entre imputacao, indicador e tratamento nativo."""
    metricas = [("kappa_quadratico", "Kappa quadrático", False),
                ("acuracia_balanceada", "Acurácia balanceada", False),
                ("rps", "RPS (menor é melhor)", True)]
    linhas = {nome: _linha_sistema(rel / p) for nome, p in pastas.items()}
    if any(v is None for v in linhas.values()):
        print("  (tratamentos: pasta ausente ou sem a configuracao do sistema -- pulada)")
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
        _eixo_br(ax, "y", 2 if menor else 1)   # casas fixas no eixo
        ax.tick_params(axis="x", labelrotation=30)
        for r in ax.get_xticklabels():
            r.set_ha("right")
    _salvar(fig, destino, "fig_tratamentos_ausentes")


def fig_sem_creatinina(rel: Path, pastas: dict, destino: Path) -> None:
    """Deteccao do estagio IV com e sem creatinina, por tratamento."""
    dados = {}
    for nome, p in pastas.items():
        arq = rel / p / "predicoes_sistema.csv"
        if not arq.exists():
            print("  (sem creatinina: predicoes_sistema.csv ausente -- pulada)")
            return
        ps = pd.read_csv(arq)
        tem = ps.predito_regra.notna()
        iv = ps.iris_registrado == 4
        ok = ps.predito_modelo == 4
        dados[nome] = (int((iv & tem & ok).sum()), int((iv & tem).sum()),
                       int((iv & ~tem & ok).sum()), int((iv & ~tem).sum()))
    fig, ax = plt.subplots(figsize=(LARGURA, 3.0))
    x = np.arange(len(dados))
    larg = .36
    com = [100 * a / b if b else 0 for a, b, _, _ in dados.values()]
    sem = [100 * c / d if d else 0 for _, _, c, d in dados.values()]
    b1 = ax.bar(x - larg / 2, com, larg, color=CLARO, edgecolor="black",
                linewidth=.6, label="com creatinina")
    b2 = ax.bar(x + larg / 2, sem, larg, color=DESTAQUE, edgecolor="black",
                linewidth=.6, label="sem creatinina")
    for barras, idx in ((b1, (0, 1)), (b2, (2, 3))):
        for b, v in zip(barras, dados.values()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                    f"{v[idx[0]]} de {v[idx[1]]}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(list(dados.keys()))
    ax.set_ylim(0, 135)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Estágio IV detectado (%)")
    ax.set_xlabel("Tratamento dos valores ausentes")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    _salvar(fig, destino, "fig_estagio4_com_sem_creatinina")


def fig_matriz(rel: Path, pasta: str, destino: Path) -> None:
    """Matriz de confusao do sistema, com a fronteira I/II destacada."""
    arq = rel / pasta / "predicoes_sistema.csv"
    if not arq.exists():
        print("  (matriz: predicoes_sistema.csv ausente -- pulada)")
        return
    ps = pd.read_csv(arq)
    m = pd.crosstab(ps.iris_registrado, ps.predito_modelo).reindex(
        index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)
    # Mesmo tamanho, escala e cinza da figura do teto do rotulo: as duas vao
    # lado a lado. Sem barra de cores: os numeros ja estao nas celulas.
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
    ax.set_xlabel("Estágio predito pelo sistema"); ax.set_ylabel("Estágio registrado")
    ax.spines[:].set_visible(False)
    _salvar(fig, destino, "fig_matriz_confusao_sistema")


def fig_teto_rotulo(rel: Path, pasta: str, destino: Path) -> None:
    """Regra IRIS aplicada a creatinina contra o estagio registrado."""
    arq = rel / pasta / "contingencia_regra_rotulo.csv"
    if not arq.exists():
        print("  (teto do rotulo: contingencia ausente -- pulada)")
        return
    c = pd.read_csv(arq, index_col=0)
    c.columns = [int(float(x)) for x in c.columns]
    c = c.reindex(index=[1, 2, 3, 4], columns=[1, 2, 3, 4], fill_value=0)
    fig, ax = plt.subplots(figsize=(MEIA, MEIA * .95))
    ax.imshow(c.values, cmap="Greys", vmin=0, vmax=max(c.values.max(), 1) * 1.15)
    for i in range(4):
        for j in range(4):
            v = c.values[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=10,
                    color="white" if v > c.values.max() * .55 else "black")
    ax.add_patch(plt.Rectangle((-.5, -.5), 2, 2, fill=False, lw=1.6,
                               ls="--", edgecolor="black"))
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(list(ROMANOS.values())); ax.set_yticklabels(list(ROMANOS.values()))
    ax.set_xlabel("Estágio pela regra IRIS (creatinina)")
    ax.set_ylabel("Estágio registrado")
    # n e concordancia vao para a legenda no texto (antes estavam no titulo)
    print(f"    regra x registro: n = {c.values.sum()}, "
          f"concordancia = {np.trace(c.values)}")
    ax.spines[:].set_visible(False)
    _salvar(fig, destino, "fig_teto_rotulo")


def fig_shap(rel: Path, destino: Path) -> None:
    """Importancia global (SHAP) e comparacao entre gatos com e sem creatinina."""
    g = rel / "escopo1_shap" / "shap_global.csv"
    cs = rel / "escopo1_shap" / "shap_com_sem_creatinina.csv"
    if g.exists():
        t = pd.read_csv(g).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(LARGURA, 3.6))
        nomes = [NOMES_PT.get(v, v) for v in t.variavel]
        cores = [DESTAQUE if v in cfg.CONJUNTO_PARCIMONIOSO else CLARO
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
    if cs.exists():
        t = pd.read_csv(cs, index_col=0).head(8).iloc[::-1]
        fig, ax = plt.subplots(figsize=(LARGURA, 3.8))
        y = np.arange(len(t))
        ax.barh(y + .2, t.com_creatinina, .4, color=CLARO, edgecolor="black",
                linewidth=.6, label="com creatinina")
        ax.barh(y - .2, t.sem_creatinina, .4, color=DESTAQUE, edgecolor="black",
                linewidth=.6, label="sem creatinina")
        ax.set_yticks(y)
        ax.set_yticklabels([NOMES_PT.get(v, v) for v in t.index])
        ax.set_xlabel("Proporção da explicação (SHAP)")
        _eixo_br(ax, "x", 2)
        # No grupo sem creatinina, a barra da creatinina e a contribuicao da
        # AUSENCIA dela; a legenda no texto precisa dizer isso.
        ax.legend(frameon=False, loc="lower right")
        _salvar(fig, destino, "fig_shap_com_sem_creatinina")


def fig_comorbidades(rel: Path, destino: Path) -> None:
    """Componente de comorbidades: diferenca por particao entre o modelo e a
    ordem fixa por prevalencia, nas quatro metricas de ordenacao."""
    arq = rel / "escopo2" / "metricas_por_particao.csv"
    if not arq.exists():
        print("  (comorbidades: metricas_por_particao.csv ausente -- pulada)")
        return
    m = pd.read_csv(arq)
    mets = [("map", "MAP"), ("ndcg", "NDCG"), ("recall_at_2", "Recall@2"),
            ("p_at_1", "Acerto no topo")][::-1]
    rng = np.random.default_rng(42)     # so o espalhamento vertical dos pontos
    fig, ax = plt.subplots(figsize=(LARGURA, 3.0))
    difs, rots = [], []
    for i, (col, _) in enumerate(mets):
        a = m[m.formulacao == "independente"].set_index("particao")[col]
        b = m[m.formulacao == "prevalencia"].set_index("particao")[col]
        d = (a - b).dropna().sort_index()
        difs.append(d)
        y = i + rng.uniform(-.15, .15, len(d))
        ax.scatter(d, y, s=16, color=DESTAQUE, edgecolor="black", linewidth=.4,
                   zorder=3)
        ax.plot([d.median()] * 2, [i - .28, i + .28], color="black", lw=2, zorder=4)
        fav, contra = int((d > 0).sum()), int((d < 0).sum())
        rots.append(f"a favor: {fav} de {len(d)}" + (f"; contra: {contra}" if contra else ""))
    ax.axvline(0, color=CINZA, lw=.8, ls="--")
    ax.set_yticks(range(len(mets)))
    ax.set_yticklabels([n for _, n in mets])
    ax.set_ylim(-.6, len(mets) - .4)
    dir_ = ax.twinx()                   # contagem por metrica, a direita
    dir_.set_ylim(ax.get_ylim())
    dir_.set_yticks(range(len(mets)))
    dir_.set_yticklabels(rots)
    dir_.tick_params(axis="y", length=0)
    dir_.spines[:].set_visible(False)
    ax.set_xlabel("Diferença por partição: modelo menos ordem por prevalência")
    # Marcas a cada 0,02: com o passo automatico (0,025) e duas casas, o
    # rotulo arredondava a posicao (0,025 aparecia como 0,03).
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    _eixo_br(ax, "x", 2)
    _salvar(fig, destino, "fig_comorbidades_particoes")


def fig_exames(rel: Path, destino: Path) -> None:
    """Componente de exames: ganho esperado do melhor exame no estado real,
    gatos com e sem a bioquimica feita."""
    arq = rel / "escopo3" / "recomendacoes_estado_real.csv"
    if not arq.exists():
        print("  (exames: recomendacoes_estado_real.csv ausente -- pulada)")
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
                    help="pasta relatorios da rodada (padrao: saida/relatorios)")
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
    fig_ausencia_por_estagio(destino)
    fig_tratamentos(rel, pastas, destino)
    fig_sem_creatinina(rel, pastas, destino)
    fig_matriz(rel, pastas["Nativo"], destino)
    fig_teto_rotulo(rel, pastas["Nativo"], destino)
    fig_shap(rel, destino)
    fig_comorbidades(rel, destino)
    fig_exames(rel, destino)
    print("\nAs curvas de sobrevida sao geradas por sobrevida_discordancias.py,")
    print("e a cascata de exclusao pela triagem.")


if __name__ == "__main__":
    main()
