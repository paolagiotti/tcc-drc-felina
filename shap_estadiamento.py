"""
Interpretabilidade por valores de Shapley (SHAP) para o componente de
estadiamento.

Duas saidas, com finalidades distintas:

  GLOBAL   -- quanto cada variavel pesa no conjunto da coorte. Serve ao texto,
              para mostrar que o modelo apoia-se em achados clinicamente
              reconheciveis.
  LOCAL    -- para cada gato, quanto cada variavel empurrou a predicao e em que
              direcao. E o insumo da camada interpretativa: o texto que o
              veterinario le nao diz "estagio 3", diz "estagio 3, porque ...".

Ponto de desenho que sustenta a validade: as atribuicoes sao calculadas FORA DA
AMOSTRA. Para cada particao, o modelo e ajustado no treino e explica apenas os
registros de teste, que ele nao viu. Explicar registros usados no proprio treino
produziria atribuicoes otimistas, pelo mesmo motivo que avaliar no treino produz
acuracia otimista.

Uso:
    python shap_estadiamento.py
    python shap_estadiamento.py --conjunto FULL --modelo floresta
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

import avaliacao as av
import config as cfg
import modelagem as mdl

warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")
SAIDA = cfg.SAIDA_RELATORIOS / "escopo1_shap"


def _agrupar(nomes: list[str], colunas: list[str]) -> list[str]:
    """Devolve, para cada coluna codificada, a variavel original de origem.

    O codificador de categorias transforma uma variavel em varias colunas; sem
    reagrupar, "Breed" apareceria fatiada em dezenas de contribuicoes pequenas.
    """
    saida = []
    for n in nomes:
        n = n.split("__", 1)[-1]
        cands = [c for c in colunas if n.startswith(c)]
        saida.append(max(cands, key=len) if cands else n)
    return saida


def calcular(df, y, colunas, nome_modelo, braco, tratamento, splits):
    """Atribuicoes SHAP fora da amostra, uma linha por registro."""
    import shap

    num, cat = mdl.separar_tipos(df, colunas)
    X = df[num + cat]
    contrib = None
    origem = None

    for i, (tr, te) in enumerate(splits, 1):
        est = mdl.estimadores(braco)[nome_modelo]
        pipe = mdl.construir_pipeline(est, num, cat, braco, y.iloc[tr], tratamento)
        pipe.fit(X.iloc[tr], y.iloc[tr])

        prep = pipe.named_steps["prep"]
        modelo = pipe.named_steps["clf"]
        modelo = getattr(modelo, "modelo_", modelo)   # desembrulha o XGBOrdinal
        X_te = prep.transform(X.iloc[te])

        # Os nomes sao recalculados a cada particao: o codificador de categorias
        # ve categorias diferentes em cada treino e produz numero diferente de
        # colunas. Reaproveitar os nomes de uma particao quebra na seguinte.
        origem = _agrupar(list(prep.get_feature_names_out()), colunas)
        if contrib is None:
            contrib = pd.DataFrame(0.0, index=X.index, columns=sorted(colunas))
            visto = pd.Series(0, index=X.index)

        valores = shap.TreeExplainer(modelo).shap_values(X_te)
        # multiclasse devolve uma matriz por classe; soma-se o modulo, porque
        # interessa o quanto a variavel moveu a decisao, nao para qual classe
        if isinstance(valores, list):
            mag = np.sum([np.abs(v) for v in valores], axis=0)
        elif valores.ndim == 3:
            mag = np.abs(valores).sum(axis=2)
        else:
            mag = np.abs(valores)

        bloco = pd.DataFrame(mag, columns=origem).T.groupby(level=0).sum().T
        bloco.index = X.index[te]
        for c in bloco.columns:
            if c in contrib.columns:
                contrib.loc[bloco.index, c] += bloco[c]
        visto.iloc[te] += 1
        print(f"  particao {i}/{len(splits)}", end="\r")

    print(" " * 30, end="\r")
    contrib = contrib.div(visto.replace(0, np.nan), axis=0)   # media entre repeticoes
    return contrib.dropna(how="all")


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP do componente de estadiamento.")
    ap.add_argument("--conjunto", default=None)
    ap.add_argument("--modelo", default="gradiente")
    ap.add_argument("--braco", default="nenhum")
    ap.add_argument("--ausentes", default="nativo")
    args = ap.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    df, y, conjuntos = mdl.carregar_coorte()
    conj = args.conjunto or getattr(cfg, "CONJUNTO_SISTEMA", "SEM_INDICADORES_EXAME")
    colunas = conjuntos[conj]
    av.definir_classes(y)

    print(f"[1] configuracao: {conj} | {args.modelo} | {args.braco} | "
          f"ausentes={args.ausentes}")
    print(f"    {len(df)} registros, {len(colunas)} variaveis")

    splits = av.materializar_splits(df, y)
    print(f"[2] calculando SHAP fora da amostra em {len(splits)} particoes")
    contrib = calcular(df, y, colunas, args.modelo, args.braco, args.ausentes, splits)

    print("[3] importancia global (media do valor absoluto por registro)")
    glob = contrib.mean().sort_values(ascending=False)
    glob_df = glob.rename("contribuicao_media").rename_axis("variavel").reset_index()
    glob_df["proporcao"] = (glob_df.contribuicao_media
                            / glob_df.contribuicao_media.sum()).round(4)
    glob_df.to_csv(SAIDA / "shap_global.csv", index=False)
    for _, r in glob_df.head(12).iterrows():
        print(f"    {r.variavel[:48]:<48} {r.proporcao:.3f}")

    lab = [c for c in glob.index
           if any(k in c for k in ("Creatinine", "BUN", "USG", "UPC",
                                   "Phosph", "Dipstick", "Blood pressure"))]
    cre = float(glob.get(cfg.COL_CREATININA, 0)) / glob.sum()
    outros = float(glob[[c for c in lab if c != cfg.COL_CREATININA]].sum()) / glob.sum()
    print(f"\n    creatinina                           {cre:.3f}")
    print(f"    demais exames laboratoriais          {outros:.3f}")
    print(f"    demografia, exame fisico, historico  {1 - cre - outros:.3f}")

    print("\n[4] contribuicoes por registro (insumo da camada interpretativa)")
    contrib.insert(0, "iris_registrado", y)
    if cfg.COL_ID in df.columns:
        contrib.insert(0, cfg.COL_ID, df[cfg.COL_ID])
    contrib.to_csv(SAIDA / "shap_por_registro.csv", index=False)

    print("[5] comparando gatos COM e SEM creatinina")
    tem = df[cfg.COL_CREATININA].notna()
    comp = pd.DataFrame({
        "com_creatinina": contrib.loc[tem].drop(columns=[cfg.COL_ID, "iris_registrado"],
                                                errors="ignore").mean(),
        "sem_creatinina": contrib.loc[~tem].drop(columns=[cfg.COL_ID, "iris_registrado"],
                                                 errors="ignore").mean(),
    })
    comp = comp.div(comp.sum(), axis=1)
    comp["diferenca"] = (comp.sem_creatinina - comp.com_creatinina).round(4)
    comp = comp.round(4).sort_values("sem_creatinina", ascending=False)
    comp.to_csv(SAIDA / "shap_com_sem_creatinina.csv")
    print("    proporcao da explicacao, por grupo (8 maiores sem creatinina):")
    print("    " + comp.head(8).to_string().replace("\n", "\n    "))
    print("\n    Responde o que o sistema usa quando o marcador definidor falta.")

    print(f"\nsaidas em {SAIDA}")


if __name__ == "__main__":
    main()
