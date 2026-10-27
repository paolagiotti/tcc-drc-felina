"""
Fichas dos pacientes: a entrada estruturada da camada interpretativa.

Para cada um dos 123 gatos, reune numa ficha unica o que os tres componentes
produziram:

  - estadiamento: estagio predito, probabilidade de cada estagio, e os cinco
    fatores que mais pesaram, COM DIRECAO (a favor ou contra o estagio predito)
    e com o valor que o gato tinha naquela variavel;
  - comorbidades: probabilidade de cada uma das quatro, em ordem;
  - exames: os que faltam, ordenados pela reducao esperada de duvida.

Tudo FORA DA AMOSTRA: cada gato e descrito por um modelo que nao o viu no
treino (primeira repeticao da validacao cruzada, cinco particoes).

O estagio registrado pelo veterinario fica na ficha apenas para a AVALIACAO, em
campo separado, e nunca e enviado ao modelo de linguagem -- no uso real ele nao
existe.

Uso:
    python fichas_pacientes.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import avaliacao as av
import config as cfg
import modelagem as mdl
from escopo3_recomendador import Recomendador, ordenar
from figuras import NOMES_PT
from shap_estadiamento import _agrupar

warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")
SAIDA = cfg.SAIDA_RELATORIOS / "camada"
TOP_FATORES = 5

# Traducao de valores categoricos e nomes para o texto em portugues. Feita aqui,
# e nao pelo modelo de linguagem, para que a verificacao de fidelidade compare
# exatamente os mesmos termos.
TRADUCAO = {
    "Clinical signs": "sinais clínicos", "Other": "outro",
    "Geriatric health clinic": "check-up geriátrico",
    "Pre-anasethetic profile": "perfil pré-anestésico",
    "Pre-anaesthetic profile": "perfil pré-anestésico",
    "Not recorded": "não registrado", "Not tested": "não testado",
    "None recorded": "nenhum registrado",
    "Female": "fêmea", "Male": "macho", "TRUE": "sim", "FALSE": "não",
    "True": "sim", "False": "não",
}
COMORB_PT = {
    "Periodontal disease": "doença periodontal", "Hyperthyroidism": "hipertireoidismo",
    "Cystitis/FLUTD": "cistite/DTUIF", "Arthritis": "artrite",
}

UNIDADES = {
    "Creatinine at diagnosis": "µmol/L", "BUN at diagnosis": "mmol/L",
    "Phosphorus at diagnosis": "mmol/L", "UPC at diagnosis": "",
    "USG at diagnosis": "", "Blood pressure at first measurement": "mmHg",
    "Age": "anos", "Dipstick protein at diagnosis": "",
}


def _valor(df, idx, var) -> str:
    """Valor da variavel do gato como texto em portugues, com unidade; 'ausente' se faltar."""
    v = df.loc[idx, var] if var in df.columns else np.nan
    if pd.isna(v):
        return "ausente"
    if isinstance(v, str) or isinstance(v, (bool, np.bool_)):
        return TRADUCAO.get(str(v).strip(), str(v).strip())
    if isinstance(v, (float, np.floating)):
        v = f"{v:.3f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.0f}"
    u = UNIDADES.get(var, "")
    return f"{v} {u}".strip()


def _shap_classe(explainer, X_te, k: int) -> np.ndarray:
    """Valores SHAP COM SINAL para a classe k (o estagio predito)."""
    v = explainer.shap_values(X_te)
    if isinstance(v, list):          # versoes antigas: uma matriz por classe
        return np.asarray(v[k])
    v = np.asarray(v)
    return v[..., k] if v.ndim == 3 else v


def main() -> None:
    import shap

    SAIDA.mkdir(parents=True, exist_ok=True)
    df, y, conj = mdl.carregar_coorte()
    cs = cfg.CONFIGURACAO_SISTEMA
    colunas = conj[cs["conjunto"]]
    num, cat = mdl.separar_tipos(df, colunas)
    colunas = num + cat
    exames = cfg.EXAMES_RECOMENDAVEIS
    tem = pd.DataFrame({e: df[vs].notna().any(axis=1) for e, vs in exames.items()})
    av.definir_classes(y)

    # comorbidades do escopo 2, se ja calculadas
    comorb = None
    arq2 = cfg.SAIDA_RELATORIOS / "escopo2" / "probabilidades_por_gato.csv"
    if arq2.exists() and cfg.COL_ID in df.columns:
        comorb = pd.read_csv(arq2).set_index(cfg.COL_ID)
    else:
        print("  AVISO: escopo 2 nao encontrado; fichas sem comorbidades. "
              "Rode antes: python escopo2_comorbidades.py")

    # primeira repeticao: cada gato aparece em teste exatamente uma vez
    splits = av.materializar_splits(df, y)[:cfg.N_FOLDS]
    print(f"[1] gerando fichas fora da amostra ({len(splits)} particoes, "
          f"sistema {cs['conjunto']} | {cs['modelo']} | {cs['braco']})")

    fichas = []
    for i, (tr, te) in enumerate(splits, 1):
        est = mdl.estimadores(cs["braco"])[cs["modelo"]]
        pipe = mdl.construir_pipeline(est, num, cat, cs["braco"], y.iloc[tr], "nativo")
        pipe.fit(df.iloc[tr][colunas], y.iloc[tr])
        rec = Recomendador(pipe, colunas, df.iloc[tr][colunas], y.iloc[tr], exames,
                           amostras=15, semente=i)

        prep = pipe.named_steps["prep"]
        modelo = getattr(pipe.named_steps["clf"], "modelo_", pipe.named_steps["clf"])
        explainer = shap.TreeExplainer(modelo)
        origem = _agrupar(list(prep.get_feature_names_out()), colunas)
        X_te = prep.transform(df.iloc[te][colunas])
        probs = rec.prob(df.iloc[te][colunas])

        for j, idx in enumerate(df.index[te]):
            p = probs[j]
            k = int(np.argmax(p))
            sv = _shap_classe(explainer, X_te[j:j + 1], k)[0]
            contrib = pd.Series(sv, index=origem).groupby(level=0).sum()
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
            fatores = [{
                "variavel": NOMES_PT.get(v, v),
                "valor": _valor(df, idx, v),
                "direcao": "a favor" if c > 0 else "contra",
                "contribuicao": round(float(c), 3),
            } for v, c in top.head(TOP_FATORES).items()]

            faltam = [e for e in exames if not tem.loc[idx, e]]
            recomend = []
            if faltam:
                g = rec.ganho_esperado(df.loc[idx, colunas], faltam)
                # ganho <= 0: a aproximacao por valores amostrados nao garante
                # ganho positivo (limitacao registrada no escopo 3). Esses exames
                # sao marcados para NAO serem recomendados.
                recomend = [{"exame": e, "ganho_bits": round(float(g[e]), 3),
                             "reduz_duvida": bool(g[e] > 0)}
                            for e in ordenar(g, False)]

            comorbs = []
            gid = df.loc[idx, cfg.COL_ID] if cfg.COL_ID in df.columns else None
            if comorb is not None and gid in comorb.index:
                linha = comorb.loc[gid]
                nomes_c = [c for c in linha.index if c != "ranking"]
                comorbs = sorted(({"nome": COMORB_PT.get(c, c),
                                   "probabilidade": round(float(linha[c]), 3)}
                                  for c in nomes_c), key=lambda d: -d["probabilidade"])

            fichas.append({
                "id": None if gid is None else (int(gid) if str(gid).isdigit() else str(gid)),
                "_avaliacao": {"estagio_registrado": int(y[idx]),
                               "sem_creatinina": bool(pd.isna(df.loc[idx, cfg.COL_CREATININA]))},
                "idade": _valor(df, idx, "Age"),
                "sexo": _valor(df, idx, "Sex"),
                "estagio_predito": k + 1,
                "probabilidades": {str(c + 1): round(float(p[c]), 3) for c in range(4)},
                "fatores": fatores,
                "exames_disponiveis": [e for e in exames if tem.loc[idx, e]],
                "exames_faltantes": faltam,
                "recomendacao_exames": recomend,
                "comorbidades": comorbs,
            })
        print(f"  particao {i}/{len(splits)}", end="\r")
    print(" " * 30, end="\r")

    fichas.sort(key=lambda f: str(f["id"]))
    with open(SAIDA / "fichas.jsonl", "w", encoding="utf-8") as fh:
        for f in fichas:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    acerto = np.mean([f["estagio_predito"] == f["_avaliacao"]["estagio_registrado"]
                      for f in fichas])
    print(f"[2] {len(fichas)} fichas gravadas em {SAIDA / 'fichas.jsonl'}")
    print(f"    concordancia com o registro (conferencia): {acerto:.1%}")
    print(f"    com comorbidades: {sum(bool(f['comorbidades']) for f in fichas)}")
    print(f"    com exame recomendado: {sum(bool(f['recomendacao_exames']) for f in fichas)}")
    print("\n    exemplo:")
    ex = next(f for f in fichas if f["recomendacao_exames"] and f["comorbidades"])
    print("    " + json.dumps({k: v for k, v in ex.items() if k != "_avaliacao"},
                              ensure_ascii=False, indent=2).replace("\n", "\n    "))


if __name__ == "__main__":
    main()
