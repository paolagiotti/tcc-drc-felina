"""
Auditoria das colunas do arquivo de origem.

Motivo de existir: uma coluna pode ser usada, descartada ou derivada de forma
errada sem que nada falhe. O codigo roda, os numeros saem, e o defeito so
aparece se alguem abrir a coluna e olhar. Foi o que aconteceu com a hipertensao
-- ela tem qualificador temporal em formato de data, o pipeline so sabe tratar
qualificador em texto, e ela foi descartada inteira sem nenhum aviso.

Este script inspeciona TODAS as colunas e emite alertas do que merece conferencia
humana. Nao decide nada: aponta.

Uso:
    python auditar_colunas.py
    python auditar_colunas.py --papel entrada     # so as variaveis de entrada
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

import config as cfg

SAIDA = cfg.SAIDA_RELATORIOS
QUALIFICADOR_TEXTO = re.compile(
    r"(?:before|at|after)\s+ckd\s+diagnosis", re.IGNORECASE)


def perfilar(s: pd.Series, nome: str, papel: str, rot: pd.Series) -> dict:
    """Perfil de uma coluna no subconjunto rotulado, com os alertas que ela dispara."""
    sr = s[rot]
    r = {"variavel": nome, "papel": papel,
         "ausente_pct": round(100 * sr.isna().mean(), 1),
         "distintos": int(sr.nunique(dropna=True)), "alertas": []}

    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        r["tipo"] = "numerica"
        v = sr.dropna()
        if len(v):
            r["min"], r["mediana"], r["max"] = (round(float(v.min()), 3),
                                                round(float(v.median()), 3),
                                                round(float(v.max()), 3))
            faixa = getattr(cfg, "FAIXAS_PLAUSIVEIS", {}).get(nome)
            if faixa:
                fora = int(((v < faixa[0]) | (v > faixa[1])).sum())
                if fora:
                    r["alertas"].append(f"{fora} valores fora da faixa {faixa}")
            else:
                # codigos-sentinela numericos disfarcados de medida
                for cod in (999, 9999, -1, -99):
                    n = int((v == cod).sum())
                    if n:
                        r["alertas"].append(
                            f"{n} valores iguais a {cod} e SEM faixa de "
                            f"plausibilidade configurada")
        r["exemplos"] = ""
    else:
        r["tipo"] = "categorica"
        v = sr.astype(str).str.strip().str.replace("\n", "", regex=False)
        top = v.value_counts().head(4)
        r["exemplos"] = "; ".join(f"{k}={n}" for k, n in top.items())[:70]
        sent = [x for x in v.unique() if x in set(cfg.SENTINELAS)]
        if sent:
            n = int(v.isin(sent).sum())
            r["alertas"].append(f"{n} registros em sentinela ({', '.join(sent)})")
        # rotulos ambiguos: "None recorded" convivendo com FALSE explicito
        vals = set(v.unique())
        if {"FALSE", "False"} & vals and {"None recorded", "Not recorded"} & vals:
            r["alertas"].append(
                "convivem FALSE e 'not/none recorded': ausencia de registro e "
                "negacao explicita podem significar a mesma coisa")
    return r


def achar_qualificadores(bruto: pd.DataFrame, papeis: dict) -> pd.DataFrame:
    """Colunas com qualificador temporal, classificadas por situacao.

    Dois formatos aparecem no arquivo: texto ("Before CKD diagnosis") e data
    ("Date hypertension diagnosed"). O pipeline trata o primeiro; o segundo
    passou despercebido ate esta auditoria, e foi assim que a hipertensao acabou
    descartada inteira em vez de derivada.

    A saida separa o que ja esta resolvido do que exige decisao, para que a lista
    seja curta o suficiente para ser lida.
    """
    tratados = {q for _, q in getattr(cfg, "PARES_TEMPORAIS", [])}
    linhas = []
    for c in bruto.columns:
        v = bruto[c].astype(str)
        tem_txt = bool(v.str.contains(QUALIFICADOR_TEXTO, na=False).any())
        tem_data = c.lower().startswith("date ") and c != cfg.COL_DATA_DIAGNOSTICO
        if not (tem_txt or tem_data):
            continue

        if c in tratados:
            sit = "ja derivado (_pre)"
        elif re.search(r"prescri|dose", c, re.I):
            sit = "terapia -- irrelevante para o estadiamento"
        elif papeis.get(c) == "derivada":
            sit = "marcado como derivada no dicionario"
        else:
            sit = "A CONFERIR"
        linhas.append({"qualificador": c,
                       "formato": "texto" if tem_txt else "data",
                       "papel_no_dicionario": papeis.get(c, "(fora do dicionario)"),
                       "situacao": sit})
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoria das colunas de origem.")
    ap.add_argument("--papel", default=None, help="filtrar por papel do dicionario")
    args = ap.parse_args()

    bruto = pd.read_csv(cfg.resolver_bruto(), low_memory=False)
    bruto.columns = [c.strip() for c in bruto.columns]
    dic = pd.read_csv(cfg.DICIONARIO)
    col_nome, col_papel = dic.columns[0], ("papel" if "papel" in dic.columns
                                           else dic.columns[1])
    papeis = dict(zip(dic[col_nome], dic[col_papel]))
    rot = bruto[cfg.COL_IRIS].notna()

    print(f"AUDITORIA DE COLUNAS — {len(bruto.columns)} colunas, "
          f"{int(rot.sum())} registros rotulados\n")

    linhas = []
    for c in bruto.columns:
        papel = papeis.get(c, "(fora do dicionario)")
        if args.papel and papel != args.papel:
            continue
        linhas.append(perfilar(bruto[c], c, papel, rot))
    t = pd.DataFrame(linhas)

    com_alerta = t[t.alertas.map(len) > 0]
    print(f"[1] colunas com alerta: {len(com_alerta)} de {len(t)}\n")
    for _, r in com_alerta.iterrows():
        marca = "!!" if r.papel == "entrada" else "  "
        print(f"{marca} {r.variavel[:44]:<44} [{r.papel}]")
        for a in r.alertas:
            print(f"      {a}")

    print("\n[2] qualificadores temporais NAO aproveitados")
    q = achar_qualificadores(bruto, papeis)
    if q.empty:
        print("  nenhum")
    else:
        resumo = q.situacao.value_counts()
        for sit, n in resumo.items():
            print(f"  {sit}: {n}")
        pend = q[q.situacao == "A CONFERIR"]
        if len(pend):
            print("\n  EXIGEM DECISAO:")
            print("  " + pend[["qualificador", "formato", "papel_no_dicionario"]]
                  .to_string(index=False).replace("\n", "\n  "))
            print("\n  Para cada uma: ou se deriva a versao anterior ao diagnostico,")
            print("  ou se declara por que nao. Silencio nao e decisao.")

    t["alertas"] = t.alertas.map(lambda a: " | ".join(a))
    t.to_csv(SAIDA / "auditoria_colunas.csv", index=False)
    if not q.empty:
        q.to_csv(SAIDA / "auditoria_qualificadores.csv", index=False)
    print(f"\nsaidas em {SAIDA}")


if __name__ == "__main__":
    main()
