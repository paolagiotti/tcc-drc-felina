"""
Inventario do arquivo de origem: congela o dado bruto e gera o rascunho do
dicionario de variaveis.

Este script NAO altera nada e NAO decide nada. Ele:
  1. registra o hash do arquivo bruto, para garantir que ele nao mudou;
  2. perfila as 121 colunas (tipo, cardinalidade, ausencia, exemplos);
  3. tenta sugerir o papel de cada coluna por heuristica de nome;
  4. grava `dados/dicionario_variaveis.csv` para preenchimento manual.

A coluna `papel` da saida e uma SUGESTAO. Ela precisa ser revisada uma a uma
antes de rodar `triagem.py` -- e a revisao e onde voce identifica o vazamento
estrutural, que nenhuma heuristica pega.

Uso:
    python inventario.py

---
    lê o Survival_data.csv, registra o SHA-256, perfila as 121 colunas (tipo inferido, cobertura, cardinalidade) e sugere um papel por 
    heurística de nome (id, alvo_iris, desfecho, terapia, alvo_comorbidade, excluir; na dúvida, entrada).
    Grava o rascunho de dados/dicionario_variaveis.csv. A coluna papel foi revisada à mão depois: é na revisão que se identifica vazamento estrutural (
    subestadiamento calculado, terapia posterior, desfecho recodificado), que nenhuma heurística pega.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime

import pandas as pd

import config as cfg


# ---------------------------------------------------------------------------
# Passo 1: congelar o bruto
# ---------------------------------------------------------------------------

def hash_arquivo(caminho, bloco: int = 1 << 20) -> str:
    """SHA-256 do arquivo, lido em blocos."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        while pedaco := f.read(bloco):
            h.update(pedaco)
    return h.hexdigest()


def carregar_bruto(caminho=None) -> pd.DataFrame:
    """Le o CSV de origem com os valores ausentes de config e limpa os textos."""
    caminho = caminho or cfg.resolver_bruto()
    try:
        df = pd.read_csv(
            caminho, sep=cfg.CSV_SEP, encoding=cfg.CSV_ENCODING,
            decimal=cfg.CSV_DECIMAL, na_values=cfg.VALORES_AUSENTES,
            keep_default_na=True, low_memory=False,
        )
    except UnicodeDecodeError:
        print("  aviso: falhou em utf-8, relendo em latin-1")
        df = pd.read_csv(
            caminho, sep=cfg.CSV_SEP, encoding="latin-1",
            decimal=cfg.CSV_DECIMAL, na_values=cfg.VALORES_AUSENTES,
            keep_default_na=True, low_memory=False,
        )
    # Valores de texto vem com quebra de linha embutida neste arquivo.
    for col in df.columns[df.dtypes == "object"]:
        df[col] = df[col].str.strip()
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Passo 2: perfilar e sugerir papel
# ---------------------------------------------------------------------------

# Heuristicas de nome. Deliberadamente conservadoras: na duvida, sugerem
# "entrada", para que a revisao manual tenha de olhar a coluna.
REGRAS_PAPEL: list[tuple[str, str]] = [
    (r"\b(id|ident|patient|animal|case|record)[_\s]?(id|no|num)?\b", "id"),
    (r"iris.*(stage|estad)|stage.*iris", "alvo_iris"),
    (r"surviv|death|died|dead|outcome|euthan|follow[_\s]?up|time[_\s]?to",
     "desfecho"),
    (r"treat|therap|medicat|drug|diet|benazepr|amlodipin|phosphate[_\s]binder",
     "terapia"),
    (r"hyperthyroid|periodont|dental|cystitis|flutd|arthrit|diabet|"
     r"cardiomyopath|neoplas|hypertens.*diagnos|comorbid",
     "alvo_comorbidade"),
    (r"comment|note|free[_\s]?text|description|remark", "excluir"),
]


def sugerir_papel(nome: str, serie: pd.Series) -> str:
    """Papel sugerido pelo nome da coluna; 'entrada' na duvida, 'excluir' se vazia."""
    baixo = nome.strip().lower()
    for padrao, papel in REGRAS_PAPEL:
        if re.search(padrao, baixo):
            return papel
    if serie.notna().sum() == 0:
        return "excluir"
    return "entrada"


def inferir_tipo(serie: pd.Series) -> str:
    """Tipo da coluna: binaria, ordinal ou categorica, numerica, data, texto, vazia."""
    limpa = serie.dropna()
    if limpa.empty:
        return "vazia"
    if pd.api.types.is_numeric_dtype(limpa):
        if set(limpa.unique()) <= {0, 1}:
            return "binaria"
        if (limpa == limpa.round()).all() and limpa.nunique() <= 12:
            return "ordinal_ou_categorica"
        return "numerica"
    convertida = pd.to_numeric(limpa, errors="coerce")
    if convertida.notna().mean() > 0.90:
        return "numerica_como_texto"
    if pd.to_datetime(limpa, errors="coerce", format="mixed").notna().mean() > 0.80:
        return "data"
    if limpa.nunique() <= 15:
        return "categorica"
    return "texto"


def exemplos(serie: pd.Series, n: int = 4) -> str:
    """Os primeiros n valores distintos, para a revisao manual."""
    vals = serie.dropna().unique()[:n]
    return " | ".join(str(v)[:28] for v in vals)


def perfilar(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por coluna: tipo, cobertura, cardinalidade e papel sugerido."""
    linhas = []
    for col in df.columns:
        s = df[col]
        linhas.append({
            "coluna": col,
            "tipo_inferido": inferir_tipo(s),
            "n_preenchidos": int(s.notna().sum()),
            "pct_ausente": round(float(s.isna().mean()) * 100, 1),
            "n_unicos": int(s.nunique(dropna=True)),
            "exemplos": exemplos(s),
            "papel_sugerido": sugerir_papel(col, s),
            "papel": sugerir_papel(col, s),   # <- editar esta coluna a mao
            "unidade": "",
            "notas": "",
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Perfila o CSV bruto e gera o dicionario.")
    ap.add_argument("--bruto", default=None,
                    help="caminho do Survival_data.csv (sobrescreve a busca automatica)")
    args = ap.parse_args()

    cfg.SAIDA_RELATORIOS.mkdir(parents=True, exist_ok=True)
    cfg.DICIONARIO.parent.mkdir(parents=True, exist_ok=True)

    bruto = cfg.resolver_bruto(args.bruto)
    print(f"lendo {bruto}")
    sha = hash_arquivo(bruto)
    df = carregar_bruto(bruto)
    print(f"  {df.shape[0]} registros x {df.shape[1]} colunas")
    print(f"  sha256: {sha}")

    procedencia = {
        "arquivo": str(bruto),
        "sha256": sha,
        "n_registros": int(df.shape[0]),
        "n_colunas": int(df.shape[1]),
        "lido_em": datetime.now().isoformat(timespec="seconds"),
    }
    (cfg.SAIDA_RELATORIOS / "procedencia_bruto.json").write_text(
        json.dumps(procedencia, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if cfg.DICIONARIO.exists():
        destino = cfg.DICIONARIO.with_suffix(".novo.csv")
        print(f"\n  dicionario ja existe; gravando sugestao em {destino.name}")
        print("  (nao sobrescrevo o seu trabalho manual)")
    else:
        destino = cfg.DICIONARIO

    dic = perfilar(df)
    dic.to_csv(destino, index=False, encoding="utf-8")

    print(f"\ndicionario gravado em {destino}")
    print("\ndistribuicao dos papeis sugeridos:")
    for papel, n in dic["papel_sugerido"].value_counts().items():
        print(f"  {papel:>18}: {n}")

    vazias = dic.loc[dic["tipo_inferido"] == "vazia", "coluna"].tolist()
    if vazias:
        print(f"\ncolunas inteiramente vazias ({len(vazias)}): {', '.join(vazias[:10])}")

    quase = dic.loc[dic["pct_ausente"] >= cfg.PROP_MAXIMA_AUSENTE_COLUNA * 100]
    print(f"\ncolunas com ausencia >= {cfg.PROP_MAXIMA_AUSENTE_COLUNA:.0%}: {len(quase)}")

    print(
        "\nPROXIMO PASSO (manual, nao tem atalho):\n"
        f"  1. abrir {destino.name} e revisar a coluna `papel` linha a linha;\n"
        "  2. marcar como `excluir` tudo que for consequencia do diagnostico\n"
        "     (substadiamento calculado, terapia posterior, desfecho recodificado);\n"
        "  3. preencher COL_IRIS, COL_CREATININA, COL_UPC, COL_PAS e as faixas\n"
        "     fisiologicas em config.py com os nomes reais das colunas;\n"
        "  4. rodar: python triagem.py"
    )


if __name__ == "__main__":
    main()
