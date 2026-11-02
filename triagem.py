"""
Triagem da coorte: do arquivo de origem as coortes do estadiamento (123 gatos)
e das comorbidades (618 gatos).

Converte tipos e unidades, deriva os achados anteriores ao diagnostico, aplica
as faixas de plausibilidade e os criterios de inclusao (contando a cascata),
levanta os mapas de ausencia, congela os conjuntos de variaveis e grava a
coorte limpa e os relatorios. Nao imputa, nao codifica e nao olha a relacao
entre variavel e alvo: isso pertence ao pipeline, dentro de cada particao.

Uso:
    python triagem.py


    ----


    do arquivo bruto às duas coortes
    Valida o dicionário; converte números gravados como texto e as datas (explicitamente, dia primeiro, porque comparar datas como texto dá ordem alfabética); 
    deriva os achados _pre (verdadeiro só se anterior ou simultâneo ao diagnóstico; conta os positivos posteriores descartados); 
    aplica as faixas de plausibilidade; 
    compara os 124 com estágio contra os 501 sem estágio (validade externa; o desfecho fica fora da tabela, por ser consequência); 
    aplica a cascata de inclusão (estágio presente → datas consistentes → diagnóstico na janela → sem injúria aguda → ao menos metade das entradas preenchidas) contando cada corte; 
    monta os mapas de ausência por variável e a cobertura informativa descontando sentinelas; 
    aplica as decisões de entrada (troca o achado cru pelo _pre, descarta as seis de conteúdo insuficiente); 
    cria os indicadores _ausente do conjunto de controle SO_AUSENCIAS; 
    congela os conjuntos de variáveis da ablação (FULL, SEM_CREATININA, SEM_SUBESTADIAMENTO, SEM_INDICADORES_EXAME etc.); 
    Grava coorte.parquet (123), coorte_comorbidades.parquet (618, mesmos critérios sem exigir estágio), conjuntos_variaveis.json e as tabelas.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import re

import pandas as pd

import config as cfg
from inventario import carregar_bruto, hash_arquivo


# ---------------------------------------------------------------------------
# Cascata
# ---------------------------------------------------------------------------

class Cascata:
    """Registra quantos registros sobram apos cada criterio."""

    def __init__(self, n_inicial: int, rotulo: str = "Registros brutos"):
        self.linhas = [{"criterio": rotulo, "removidos": 0, "restantes": n_inicial}]

    def registrar(self, criterio: str, antes: int, depois: int) -> None:
        self.linhas.append({
            "criterio": criterio,
            "removidos": antes - depois,
            "restantes": depois,
        })

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame(self.linhas)

    def imprimir(self) -> None:
        print("\n--- cascata de exclusao ---")
        print(f"  {'criterio':<46}{'removidos':>11}{'restantes':>11}")
        for i, l in enumerate(self.linhas):
            rem = "" if i == 0 else f"-{l['removidos']}"
            print(f"  {l['criterio']:<46}{rem:>11}{l['restantes']:>11}")


# ---------------------------------------------------------------------------
# Dicionario de variaveis
# ---------------------------------------------------------------------------

def carregar_dicionario(df: pd.DataFrame) -> pd.DataFrame:
    """Le o dicionario de variaveis e confere papeis e nomes de coluna contra o CSV."""
    if not cfg.DICIONARIO.exists():
        sys.exit(
            f"Dicionario nao encontrado em {cfg.DICIONARIO}\n"
            f"Rode antes: python inventario.py"
        )
    dic = pd.read_csv(cfg.DICIONARIO, encoding="utf-8")

    if "papel" not in dic.columns or "coluna" not in dic.columns:
        sys.exit("Dicionario precisa ter as colunas `coluna` e `papel`.")

    invalidos = set(dic["papel"].dropna()) - cfg.PAPEIS_VALIDOS
    if invalidos:
        sys.exit(f"Papeis invalidos no dicionario: {sorted(invalidos)}\n"
                 f"Validos: {sorted(cfg.PAPEIS_VALIDOS)}")

    faltando = set(dic["coluna"]) - set(df.columns)
    if faltando:
        sys.exit(f"Dicionario cita colunas que nao existem no CSV: {sorted(faltando)[:5]}")

    sobrando = set(df.columns) - set(dic["coluna"])
    if sobrando:
        print(f"  aviso: {len(sobrando)} colunas do CSV nao estao no dicionario "
              f"e serao ignoradas")

    if (dic["papel"] == dic.get("papel_sugerido")).all():
        print("\n  ATENCAO: nenhum papel foi editado manualmente no dicionario.\n"
              "  A heuristica nao identifica vazamento estrutural. Revise antes\n"
              "  de confiar em qualquer resultado que saia daqui.\n")

    return dic


def colunas_por_papel(dic: pd.DataFrame, papel: str) -> list[str]:
    """Colunas do dicionario com o papel dado."""
    return dic.loc[dic["papel"] == papel, "coluna"].tolist()


# ---------------------------------------------------------------------------
# Passo 4: unidades e tipos
# ---------------------------------------------------------------------------

def converter_tipos(df: pd.DataFrame, dic: pd.DataFrame) -> pd.DataFrame:
    """Numeros gravados como texto viram numero; datas sao convertidas explicitamente."""
    df = df.copy()
    convertidas = []
    for _, linha in dic.iterrows():
        col, tipo = linha["coluna"], linha.get("tipo_inferido")
        if col not in df.columns:
            continue
        if tipo == "numerica_como_texto":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            convertidas.append(col)
    if convertidas:
        print(f"  convertidas de texto para numero: {len(convertidas)} colunas")

    # Datas: conversao explicita, nunca por inferencia de tipo.
    datas = []
    for col in cfg.COLUNAS_DATA:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=cfg.DIA_PRIMEIRO,
                                     errors="coerce")
            datas.append(col)
    if datas:
        print(f"  convertidas para data ({'dia primeiro' if cfg.DIA_PRIMEIRO else 'mes primeiro'}): "
              f"{len(datas)} colunas")
    return df


def derivar_achados_pre_diagnostico(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria `<achado>_pre`: verdadeiro so quando o achado e anterior ou
    simultaneo ao diagnostico (o booleano cru vaza informacao futura)."""
    df = df.copy()
    linhas = []
    for achado, quando in cfg.PARES_TEMPORAIS:
        if achado not in df.columns or quando not in df.columns:
            print(f"  aviso: par temporal ignorado, coluna ausente: {achado} / {quando}")
            continue
        pos = df[achado].astype(str).str.strip().str.upper().isin(["TRUE", "T"])
        qual = df[quando].astype(str).str.strip()
        aceito = qual.isin(cfg.TEMPORAL_ACEITO)
        novo = f"{achado}_pre"
        df[novo] = (pos & aceito)
        linhas.append({
            "achado": achado,
            "positivos_brutos": int(pos.sum()),
            "posteriores_ao_diagnostico": int((pos & ~aceito).sum()),
            "positivos_retidos": int(df[novo].sum()),
            "coluna_derivada": novo,
        })
    tabela = pd.DataFrame(linhas)
    if not tabela.empty:
        vazados = int(tabela["posteriores_ao_diagnostico"].sum())
        print(f"  achados derivados: {len(tabela)} | positivos descartados por "
              f"serem posteriores ao diagnostico: {vazados}")
    return df, tabela


def normalizar_creatinina(df: pd.DataFrame) -> pd.DataFrame:
    """Garante creatinina em umol/L, que e a unidade dos limiares IRIS usados."""
    if cfg.COL_CREATININA is None or cfg.COL_CREATININA not in df.columns:
        return df
    df = df.copy()
    col = cfg.COL_CREATININA
    if not cfg.CREATININA_EM_UMOL:
        df[col] = df[col] * cfg.UMOL_POR_MGDL
        print(f"  creatinina convertida de mg/dL para umol/L (x{cfg.UMOL_POR_MGDL})")
    mediana = df[col].median()
    if pd.notna(mediana) and mediana < 30:
        print(f"  AVISO: mediana da creatinina = {mediana:.2f}. Valor tipico de "
              f"mg/dL, nao de umol/L. Confira cfg.CREATININA_EM_UMOL.")
    return df


# ---------------------------------------------------------------------------
# Passo 5: plausibilidade fisiologica
# ---------------------------------------------------------------------------

def aplicar_faixas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valores fora das faixas de plausibilidade viram ausentes; devolve a contagem por coluna."""
    df = df.copy()
    registro = []
    for col, (minimo, maximo) in cfg.FAIXAS_PLAUSIVEIS.items():
        if col not in df.columns:
            print(f"  aviso: faixa definida para coluna inexistente: {col}")
            continue
        fora = (df[col] < minimo) | (df[col] > maximo)
        n = int(fora.sum())
        if n:
            df.loc[fora, col] = np.nan
        registro.append({
            "coluna": col, "minimo": minimo, "maximo": maximo,
            "n_convertidos_para_ausente": n,
        })
    tabela = pd.DataFrame(registro)
    total = int(tabela["n_convertidos_para_ausente"].sum()) if not tabela.empty else 0
    print(f"  valores implausiveis convertidos para ausente: {total}")
    if not cfg.FAIXAS_PLAUSIVEIS:
        print("  aviso: nenhuma faixa fisiologica definida em config.FAIXAS_PLAUSIVEIS")
    return df, tabela


# ---------------------------------------------------------------------------
# Passo 6: inclusao e exclusao
# ---------------------------------------------------------------------------

def datas_inconsistentes(df: pd.DataFrame) -> pd.Series:
    """Registros em que a data inicial e posterior a final. Exige colunas
    datetime: comparadas como texto, as datas sairiam em ordem alfabetica."""
    ruim = pd.Series(False, index=df.index)
    for inicio, fim in cfg.PARES_DATAS:
        if inicio not in df.columns or fim not in df.columns:
            print(f"  aviso: par de datas ignorado, coluna ausente: {inicio} / {fim}")
            continue
        for c in (inicio, fim):
            if not pd.api.types.is_datetime64_any_dtype(df[c]):
                raise TypeError(
                    f"A coluna '{c}' nao e datetime ({df[c].dtype}). "
                    f"Acrescente-a a cfg.COLUNAS_DATA antes de usa-la em "
                    f"cfg.PARES_DATAS."
                )
        ruim |= (df[inicio] > df[fim]).fillna(False)
    return ruim


def diagnostico_fora_da_janela(df: pd.DataFrame) -> pd.Series:
    """Data de diagnostico fora do periodo de recrutamento do estudo."""
    col = cfg.COL_DATA_DIAGNOSTICO
    janela = cfg.JANELA_DIAGNOSTICO
    if not col or not janela or col not in df.columns:
        return pd.Series(False, index=df.index)
    d = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    inicio, fim = pd.Timestamp(janela[0]), pd.Timestamp(janela[1])
    return ((d < inicio) | (d > fim)).fillna(False)


def marcar_ira(df: pd.DataFrame) -> pd.Series:
    """Registros com injuria renal aguda associada (config.COL_IRA)."""
    if cfg.COL_IRA is None or cfg.COL_IRA not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[cfg.COL_IRA]
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(bool)
    padrao = "|".join(cfg.TERMOS_IRA)
    return s.astype(str).str.lower().str.contains(padrao, na=False)


def aplicar_criterios(
    df: pd.DataFrame, entradas: list[str], exigir_iris: bool | None = None
) -> tuple[pd.DataFrame, Cascata]:
    """Criterios de inclusao das duas coortes; `exigir_iris=False` produz a
    coorte das comorbidades, com os demais criterios identicos."""
    casc = Cascata(len(df))
    exigir_iris = cfg.EXIGIR_IRIS if exigir_iris is None else exigir_iris

    if exigir_iris:
        if cfg.COL_IRIS is None:
            sys.exit("cfg.COL_IRIS precisa ser preenchido para exigir estagio IRIS.")
        antes = len(df)
        df = df[df[cfg.COL_IRIS].notna()]
        casc.registrar("sem estagio IRIS no diagnostico", antes, len(df))

    if not cfg.PARES_DATAS:
        print("  AVISO: cfg.PARES_DATAS esta vazio. O criterio de datas "
              "inconsistentes nao testa nada e removera zero por construcao.")
    antes = len(df)
    df = df[~datas_inconsistentes(df)]
    casc.registrar("datas inconsistentes", antes, len(df))

    antes = len(df)
    df = df[~diagnostico_fora_da_janela(df)]
    casc.registrar("diagnostico fora da janela do estudo", antes, len(df))

    antes = len(df)
    df = df[~marcar_ira(df)]
    casc.registrar("injuria renal aguda associada", antes, len(df))

    presentes = [c for c in entradas if c in df.columns]
    proporcao = df[presentes].notna().mean(axis=1)
    antes = len(df)
    df = df[proporcao >= cfg.PROP_MINIMA_PREENCHIDA]
    casc.registrar(
        f"menos de {cfg.PROP_MINIMA_PREENCHIDA:.0%} das entradas preenchidas",
        antes, len(df),
    )

    return df, casc


# ---------------------------------------------------------------------------
# Passo 7: mapas de ausencia
# ---------------------------------------------------------------------------

def ausencia_por_variavel(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cobertura de cada coluna, da mais ausente para a menos."""
    cols = [c for c in cols if c in df.columns]
    return (
        pd.DataFrame({
            "coluna": cols,
            "n_preenchidos": [int(df[c].notna().sum()) for c in cols],
            "pct_ausente": [round(float(df[c].isna().mean()) * 100, 1) for c in cols],
        })
        .sort_values("pct_ausente", ascending=False)
        .reset_index(drop=True)
    )


def cobertura_informativa(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cobertura descontando as sentinelas textuais ("Not recorded" nao e dado
    presente)."""
    sent = set(cfg.SENTINELAS)
    linhas = []
    for c in [c for c in cols if c in df.columns]:
        s = df[c]
        ausente = int(s.isna().sum())
        if pd.api.types.is_numeric_dtype(s):
            n_sent = 0
        else:
            n_sent = int(s.astype(str).str.strip().isin(sent).sum())
        informativo = len(df) - ausente - n_sent
        linhas.append({
            "coluna": c,
            "n_ausente": ausente,
            "n_sentinela": n_sent,
            "n_informativo": informativo,
            "pct_informativo": round(100 * informativo / len(df), 1),
        })
    return (pd.DataFrame(linhas)
            .sort_values("pct_informativo")
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Passo 9: conjuntos de variaveis
# ---------------------------------------------------------------------------

def aplicar_decisoes_de_entrada(entradas: list[str], df: pd.DataFrame) -> list[str]:
    """Troca o achado cru pela derivada `_pre` (a unica versao disponivel no
    diagnostico) e remove as variaveis de conteudo informativo insuficiente."""
    entradas = list(entradas)

    trocas = []
    for achado, _ in cfg.PARES_TEMPORAIS:
        derivada = f"{achado}_pre"
        if achado in entradas and derivada in df.columns:
            entradas[entradas.index(achado)] = derivada
            trocas.append((achado, derivada))
    for cru, der in trocas:
        print(f"    {cru}  ->  {der}")
    if trocas:
        print(f"  achados substituidos pela versao anterior ao diagnostico: {len(trocas)}")

    descartar = [c for c in cfg.DESCARTAR_POR_BAIXO_CONTEUDO if c in entradas]
    if descartar:
        entradas = [c for c in entradas if c not in descartar]
        print(f"  descartadas por conteudo insuficiente: {len(descartar)}")
        for c in descartar:
            print(f"    {c}")

    return entradas


def derivar_indicadores_ausencia(df: pd.DataFrame, entradas: list[str]
                                 ) -> tuple[pd.DataFrame, list[str]]:
    """Cria `<variavel>_ausente` para cada entrada com ausencia parcial: o
    controle SO_AUSENCIAS, que nao entra nos demais conjuntos."""
    criadas = []
    for c in entradas:
        if c not in df.columns:
            continue
        falta = df[c].isna()
        if 0 < falta.mean() < 1:
            nome = f"{c}_ausente"
            df[nome] = falta
            criadas.append(nome)
    return df, criadas


# Desfecho e terapia sao posteriores ao diagnostico: ficam fora da comparacao
# entre registros com e sem estagio.
PAPEIS_FORA_DA_COMPARACAO = {"id", "desfecho", "terapia"}


def comparar_rotulado_vs_nao_rotulado(bruto: pd.DataFrame, dic: pd.DataFrame
                                      ) -> pd.DataFrame:
    """Compara os registros com estagio IRIS (124) com os sem estagio (501):
    questao de validade externa."""
    tem = bruto[cfg.COL_IRIS].notna()
    col_nome = dic.columns[0]
    col_papel = "papel" if "papel" in dic.columns else dic.columns[1]
    papeis = dict(zip(dic[col_nome], dic[col_papel]))
    linhas = []
    for col in bruto.columns:
        if col == cfg.COL_IRIS or bruto[col].isna().all():
            continue
        papel = papeis.get(col, "(fora do dicionario)")
        if papel in PAPEIS_FORA_DA_COMPARACAO:
            continue
        if re.search(r"^date |prescri|dose", col, re.I):
            continue
        s = bruto[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            a, b = s[tem].dropna(), s[~tem].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            from scipy.stats import mannwhitneyu
            try:
                p = mannwhitneyu(a, b).pvalue
            except ValueError:
                continue
            linhas.append({"variavel": col, "papel": papel, "tipo": "numerica",
                           "com_estagio": round(a.median(), 2),
                           "sem_estagio": round(b.median(), 2),
                           "medida": "mediana", "p": round(p, 4),
                           "n_com": len(a), "n_sem": len(b)})
        else:
            # proporcao de nao-ausentes, ou de positivos se for booleana
            if pd.api.types.is_bool_dtype(s):
                a, b = s[tem].mean(), s[~tem].mean()
                medida = "proporcao positiva"
            else:
                a, b = s[tem].notna().mean(), s[~tem].notna().mean()
                medida = "proporcao preenchida"
            from scipy.stats import chi2_contingency
            tab = pd.crosstab(tem, s.notna() if medida.startswith("proporcao p")
                              else s.fillna(False).astype(bool))
            try:
                p = chi2_contingency(tab).pvalue if tab.shape == (2, 2) else float("nan")
            except ValueError:
                p = float("nan")
            linhas.append({"variavel": col, "papel": papel, "tipo": "categorica",
                           "com_estagio": round(float(a), 3),
                           "sem_estagio": round(float(b), 3),
                           "medida": medida,
                           "p": round(p, 4) if p == p else None,
                           "n_com": int(tem.sum()), "n_sem": int((~tem).sum())})
    out = pd.DataFrame(linhas)
    return out.sort_values("p", na_position="last")


def montar_conjuntos(entradas: list[str]) -> dict[str, list[str]]:
    """Conjuntos de variaveis da ablacao. SEM_CREATININA remove o determinante
    do estagio; SEM_SUBESTADIAMENTO remove UPC e pressao, que definem o
    subestagio; SEM_INDICADORES_EXAME e o conjunto do sistema."""
    definidor = [c for c in (cfg.COL_CREATININA,) if c and c in entradas]
    subestadiamento = [c for c in (cfg.COL_UPC, cfg.COL_PAS) if c and c in entradas]

    conjuntos = {
        "FULL": list(entradas),
        "DEFINIDOR_ESTAGIO": definidor,
        "SUBESTADIAMENTO": subestadiamento,
        "SEM_CREATININA": [c for c in entradas if c not in definidor],
        "SEM_SUBESTADIAMENTO": [c for c in entradas if c not in subestadiamento],
        "SEM_CREATININA_NEM_SUBESTADIAMENTO": [
            c for c in entradas if c not in definidor + subestadiamento],
        "SEM_INDICADORES_EXAME": [
            c for c in entradas
            if c not in cfg.INDICADORES_EXAME_REALIZADO],
    }
    for nome, cols in conjuntos.items():
        print(f"  {nome:<34} {len(cols):>3} variaveis")
    if not definidor:
        print("  AVISO: creatinina ausente do conjunto de entrada. A ablacao "
              "central do escopo 1 depende dela.")
    return conjuntos


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Triagem da coorte (etapa 1).")
    ap.add_argument("--bruto", default=None, help="caminho do Survival_data.csv")
    args = ap.parse_args()

    for pasta in (cfg.SAIDA_DADOS, cfg.SAIDA_RELATORIOS, cfg.SAIDA_FIGURAS):
        pasta.mkdir(parents=True, exist_ok=True)

    caminho = cfg.resolver_bruto(args.bruto)
    print(f"[1] carregando bruto: {caminho}")
    bruto = carregar_bruto(caminho)
    sha = hash_arquivo(caminho)
    print(f"  {bruto.shape[0]} registros x {bruto.shape[1]} colunas")

    print("[2] carregando dicionario de variaveis")
    dic = carregar_dicionario(bruto)
    entradas = colunas_por_papel(dic, "entrada")
    comorbidades = colunas_por_papel(dic, "alvo_comorbidade")
    desfechos = colunas_por_papel(dic, "desfecho")
    print(f"  entradas: {len(entradas)} | comorbidades: {len(comorbidades)} "
          f"| desfechos: {len(desfechos)}")

    print("[3] convertendo tipos e unidades")
    df = converter_tipos(bruto, dic)
    df = normalizar_creatinina(df)

    print("[4] derivando achados anteriores ao diagnostico")
    df, tab_temporal = derivar_achados_pre_diagnostico(df)

    print("[5] aplicando faixas de plausibilidade")
    df, tab_faixas = aplicar_faixas(df)

    print("[6] comparando registros com e sem estagio IRIS registrado")
    comp = comparar_rotulado_vs_nao_rotulado(df, dic)
    n_com = int(df[cfg.COL_IRIS].notna().sum())
    n_sem = int(df[cfg.COL_IRIS].isna().sum())
    print(f"  com estagio: {n_com} | sem estagio: {n_sem}")
    dif = comp[(comp.p.notna()) & (comp.p < 0.05)]
    print(f"  variaveis com diferenca ao nivel de 5%: {len(dif)} de {len(comp)}")
    for _, r in dif.head(12).iterrows():
        print(f"    {str(r['variavel'])[:42]:<42} {r['com_estagio']:>9} vs "
              f"{r['sem_estagio']:>9}  ({r['medida']}, p={r['p']})")

    # O desfecho fica FORA da tabela acima, por ser posterior ao diagnostico,
    # mas e reportado a parte: e a evidencia de que os excluidos sao mais graves.
    if "Death" in df.columns:
        ob = df["Death"]
        ob = (ob.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])
              if ob.dtype == object else ob.astype(bool))
        tem_est = df[cfg.COL_IRIS].notna()
        print("  desfecho (posterior ao diagnostico, reportado a parte):")
        print(f"    obito -- com estagio: {ob[tem_est].mean():.1%} | "
              f"sem estagio: {ob[~tem_est].mean():.1%}")
        if "Method of death" in df.columns:
            eut = df["Method of death"].astype(str).str.strip().eq("Euthanasia")
            print(f"    eutanasia -- com estagio: {eut[tem_est].mean():.1%} | "
                  f"sem estagio: {eut[~tem_est].mean():.1%}")
        print("    (fora da tabela de caracteristicas: e consequencia, nao caracteristica)")

    print("[7] mapa de ausencia na coorte completa")
    aus_coorte = ausencia_por_variavel(df, entradas + comorbidades)

    print("[8] aplicando criterios de inclusao e exclusao")
    coorte, casc = aplicar_criterios(df, entradas)
    casc.imprimir()

    print("\n[9] mapa de ausencia no subconjunto rotulado")
    aus_rotulado = ausencia_por_variavel(coorte, entradas)

    cob = cobertura_informativa(coorte, entradas)
    fracas = cob[cob["pct_informativo"] < cfg.PROP_MINIMA_INFORMATIVA * 100]
    print("\n[10] cobertura informativa (descontando sentinelas textuais)")
    print(f"  variaveis abaixo de {cfg.PROP_MINIMA_INFORMATIVA:.0%} de conteudo informativo: "
          f"{len(fracas)}")
    for _, r in fracas.iterrows():
        print(f"    {r['coluna']:<44} {r['pct_informativo']:>5}%  "
              f"(sentinela em {r['n_sentinela']} registros)")

    print("[11] colunas com ausencia acima do limite")
    limite = cfg.PROP_MAXIMA_AUSENTE_COLUNA * 100
    revisar = aus_rotulado.loc[aus_rotulado["pct_ausente"] > limite, "coluna"].tolist()
    print(f"  colunas acima de {limite:.0f}% de ausencia no rotulado: {len(revisar)}")
    for c in revisar:
        pct = aus_rotulado.loc[aus_rotulado['coluna'] == c, 'pct_ausente'].iat[0]
        print(f"    {c}  ({pct}%)")
    print("  (listadas para revisao; nenhuma coluna e removida aqui)")

    print("[12] decisoes sobre o conjunto de entrada")
    entradas_finais = aplicar_decisoes_de_entrada(
        [c for c in entradas if c in coorte.columns or c.endswith("_pre")], coorte)
    entradas_finais = [c for c in entradas_finais if c in coorte.columns]

    print("[13] derivando indicadores de ausencia (conjunto SO_AUSENCIAS)")
    coorte, indic_ausencia = derivar_indicadores_ausencia(coorte, entradas_finais)
    print(f"  indicadores criados: {len(indic_ausencia)}")

    print("[14] conjuntos de variaveis")
    conjuntos = montar_conjuntos(entradas_finais)
    # Controle que nao ve nenhum valor clinico: so quais exames faltam, mais
    # idade e sexo.
    demografia = [c for c in ("Age", "Sex") if c in coorte.columns]
    conjuntos["SO_AUSENCIAS"] = indic_ausencia + demografia

    print("\n[15] gravando saidas")
    comp.to_csv(cfg.SAIDA_RELATORIOS / "rotulado_vs_nao_rotulado.csv", index=False)
    colunas_guardar = list(dict.fromkeys(
        ([cfg.COL_ID] if cfg.COL_ID else [])
        + [cfg.COL_IRIS] + entradas_finais + comorbidades + desfechos
    ))
    colunas_guardar = [c for c in colunas_guardar if c in coorte.columns]
    coorte[colunas_guardar].to_parquet(cfg.SAIDA_DADOS / "coorte.parquet", index=False)

    tabela_casc = casc.tabela()
    tabela_casc.to_csv(cfg.SAIDA_RELATORIOS / "cascata_exclusao.csv", index=False)

    print("\n[16] coorte das comorbidades: mesmos criterios, sem exigir estagio IRIS")
    coorte2, casc2 = aplicar_criterios(df, entradas, exigir_iris=False)
    casc2.imprimir()
    alvos2 = [f"{a}_pre" for a, q in cfg.PARES_TEMPORAIS
              if a in cfg.COMORBIDADES_MODELAVEIS]
    cols2 = [c for c in dict.fromkeys(
        ([cfg.COL_ID] if cfg.COL_ID else []) + [cfg.COL_IRIS]
        + entradas_finais + alvos2 + desfechos) if c in coorte2.columns]
    coorte2[cols2].to_parquet(cfg.SAIDA_DADOS / "coorte_comorbidades.parquet",
                              index=False)
    casc2.tabela().to_csv(cfg.SAIDA_RELATORIOS / "cascata_exclusao_escopo2.csv",
                          index=False)
    for a in alvos2:
        if a in coorte2.columns:
            print(f"    {a:<28} positivos: {int(coorte2[a].astype(bool).sum()):>4} "
                  f"({coorte2[a].astype(bool).mean():.1%})")
    aus_coorte.to_csv(cfg.SAIDA_RELATORIOS / "ausencia_coorte_completa.csv", index=False)
    aus_rotulado.to_csv(cfg.SAIDA_RELATORIOS / "ausencia_subconjunto_rotulado.csv", index=False)
    cob.to_csv(cfg.SAIDA_RELATORIOS / "cobertura_informativa.csv", index=False)
    tab_faixas.to_csv(cfg.SAIDA_RELATORIOS / "valores_implausiveis.csv", index=False)
    tab_temporal.to_csv(cfg.SAIDA_RELATORIOS / "achados_pre_diagnostico.csv", index=False)

    (cfg.SAIDA_DADOS / "conjuntos_variaveis.json").write_text(
        json.dumps(conjuntos, indent=2, ensure_ascii=False), encoding="utf-8")

    (cfg.SAIDA_RELATORIOS / "execucao_triagem.json").write_text(
        json.dumps({
            "sha256_bruto": sha,
            "executado_em": datetime.now().isoformat(timespec="seconds"),
            "semente": cfg.SEMENTE,
            "n_final": int(len(coorte)),
            "colunas_para_revisao": revisar,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\ncoorte final: {len(coorte)} registros, "
          f"{len(entradas_finais)} variaveis de entrada")
    print(f"saidas em {cfg.SAIDA}")


if __name__ == "__main__":
    main()
