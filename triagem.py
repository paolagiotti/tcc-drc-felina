"""
Triagem da coorte: do arquivo de origem as coortes do estadiamento (123 gatos)
e das comorbidades (618 gatos).

O que este script FAZ:
  4. converte unidades e tipos;
  5. aplica faixas de plausibilidade fisiologica (fora da faixa vira ausente);
  6. aplica inclusao/exclusao contando a cascata;
  7. levanta os mapas de ausencia (por variavel, por registro, co-ausencia);
  8. decide cortes de coluna e de linha;
  9. congela os quatro conjuntos de variaveis;
 10. conta a distribuicao do alvo e faz a aritmetica que decide o desenho;
 11. grava a coorte limpa e os relatorios.

O que este script NAO FAZ, de proposito:
  - nao imputa valor ausente;
  - nao padroniza nem codifica;
  - nao gera dado sintetico;
  - nao olha relacao entre variavel e alvo.
Tudo isso aprende dos dados e pertence ao pipeline, dentro do fold.

Uso:
    python triagem.py


    ----


    do arquivo bruto às duas coortes
    Valida o dicionário; converte números gravados como texto e as datas (explicitamente, dia primeiro, porque comparar datas como texto dá ordem alfabética); 
    deriva os achados _pre (verdadeiro só se anterior ou simultâneo ao diagnóstico; conta os positivos posteriores descartados); 
    aplica as faixas de plausibilidade; 
    compara os 124 com estágio contra os 501 sem estágio (validade externa; o desfecho fica fora da tabela, por ser consequência); 
    aplica a cascata de inclusão (estágio presente → datas consistentes → diagnóstico na janela → sem injúria aguda → ao menos metade das entradas preenchidas) contando cada corte; 
    monta os mapas de ausência (por variável, por registro e coausência de exames, insumo do recomendador) e a cobertura informativa descontando sentinelas; 
    aplica as decisões de entrada (troca o achado cru pelo _pre, descarta as seis de conteúdo insuficiente); 
    cria os indicadores _ausente do conjunto de controle SO_AUSENCIAS; 
    congela os conjuntos de variáveis da ablação (FULL, SEM_CREATININA, SEM_SUBESTADIAMENTO, SEM_INDICADORES_EXAME, parcimoniosos etc.); 
    faz a aritmética do desenho (menor classe com 18 → 3 a 4 por partição de teste; holdout inviável). 
    Grava coorte.parquet (123), coorte_comorbidades.parquet (618, mesmos critérios sem exigir estágio), conjuntos_variaveis.json, as tabelas e a figura da cascata.
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
    for col in getattr(cfg, "COLUNAS_DATA", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=cfg.DIA_PRIMEIRO,
                                     errors="coerce")
            datas.append(col)
    if datas:
        print(f"  convertidas para data ({'dia primeiro' if cfg.DIA_PRIMEIRO else 'mes primeiro'}): "
              f"{len(datas)} colunas")
    return df


def derivar_achados_pre_diagnostico(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria `<achado>_pre` a partir do achado e do seu qualificador temporal.

    O booleano cru afirma que o achado existe em algum momento do historico.
    Como preditor do estagio no diagnostico, isso vaza informacao futura sempre
    que o achado foi documentado depois. A derivada e verdadeira apenas quando
    o achado e anterior ou simultaneo ao diagnostico.
    """
    df = df.copy()
    linhas = []
    for achado, quando in getattr(cfg, "PARES_TEMPORAIS", []):
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
    """Par de datas em que a inicial e posterior a final.

    Recusa comparar colunas que nao sejam datetime: comparar "19/06/2012" com
    "06/10/2014" como texto e ordem alfabetica, nao cronologica, e o resultado
    sai errado sem nenhum aviso.
    """
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
    col = getattr(cfg, "COL_DATA_DIAGNOSTICO", None)
    janela = getattr(cfg, "JANELA_DIAGNOSTICO", None)
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
    """Criterios de inclusao. `exigir_iris=False` produz a coorte do escopo 2.

    Os criterios vivem num lugar so para as duas coortes: o escopo 1 exige
    estagio IRIS registrado, o escopo 2 nao -- as comorbidades nao dependem do
    estagio. Todos os demais criterios sao identicos.
    """
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

    if cfg.EXIGIR_CREATININA and cfg.COL_CREATININA in df.columns:
        antes = len(df)
        df = df[df[cfg.COL_CREATININA].notna()]
        casc.registrar("sem creatinina", antes, len(df))

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
    """Separa ausencia real de sentinela textual.

    Uma coluna como `Heart Murmur` aparenta 100% de cobertura, mas 32 dos 124
    registros rotulados trazem "Not recorded". Contar isso como dado presente
    superestima a cobertura e faz o modelo aprender a partir de uma categoria
    que so informa que o veterinario nao registrou o achado.
    """
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


def ausencia_por_registro(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Proporcao de entradas ausentes em cada registro."""
    cols = [c for c in cols if c in df.columns]
    pct = df[cols].isna().mean(axis=1) * 100
    return pd.DataFrame({
        "indice": df.index,
        "pct_ausente": pct.round(1).values,
        "n_preenchidos": df[cols].notna().sum(axis=1).values,
    }).sort_values("pct_ausente", ascending=False).reset_index(drop=True)


def coausencia(df: pd.DataFrame, cols: list[str], limite: int = 25) -> pd.DataFrame:
    """Quais exames faltam juntos. Alimenta o escopo 3."""
    cols = [c for c in cols if c in df.columns][:limite]
    faltante = df[cols].isna().astype(int)
    m = faltante.T.dot(faltante)
    n = faltante.sum()
    uniao = np.add.outer(n.values, n.values) - m.values
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(uniao > 0, m.values / uniao, 0.0)
    return pd.DataFrame(jaccard.round(3), index=cols, columns=cols)


# ---------------------------------------------------------------------------
# Passo 9: conjuntos de variaveis
# ---------------------------------------------------------------------------

def aplicar_decisoes_de_entrada(entradas: list[str], df: pd.DataFrame) -> list[str]:
    """Aplica as decisoes de composicao do conjunto de entrada.

    1. Troca o achado cru pela derivada `_pre`, que e a unica versao disponivel
       no momento do diagnostico.
    2. Remove as variaveis descartadas por conteudo informativo insuficiente.
    3. Opcionalmente acrescenta as comorbidades anteriores como preditoras.
    """
    entradas = list(entradas)

    if getattr(cfg, "SUBSTITUIR_POR_DERIVADA", False):
        trocas = []
        for achado, _ in getattr(cfg, "PARES_TEMPORAIS", []):
            derivada = f"{achado}_pre"
            if achado in entradas and derivada in df.columns:
                entradas[entradas.index(achado)] = derivada
                trocas.append((achado, derivada))
        for cru, der in trocas:
            print(f"    {cru}  ->  {der}")
        if trocas:
            print(f"  achados substituidos pela versao anterior ao diagnostico: {len(trocas)}")

    descartar = [c for c in getattr(cfg, "DESCARTAR_POR_BAIXO_CONTEUDO", [])
                 if c in entradas]
    if descartar:
        entradas = [c for c in entradas if c not in descartar]
        print(f"  descartadas por conteudo insuficiente: {len(descartar)}")
        for c in descartar:
            print(f"    {c}")

    if getattr(cfg, "COMORBIDADES_PRE_COMO_ENTRADA", False):
        extras = [f"{a}_pre" for a, _ in getattr(cfg, "PARES_TEMPORAIS", [])
                  if f"{a}_pre" in df.columns and f"{a}_pre" not in entradas]
        entradas += extras
        if extras:
            print(f"  comorbidades anteriores acrescentadas como preditoras: {len(extras)}")

    return entradas


def derivar_indicadores_ausencia(df: pd.DataFrame, entradas: list[str]
                                 ) -> tuple[pd.DataFrame, list[str]]:
    """Cria uma coluna `<variavel>_ausente` para cada entrada com ausencia.

    Sustenta o conjunto SO_AUSENCIAS, que responde quanto do desempenho vem do
    padrao de solicitacao de exames da clinica e quanto vem do valor clinico.
    As colunas NAO entram nos demais conjuntos.
    """
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


# Papeis que NAO entram na comparacao 124 x 501. Desfecho e terapia sao
# posteriores ao diagnostico: diferir neles e consequencia, nao caracteristica
# de quem entrou na coorte. O identificador nao e variavel.
PAPEIS_FORA_DA_COMPARACAO = {"id", "desfecho", "terapia"}


def comparar_rotulado_vs_nao_rotulado(bruto: pd.DataFrame, dic: pd.DataFrame
                                      ) -> pd.DataFrame:
    """Compara quem tem estagio IRIS registrado com quem nao tem.

    Os 501 registros sem estagio sao 80% da coorte de origem. Saber se os 124
    restantes diferem sistematicamente deles e questao de validade externa, nao
    detalhe: define a que populacao o modelo se aplica.
    """
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
    """Congela os conjuntos de variaveis usados na ablacao do escopo 1.

    Distincao que vem da diretriz IRIS e nao pode ser embaralhada:
      - o ESTAGIO (1-4) e determinado pela creatinina (e, quando disponivel,
        pela SDMA), medida em paciente estavel;
      - a proteinuria (UPC) e a pressao arterial determinam o SUBESTAGIO, que
        qualifica o estagio mas nao altera o numero.

    Por isso `SEM_CREATININA` e a ablacao central: ela remove o determinante do
    rotulo. `SEM_SUBESTADIAMENTO` remove as variaveis prognosticas e responde a
    outra pergunta. Junta-las num unico conjunto tornaria o resultado ininterpretavel.
    """
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
            if c not in getattr(cfg, "INDICADORES_EXAME_REALIZADO", [])],
    }
    for nome, cols in conjuntos.items():
        print(f"  {nome:<34} {len(cols):>3} variaveis")
    if not definidor:
        print("  AVISO: creatinina ausente do conjunto de entrada. A ablacao "
              "central do escopo 1 depende dela.")
    return conjuntos


# ---------------------------------------------------------------------------
# Passo 10: aritmetica do desenho
# ---------------------------------------------------------------------------

def aritmetica_desenho(y: pd.Series) -> dict:
    """Contas que decidem o desenho de validacao a partir do tamanho da menor classe."""
    dist = y.value_counts().sort_index()
    menor = int(dist.min())

    # A menor classe raramente divide pelo numero de folds, entao os folds nao
    # tem todos o mesmo tamanho. Com 18 registros em 5 folds, a distribuicao
    # real e 4, 4, 4, 3, 3. Reportar a faixa evita a impressao falsa de que
    # todo fold de teste tem o mesmo numero de casos.
    base, resto = divmod(menor, cfg.N_FOLDS)
    teste_min, teste_max = base, base + (1 if resto else 0)
    treino_min, treino_max = menor - teste_max, menor - teste_min

    k_smote = max(1, min(3, treino_min - 1))
    holdout_menor = int(np.floor(menor * cfg.FRACAO_HOLDOUT))
    holdout_viavel = holdout_menor >= cfg.MIN_TESTE_POR_CLASSE

    resumo = {
        "distribuicao": {str(k): int(v) for k, v in dist.items()},
        "n_total": int(dist.sum()),
        "menor_classe": menor,
        "n_treino_menor_classe_por_fold": [treino_min, treino_max],
        "n_teste_menor_classe_por_fold": [teste_min, teste_max],
        "folds_desbalanceados": bool(resto),
        "k_neighbors_smote_recomendado": k_smote,
        "holdout_menor_classe": holdout_menor,
        "holdout_independente_viavel": bool(holdout_viavel),
        "impacto_de_1_erro_na_sensibilidade_pp": [
            round(100 / max(teste_max, 1), 1), round(100 / max(teste_min, 1), 1)],
    }

    print("\n--- aritmetica do desenho ---")
    for estagio, n in dist.items():
        print(f"  estagio {estagio}: {n:>4}  ({n / dist.sum():.1%})")
    faixa = lambda a, b: f"{a}" if a == b else f"{a} a {b}"
    print(f"  menor classe: {menor}")
    if resto:
        print(f"    {menor} nao divide por {cfg.N_FOLDS} folds ({menor}/{cfg.N_FOLDS} = "
              f"{menor / cfg.N_FOLDS:.1f}), entao os folds tem tamanhos diferentes")
    print(f"  na particao de treino de cada fold: {faixa(treino_min, treino_max)}")
    print(f"  na particao de teste de cada fold:  {faixa(teste_min, teste_max)}")
    print(f"  -> k_neighbors do SMOTE: {k_smote} (limitado pelo menor fold de treino)")
    print(f"  -> holdout de {cfg.FRACAO_HOLDOUT:.0%} reteria {holdout_menor} casos da menor "
          f"classe: {'viavel' if holdout_viavel else 'inviavel'}")
    pp = resumo["impacto_de_1_erro_na_sensibilidade_pp"]
    print(f"  -> 1 erro na menor classe move a sensibilidade em "
          f"{faixa(pp[0], pp[1])} pontos percentuais, conforme o fold")

    if k_smote < 2:
        print("  AVISO: menor classe pequena demais para SMOTE confiavel. "
              "Considerar apenas ponderacao de classes.")
    if not holdout_viavel:
        print("  -> holdout inviavel: manter apenas validacao cruzada repetida, "
              "sem conjunto de teste independente.")
    return resumo


# ---------------------------------------------------------------------------
# Figura da cascata
# ---------------------------------------------------------------------------

def figura_cascata(tab: pd.DataFrame, destino) -> None:
    """Figura da cascata de exclusao, no padrao das figuras do texto."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib ausente; figura da cascata nao gerada")
        return

    # Forma das figuras do texto: acentos so no rotulo mostrado (a tabela CSV
    # nao muda), 300 dpi, largura final de 16 cm, letra de 10 pt, sem titulo.
    acentos = {
        "Registros brutos": "Registros do arquivo",
        "sem estagio IRIS no diagnostico": "Sem estágio IRIS no diagnóstico",
        "datas inconsistentes": "Datas inconsistentes",
        "diagnostico fora da janela do estudo": "Diagnóstico fora da janela do estudo",
        "injuria renal aguda associada": "Injúria renal aguda associada",
        "sem creatinina": "Sem creatinina",
    }
    def mostrar(c: str) -> str:
        c = acentos.get(c, c)
        return c[:1].upper() + c[1:]
    fig, ax = plt.subplots(figsize=(16 / 2.54, 0.42 * len(tab) + 0.8))
    y = range(len(tab))
    ax.barh(list(y), tab["restantes"], color="#5b6670")
    ax.set_yticks(list(y))
    ax.set_yticklabels([mostrar(c) for c in tab["criterio"]], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Registros restantes", fontsize=10)
    ax.tick_params(axis="x", labelsize=10)
    for i, (rest, rem) in enumerate(zip(tab["restantes"], tab["removidos"])):
        rotulo = f"{rest}" if rem == 0 else f"{rest}  (−{rem})"
        ax.text(rest + max(tab['restantes']) * 0.01, i, rotulo,
                va="center", fontsize=10)
    ax.set_xlim(0, max(tab["restantes"]) * 1.22)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    print(f"  figura: {destino.name}")


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

    print("[4] convertendo tipos e unidades")
    df = converter_tipos(bruto, dic)
    df = normalizar_creatinina(df)

    print("[4b] derivando achados anteriores ao diagnostico")
    df, tab_temporal = derivar_achados_pre_diagnostico(df)

    print("[5] aplicando faixas de plausibilidade")
    df, tab_faixas = aplicar_faixas(df)

    print("[6b] comparando registros com e sem estagio IRIS registrado")
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
        print("    Nao entra na tabela de caracteristicas: e consequencia, nao")
        print("    caracteristica de quem foi incluido. Reportar como tal.")

    print("[7a] mapa de ausencia na coorte completa")
    aus_coorte = ausencia_por_variavel(df, entradas + comorbidades)

    print("[6] aplicando criterios de inclusao e exclusao")
    coorte, casc = aplicar_criterios(df, entradas)
    casc.imprimir()

    print("\n[7b] mapas de ausencia no subconjunto rotulado")
    aus_rotulado = ausencia_por_variavel(coorte, entradas)
    aus_registro = ausencia_por_registro(coorte, entradas)
    mat_coausencia = coausencia(coorte, entradas)

    cob = cobertura_informativa(coorte, entradas)
    fracas = cob[cob["pct_informativo"] < cfg.PROP_MINIMA_INFORMATIVA * 100]
    print(f"\n[7c] cobertura informativa (descontando sentinelas textuais)")
    print(f"  variaveis abaixo de {cfg.PROP_MINIMA_INFORMATIVA:.0%} de conteudo informativo: "
          f"{len(fracas)}")
    for _, r in fracas.iterrows():
        print(f"    {r['coluna']:<44} {r['pct_informativo']:>5}%  "
              f"(sentinela em {r['n_sentinela']} registros)")

    print("[8] cortes de coluna")
    limite = cfg.PROP_MAXIMA_AUSENTE_COLUNA * 100
    revisar = aus_rotulado.loc[aus_rotulado["pct_ausente"] > limite, "coluna"].tolist()
    print(f"  colunas acima de {limite:.0f}% de ausencia no rotulado: {len(revisar)}")
    for c in revisar:
        pct = aus_rotulado.loc[aus_rotulado['coluna'] == c, 'pct_ausente'].iat[0]
        print(f"    {c}  ({pct}%)")
    print("  (decisao de manter ou cortar e sua; nenhuma coluna foi removida aqui)")

    print("[8b] decisoes sobre o conjunto de entrada")
    entradas_finais = aplicar_decisoes_de_entrada(
        [c for c in entradas if c in coorte.columns or c.endswith("_pre")], coorte)
    entradas_finais = [c for c in entradas_finais if c in coorte.columns]

    if getattr(cfg, "SENTINELA_COMO_AUSENTE", False):
        sent = set(cfg.SENTINELAS)
        n_conv = 0
        for col in entradas_finais:
            if col in coorte.columns and not pd.api.types.is_numeric_dtype(coorte[col]):
                m = coorte[col].astype(str).str.strip().isin(sent)
                n_conv += int(m.sum())
                coorte.loc[m, col] = pd.NA
        print(f"[8b2] sentinelas convertidas em ausente: {n_conv} valores")
    else:
        print("[8b2] sentinelas MANTIDAS como categoria propria "
              "(config.SENTINELA_COMO_AUSENTE = False)")

    print("[8c] derivando indicadores de ausencia (conjunto SO_AUSENCIAS)")
    coorte, indic_ausencia = derivar_indicadores_ausencia(coorte, entradas_finais)
    print(f"  indicadores criados: {len(indic_ausencia)}")

    print("[9] conjuntos de variaveis")
    conjuntos = montar_conjuntos(entradas_finais)
    # Baseline que NAO ve nenhum valor clinico: apenas quais exames faltam, mais
    # demografia. Mede quanto do desempenho vem do padrao de solicitacao da
    # clinica e quanto vem do estado do paciente.
    demografia = [c for c in ("Age", "Sex") if c in coorte.columns]
    conjuntos["SO_AUSENCIAS"] = indic_ausencia + demografia
    # Conjunto parcimonioso da literatura (Bradley et al., 2019). Candidato a
    # sistema, nao controle: se empatar com o conjunto completo, o sistema opera
    # com quatro exames.
    parcim = [c for c in getattr(cfg, "CONJUNTO_PARCIMONIOSO", [])
              if c in coorte.columns]
    if len(parcim) >= 3:
        conjuntos["PARCIMONIOSO_BRADLEY"] = parcim
    parcim_mais = [c for c in getattr(cfg, "CONJUNTO_PARCIMONIOSO_MAIS", [])
                   if c in coorte.columns]
    if len(parcim_mais) >= 4:
        conjuntos["PARCIMONIOSO_MAIS"] = parcim_mais

    print("[10] aritmetica do desenho")
    resumo = aritmetica_desenho(coorte[cfg.COL_IRIS].astype("Int64").dropna())

    print("\n[11] gravando saidas")
    comp.to_csv(cfg.SAIDA_RELATORIOS / "rotulado_vs_nao_rotulado.csv", index=False)
    colunas_guardar = list(dict.fromkeys(
        ([cfg.COL_ID] if cfg.COL_ID else [])
        + [cfg.COL_IRIS] + entradas_finais + comorbidades + desfechos
    ))
    colunas_guardar = [c for c in colunas_guardar if c in coorte.columns]
    coorte[colunas_guardar].to_parquet(cfg.SAIDA_DADOS / "coorte.parquet", index=False)

    tabela_casc = casc.tabela()
    tabela_casc.to_csv(cfg.SAIDA_RELATORIOS / "cascata_exclusao.csv", index=False)

    # --- coorte do escopo 2: mesmos criterios, sem exigir estagio IRIS ---
    print("\n[12] coorte do escopo 2 (comorbidades): sem exigir estagio IRIS")
    coorte2, casc2 = aplicar_criterios(df, entradas, exigir_iris=False)
    casc2.imprimir()
    alvos2 = [f"{a}_pre" for a, q in getattr(cfg, "PARES_TEMPORAIS", [])
              if a in getattr(cfg, "COMORBIDADES_MODELAVEIS", [])]
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
    aus_registro.to_csv(cfg.SAIDA_RELATORIOS / "ausencia_por_registro.csv", index=False)
    mat_coausencia.to_csv(cfg.SAIDA_RELATORIOS / "coausencia_jaccard.csv")
    cob.to_csv(cfg.SAIDA_RELATORIOS / "cobertura_informativa.csv", index=False)
    tab_faixas.to_csv(cfg.SAIDA_RELATORIOS / "valores_implausiveis.csv", index=False)
    tab_temporal.to_csv(cfg.SAIDA_RELATORIOS / "achados_pre_diagnostico.csv", index=False)

    (cfg.SAIDA_DADOS / "conjuntos_variaveis.json").write_text(
        json.dumps(conjuntos, indent=2, ensure_ascii=False), encoding="utf-8")

    (cfg.SAIDA_RELATORIOS / "aritmetica_desenho.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

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

    figura_cascata(tabela_casc, cfg.SAIDA_FIGURAS / "cascata_exclusao.png")

    print(f"\ncoorte final: {len(coorte)} registros, "
          f"{len(entradas_finais)} variaveis de entrada")
    print(f"saidas em {cfg.SAIDA}")


if __name__ == "__main__":
    main()
