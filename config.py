"""
Configuracao central do projeto. Toda decisao parametrizada mora aqui; os
demais scripts so leem.

não executa nada; 
concentra toda decisão do projeto para que os outros scripts só leiam daqui.
Caminhos e semente (42); 
como ler o CSV (quais textos viram ausente; as sentinelas “Not recorded”, “Not tested” não viram ausente, ficam como categoria, porque dizem algo sobre a consulta); 
nomes das colunas-chave (estágio IRIS, creatinina, UPC, pressão); colunas de data e a janela do estudo (2012 a 2013); 
a coluna de injúria renal aguda; os limiares IRIS em µmol/L (140, 250, 440); as faixas de plausibilidade (fora delas vira ausente); 
os pares achado + “first recorded” que geram as variáveis _pre; os três tratamentos de ausentes; 
o conjunto do sistema (SEM_INDICADORES_EXAME, 18 variáveis) e a configuração do sistema (gradiente, sem correção de desbalanceamento), que todas as análises descritivas leem daqui, e não do que um passo de seleção elegeu; os quatro exames recomendáveis e a ordem fixa; os custos ilustrativos; 
as quatro comorbidades modeláveis; 
as variáveis de Bradley (destacadas na figura do SHAP); 
o desenho de validação (5 partições × 3 repetições) e o modelo de linguagem com versão fixada.

"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent

# Onde procurar o CSV de origem, em ordem. Tambem aceita a variavel de ambiente
# DRC_BRUTO ou o argumento --bruto dos scripts que o leem.
CANDIDATOS_BRUTO = [
    RAIZ / "dados" / "bruto" / "Survival_data.csv",
    RAIZ / "dados" / "Survival_data.csv",
    RAIZ / "Survival_data.csv",
    Path.cwd() / "Survival_data.csv",
]


def resolver_bruto(caminho_cli: str | None = None) -> Path:
    """Localiza o CSV de origem; interrompe com mensagem util se nao achar."""
    if caminho_cli:
        p = Path(caminho_cli).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"Arquivo nao encontrado: {p}")
        return p

    if env := os.environ.get("DRC_BRUTO"):
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"DRC_BRUTO aponta para arquivo inexistente: {p}")
        return p

    for p in CANDIDATOS_BRUTO:
        if p.exists():
            return p

    tentados = "\n".join(f"  - {p}" for p in CANDIDATOS_BRUTO)
    raise SystemExit(
        "Survival_data.csv nao encontrado. Procurei em:\n" + tentados +
        "\n\nColoque o arquivo em um desses caminhos, ou rode com:\n"
        "  python inventario.py --bruto /caminho/para/Survival_data.csv"
    )


DICIONARIO = RAIZ / "dados" / "dicionario_variaveis.csv"

SAIDA = RAIZ / "saida"
SAIDA_DADOS = SAIDA / "dados"
SAIDA_RELATORIOS = SAIDA / "relatorios"
SAIDA_FIGURAS = SAIDA / "figuras"

SEMENTE = 42

# ---------------------------------------------------------------------------
# Leitura do arquivo de origem
# ---------------------------------------------------------------------------

CSV_SEP = ","
CSV_ENCODING = "utf-8"
CSV_DECIMAL = "."

# Grafias lidas como ausente.
VALORES_AUSENTES = [
    "", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null",
    "-", "--", ".", "?", "UNK", "ND", "nd",
]

# Sentinelas textuais da coorte. Nao sao lidas como ausente: "Not recorded"
# informa que o veterinario nao registrou o achado, e isso e informacao sobre a
# consulta. Entram como categoria propria e sao contadas a parte na cobertura.
SENTINELAS = [
    "Not recorded", "None recorded", "Not tested", "Unknown", "not recorded",
    "not done", "unknown",
]

# Abaixo deste percentual de valores informativos (nem ausentes nem sentinela)
# no subconjunto rotulado, a variavel entra na lista de revisao.
PROP_MINIMA_INFORMATIVA = 0.50

# ---------------------------------------------------------------------------
# Colunas-chave
# ---------------------------------------------------------------------------

COL_ID = "ID"
COL_IRIS = "Iris stage at diagnosis"
COL_CREATININA = "Creatinine at diagnosis"
COL_UPC = "UPC at diagnosis"
COL_PAS = "Blood pressure at first measurement"

# Datas vem como texto dd/mm/aaaa e sao convertidas explicitamente (comparar
# como texto daria ordem alfabetica).
DIA_PRIMEIRO = True
COLUNAS_DATA = [
    "Date of diagnosis",
    "Censored date",
    "Date hypertension diagnosed",
]

# Pares (data inicial, data final): a primeira nao pode ser posterior a segunda.
PARES_DATAS: list[tuple[str, str]] = [
    ("Date of diagnosis", "Censored date"),
]

# Janela de recrutamento do estudo. Diagnostico fora dela indica erro de registro.
JANELA_DIAGNOSTICO = ("2012-01-01", "2013-12-31")
COL_DATA_DIAGNOSTICO = "Date of diagnosis"

# Injuria renal aguda associada: criterio de exclusao. `Previous AKI` esta em
# "Not tested" em 577 dos 625 registros; a coluna utilizavel e a booleana.
COL_IRA = "Acute-on-chronic CKD"
TERMOS_IRA = ["true", "positive"]

# ---------------------------------------------------------------------------
# Unidades e limiares IRIS
# ---------------------------------------------------------------------------

CREATININA_EM_UMOL = True
UMOL_POR_MGDL = 88.4

# Limiares IRIS para gatos, em umol/L (IRIS, 2026). O limite superior do estagio
# 1 e exclusivo (140 exatos e estagio 2); os demais sao inclusivos. A regra esta
# em modelagem.regra_iris.
LIMIARES_IRIS_UMOL = {
    1: (None, 140.0),
    2: (140.0, 250.0),
    3: (250.0, 440.0),
    4: (440.0, None),
}

# ---------------------------------------------------------------------------
# Faixas de plausibilidade fisiologica
# ---------------------------------------------------------------------------
# Valor fora da faixa vira ausente. A faixa e de plausibilidade, nao de
# referencia clinica: larga o bastante para nao descartar caso grave real.
# Creatinina em umol/L; ureia, fosforo e potassio em mmol/L; pressao em mmHg.

FAIXAS_PLAUSIVEIS: dict[str, tuple[float, float]] = {
    "Creatinine at diagnosis": (30.0, 3000.0),
    "BUN at diagnosis": (1.0, 120.0),
    "USG at diagnosis": (1.000, 1.080),
    "UPC at diagnosis": (0.0, 20.0),
    "Blood pressure at first measurement": (40.0, 300.0),
    "Phosphorus at diagnosis": (0.1, 12.0),
    "Potassium at diagnosis": (1.5, 10.0),
    "PCV at diagnosis": (5.0, 70.0),
    "BCS 2 diagnosis (1-9)": (1.0, 9.0),
}

# ---------------------------------------------------------------------------
# Achados com qualificador temporal
# ---------------------------------------------------------------------------
# Cada par (achado, qualificador) gera a variavel `<achado>_pre`, verdadeira so
# quando o achado e anterior ou simultaneo ao diagnostico. O booleano cru afirma
# que o achado existe em algum momento do historico, o que vaza informacao
# futura; a derivada substitui o cru no conjunto de entrada.
PARES_TEMPORAIS: list[tuple[str, str]] = [
    ("Heart Murmur", "HM first recorded"),
    ("Gallop rhythm", "GR first recorded"),
    ("Arrhythmia", "A first recorded"),
    ("Hyperthyroidism", "Diagnosis"),
    ("Periodontal disease", "Diagnosis.1"),
    ("Arthritis", "Diagnosis.2"),
    ("Cystitis/FLUTD", "Diagnosis.3"),
]
TEMPORAL_ACEITO = ["Before CKD diagnosis", "At CKD diagnosis"]

# ---------------------------------------------------------------------------
# Tratamento de valores ausentes e conjunto do sistema
# ---------------------------------------------------------------------------
# Tres tratamentos, comparaveis entre si:
#   "imputar"    -- imputacao multivariada;
#   "indicador"  -- imputacao mais uma coluna binaria por variavel imputada;
#   "nativo"     -- sem imputacao: o modelo recebe o ausente.
TRATAMENTO_AUSENTES = "imputar"

# Conjunto de variaveis do sistema: as entradas sem os indicadores de exame
# solicitado (que descrevem conduta, nao o paciente). Fixado por argumento
# conceitual, nao por desempenho; a selecao escolhe modelo e braco dentro dele.
CONJUNTO_SISTEMA = "SEM_INDICADORES_EXAME"

# Configuracao do sistema, definida pela validacao aninhada com criterio RPS.
# Todas as analises descritivas (decisao clinica, SHAP, fichas) leem
# esta configuracao, e nao a que um passo de selecao elege em cada execucao.
CONFIGURACAO_SISTEMA = {"conjunto": "SEM_INDICADORES_EXAME",
                        "modelo": "gradiente", "braco": "nenhum"}

# Indicadores de que o exame foi solicitado. Mantidos em FULL e isolados aqui
# para a ablacao que define SEM_INDICADORES_EXAME.
INDICADORES_EXAME_REALIZADO = [
    "Biochemistry",
    "Urinalyisis",
    "UPC",
    "Blood pressure measured",
    "Ocular exam",
    "Radiographs",
    "Ultrasound",
]

# Variaveis descartadas por conteudo informativo insuficiente no subconjunto
# rotulado (percentual de conteudo real, descontadas as sentinelas).
DESCARTAR_POR_BAIXO_CONTEUDO = [
    "Previous exposure to lilies",   # 2,4%
    "Previous AKI",                  # 3,3%
    "BCS 2 diagnosis (1-9)",         # 8,9%
    "Potassium at diagnosis",        # 9,8%
    "PCV at diagnosis",              # 10,6%
    "Grade of heart murmur",         # 30,9%, e derivada de Heart Murmur
]

# As quatro variaveis de Bradley et al. (2019), destacadas na figura do SHAP
# (convergencia com a literatura; nao formam um conjunto de variaveis do sistema).
VARIAVEIS_BRADLEY = [
    "Creatinine at diagnosis",
    "BUN at diagnosis",
    "USG at diagnosis",
    "Age",
]

# ---------------------------------------------------------------------------
# Criterios de inclusao e desenho de validacao
# ---------------------------------------------------------------------------

EXIGIR_IRIS = True                 # coorte do estadiamento: so registros com estagio
PROP_MINIMA_PREENCHIDA = 0.50      # registro precisa de >= 50% das entradas preenchidas
PROP_MAXIMA_AUSENTE_COLUNA = 0.70  # coluna acima disso entra na lista de revisao

N_FOLDS = 5
N_REPETICOES = 3

# Papeis aceitos na coluna `papel` do dicionario de variaveis.
PAPEIS_VALIDOS = {
    "id",                  # identificador, nunca entra no modelo
    "entrada",             # candidata a variavel de entrada
    "alvo_iris",           # estagio IRIS
    "alvo_comorbidade",    # indicador binario de comorbidade
    "desfecho",            # sobrevida, obito, causa
    "terapia",             # tratamento instituido apos o diagnostico
    "derivada",            # calculada a partir de outra coluna
    "excluir",             # vazamento estrutural, texto livre, coluna vazia
}
PAPEIS_FORA_DO_MODELO = {"id", "desfecho", "terapia", "derivada", "excluir"}

# ---------------------------------------------------------------------------
# Componente de recomendacao
# ---------------------------------------------------------------------------

# Comorbidades com prevalencia suficiente para modelagem, na versao `_pre`.
COMORBIDADES_MODELAVEIS = [
    "Periodontal disease",
    "Hyperthyroidism",
    "Cystitis/FLUTD",
    "Arthritis",
]

# Exames como o veterinario os pede: variaveis da mesma coleta ficam juntas.
EXAMES_RECOMENDAVEIS = {
    "Bioquímica renal": ["Creatinine at diagnosis", "BUN at diagnosis",
                         "Phosphorus at diagnosis"],
    "Urinálise": ["USG at diagnosis", "Dipstick protein at diagnosis"],
    "UPC": ["UPC at diagnosis"],
    "Pressão arterial": ["Blood pressure at first measurement"],
}

# Comparador fixo: a mesma ordem para todo gato, derivada do painel minimo do
# consenso ISFM (bioquimica e urinalise primeiro).
ORDEM_FIXA_EXAMES = ["Bioquímica renal", "Urinálise", "Pressão arterial", "UPC"]

# Penalizacao por custo, invasividade e acesso (escala 0 a 1). Valores
# ILUSTRATIVOS, sem fonte de precos: a penalizacao e analise de sensibilidade e
# nao entra na recomendacao principal.
CUSTO_EXAMES = {
    "Bioquímica renal": {"custo": 0.3, "invasividade": 0.5, "indisponibilidade": 0.0},
    "Urinálise":        {"custo": 0.2, "invasividade": 0.5, "indisponibilidade": 0.0},
    "UPC":              {"custo": 0.6, "invasividade": 0.5, "indisponibilidade": 0.5},
    "Pressão arterial": {"custo": 0.3, "invasividade": 0.0, "indisponibilidade": 0.5},
}
PESOS_CUSTO = (1.0, 1.0, 1.0)     # alfa, beta, gama
LAMBDA_CUSTO = 0.3                # peso da penalidade contra o ganho

# ---------------------------------------------------------------------------
# Camada interpretativa
# ---------------------------------------------------------------------------
# Modelo de linguagem com versao fixa durante o desenvolvimento e a avaliacao.
# A chamada usa o nome curto; o codigo confere, em cada chamada, a versao que a
# OpenAI serviu e interrompe a execucao se for outra (python verificar.py --api).
MODELO_LLM = "gpt-4.1-mini"
VERSAO_LLM = "gpt-4.1-mini-2025-04-14"
PRECO_LLM = (0.40, 1.60)          # US$ por milhao de tokens (entrada, saida)
K_TRECHOS = 5                     # trechos recuperados por gato
