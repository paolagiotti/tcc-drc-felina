"""
Configuracao central do projeto.

Tudo que e decisao de projeto mora aqui, nao espalhado pelo codigo: caminhos,
leitura do arquivo de origem, colunas-chave, faixas de plausibilidade, criterios
de inclusao, conjuntos de variaveis, configuracao do sistema, exames
recomendaveis, comorbidades modelaveis e parametros da camada interpretativa.
Os nomes de colunas e as faixas foram preenchidos a partir do inventario das
colunas (`inventario.py`) e da revisao manual do dicionario de variaveis.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

_AQUI = Path(__file__).resolve().parent

# Funciona com os arquivos dentro de uma pasta src/ ou soltos numa pasta so.
RAIZ = _AQUI.parent if _AQUI.name == "src" else _AQUI

# Onde procurar o CSV bruto, em ordem. A primeira que existir vence.
# Da para sobrescrever sem editar nada:
#   - variavel de ambiente:  export DRC_BRUTO=/caminho/Survival_data.csv
#   - argumento de linha:    python triagem.py --bruto /caminho/arquivo.csv
CANDIDATOS_BRUTO = [
    RAIZ / "dados" / "bruto" / "Survival_data.csv",
    RAIZ / "dados" / "Survival_data.csv",
    RAIZ / "Survival_data.csv",
    Path.cwd() / "Survival_data.csv",
]


def resolver_bruto(caminho_cli: str | None = None) -> Path:
    """Descobre onde esta o CSV bruto. Falha com mensagem util se nao achar."""
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


BRUTO = CANDIDATOS_BRUTO[0]          # valor nominal; resolver_bruto() manda
DICIONARIO = RAIZ / "dados" / "dicionario_variaveis.csv"

SAIDA = RAIZ / "saida"
SAIDA_DADOS = SAIDA / "dados"
SAIDA_RELATORIOS = SAIDA / "relatorios"
SAIDA_FIGURAS = SAIDA / "figuras"

SEMENTE = 42

# ---------------------------------------------------------------------------
# Leitura do arquivo bruto
# ---------------------------------------------------------------------------

CSV_SEP = ","
CSV_ENCODING = "utf-8"          # se falhar, tentar "latin-1"
CSV_DECIMAL = "."

# Strings que devem ser lidas como ausente. Dado retrospectivo de clinica
# costuma ter varias grafias para a mesma coisa.
VALORES_AUSENTES = [
    "", " ", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null",
    "-", "--", ".", "?", "UNK", "ND", "nd",
]

# Sentinelas textuais desta coorte. Elas NAO sao lidas como ausente na carga,
# porque a distincao importa: "Not recorded" significa que o veterinario nao
# registrou o achado, e isso e informacao sobre a consulta, nao sobre o gato.
# Sao contabilizadas a parte no relatorio de cobertura informativa, e a decisao
# de trata-las como categoria ou como ausente e declarada no texto.
SENTINELAS = [
    "Not recorded", "None recorded", "Not tested", "Unknown", "not recorded",
    "not done", "unknown",
]

# Abaixo deste percentual de valores informativos (nao ausentes e nao sentinela)
# no subconjunto rotulado, a variavel entra na lista de revisao.
PROP_MINIMA_INFORMATIVA = 0.50

# ---------------------------------------------------------------------------
# Colunas-chave
# ---------------------------------------------------------------------------
# Nome exato da coluna no CSV; None para o que nao existe na coorte.

COL_ID = "ID"                    # identificador do animal, se houver
COL_IRIS = "Iris stage at diagnosis"                  # estagio IRIS no diagnostico
COL_CREATININA = "Creatinine at diagnosis"            # creatinina serica no diagnostico
COL_UPC = "UPC at diagnosis"                   # relacao proteina/creatinina urinaria
COL_PAS = "Blood pressure at first measurement"                   # pressao arterial sistolica
COL_TEMPO_SOBREVIDA = None       # tempo ate o desfecho (para Kaplan-Meier)
COL_EVENTO_SOBREVIDA = None      # 1 = obito, 0 = censurado

# Colunas de data usadas na checagem de consistencia temporal.
# Pares (data_inicial, data_final): a primeira nao pode ser posterior a segunda.
# As datas do arquivo vem como texto no formato dd/mm/aaaa. Comparar texto e
# silenciosamente errado ("19/06/2012" > "06/10/2014" em ordem alfabetica), por
# isso as colunas de data sao convertidas explicitamente, e nao por inferencia.
DIA_PRIMEIRO = True
COLUNAS_DATA = [
    "Date of diagnosis",
    "Censored date",
    "Date hypertension diagnosed",
]

PARES_DATAS: list[tuple[str, str]] = [
    ("Date of diagnosis", "Censored date"),
]

# Janela do estudo. Diagnostico fora dela indica erro de registro.
JANELA_DIAGNOSTICO = ("2012-01-01", "2013-12-31")
COL_DATA_DIAGNOSTICO = "Date of diagnosis"

# Colunas ou marcadores que indicam injuria renal aguda associada.
# Pode ser uma coluna binaria, ou uma coluna de texto com termos de busca.
# `Previous AKI` registra Positive / Negative / Not tested, e 577 dos 625
# estao em Not tested, o que torna o criterio inverificavel nesta coorte.
# `Acute-on-chronic CKD` e a coluna booleana utilizavel, ainda que seja falsa
# em todos os registros rotulados.
COL_IRA = "Acute-on-chronic CKD"
TERMOS_IRA = ["true", "positive"]

# ---------------------------------------------------------------------------
# Unidades
# ---------------------------------------------------------------------------
# A coorte e britanica: creatinina em umol/L, a unidade dos limiares IRIS abaixo.

CREATININA_EM_UMOL = True
UMOL_POR_MGDL = 88.4             # creatinina: mg/dL * 88.4 = umol/L

# Limiares IRIS para gatos, em umol/L (IRIS, 2026). O limite superior do
# estagio 1 e exclusivo (140 exatos e estagio 2); os demais sao inclusivos.
# A regra esta em modelagem.regra_iris.
LIMIARES_IRIS_UMOL = {
    1: (None, 140.0),            # < 140
    2: (140.0, 250.0),
    3: (250.0, 440.0),
    4: (440.0, None),            # > 440
}

# ---------------------------------------------------------------------------
# Faixas fisiologicas plausiveis
# ---------------------------------------------------------------------------
# Valor fora da faixa vira ausente, e a triagem conta quantos foram convertidos.
# A faixa e de plausibilidade, nao de referencia clinica: deve ser larga o
# bastante para nao descartar caso grave real. Unidades: creatinina em umol/L;
# ureia, fosforo e potassio em mmol/L; pressao em mmHg; PCV em %.

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
# Varias colunas booleanas tem uma coluna companheira que registra se o achado
# foi documentado antes, no momento ou depois do diagnostico de DRC. Usar o
# booleano cru como preditor do estagio NO DIAGNOSTICO significa, nos casos
# posteriores, observar o futuro. Cada par abaixo gera uma variavel derivada
# `<nome>_pre`, verdadeira apenas quando o achado e anterior ou simultaneo ao
# diagnostico.
PARES_TEMPORAIS: list[tuple[str, str]] = [
    ("Heart Murmur", "HM first recorded"),
    ("Gallop rhythm", "GR first recorded"),
    ("Arrhythmia", "A first recorded"),
    ("Hyperthyroidism", "Diagnosis"),
    ("Periodontal disease", "Diagnosis.1"),
    ("Arthritis", "Diagnosis.2"),
    ("Cystitis/FLUTD", "Diagnosis.3"),
]

# Valores do qualificador aceitos como "nao posterior ao diagnostico".
TEMPORAL_ACEITO = ["Before CKD diagnosis", "At CKD diagnosis"]

# Substituir a versao crua do achado pela derivada `<nome>_pre` no conjunto de
# entrada. True e o comportamento correto: a versao crua afirma que o achado
# existe em algum momento do historico, o que vaza informacao futura.
SUBSTITUIR_POR_DERIVADA = True

# ---------------------------------------------------------------------------
# Tratamento de valores ausentes
# ---------------------------------------------------------------------------
# A ausencia nesta coorte NAO e aleatoria: concentra-se nos casos graves (a
# creatinina esta presente em 79% dos estagio III e em 32% dos estagio IV).
# Imputar apaga esse sinal, porque um valor plausivel, por definicao, parece
# normal. Tres tratamentos, comparaveis entre si:
#
#   "imputar"    -- imputacao multivariada (o tratamento originalmente previsto)
#   "indicador"  -- imputacao MAIS uma coluna binaria por variavel, marcando o
#                   que foi imputado. Preserva o sinal de ausencia sem quebrar
#                   modelos que nao aceitam valor ausente.
#   "nativo"     -- sem imputacao; o modelo recebe o ausente e decide sozinho.
#                   So funciona com floresta e gradiente; a regressao logistica
#                   nao aceita e e pulada.
TRATAMENTO_AUSENTES = "imputar"

# Sentinelas textuais ("Not recorded", "Not tested", "None recorded").
# False  -- entram como categoria propria: o modelo pode aprender que "nao
#           testado" tem significado clinico, coerente com o tratamento nativo
#           de ausentes adotado neste trabalho.
# True   -- convertidas em ausente, indistinguiveis de um campo vazio.
# Decisao explicita, nao comportamento padrao: e a mesma questao do tratamento
# de ausentes, aplicada a ausencia registrada em texto.
SENTINELA_COMO_AUSENTE = False

# Conjunto de variaveis do sistema. Fixado por argumento conceitual e
# arquitetural, nao por desempenho -- os candidatos empatam. O passo [6] do
# experimento seleciona modelo e braco DENTRO deste conjunto.
CONJUNTO_SISTEMA = "SEM_INDICADORES_EXAME"

# ---------------------------------------------------------------------------
# Escopo 3: recomendacao de exames
# ---------------------------------------------------------------------------
# Exames como o veterinario os PEDE: variaveis que saem da mesma coleta ficam
# juntas. Creatinina, ureia e fosforo vem da mesma amostra de sangue; densidade
# urinaria e fita, da mesma urinalise.
EXAMES_RECOMENDAVEIS = {
    "Bioquímica renal": ["Creatinine at diagnosis", "BUN at diagnosis",
                         "Phosphorus at diagnosis"],
    "Urinálise": ["USG at diagnosis", "Dipstick protein at diagnosis"],
    "UPC": ["UPC at diagnosis"],
    "Pressão arterial": ["Blood pressure at first measurement"],
}

# Piso de comparacao: ordem fixa, sempre a mesma para todo gato. Segue a logica
# do painel minimo das diretrizes -- bioquimica e urinalise primeiro.
ORDEM_FIXA_EXAMES = ["Bioquímica renal", "Urinálise", "Pressão arterial", "UPC"]

# Penalizacao por custo, na escala 0 a 1 em cada dimensao. Os numeros sao
# ILUSTRATIVOS, sem fonte de precos: servem para o mecanismo funcionar, nao para
# sustentar conclusao. Por isso a penalizacao e analise de sensibilidade, e nao
# entra na recomendacao principal.
#   custo           -- preco relativo em laboratorio veterinario brasileiro
#   invasividade    -- desconforto e risco da coleta
#   indisponibilidade -- dificuldade de acesso (equipamento, envio a laboratorio)
CUSTO_EXAMES = {
    "Bioquímica renal": {"custo": 0.3, "invasividade": 0.5, "indisponibilidade": 0.0},
    "Urinálise":        {"custo": 0.2, "invasividade": 0.5, "indisponibilidade": 0.0},
    "UPC":              {"custo": 0.6, "invasividade": 0.5, "indisponibilidade": 0.5},
    "Pressão arterial": {"custo": 0.3, "invasividade": 0.0, "indisponibilidade": 0.5},
}
PESOS_CUSTO = (1.0, 1.0, 1.0)     # alfa, beta, gama da formula da metodologia
LAMBDA_CUSTO = 0.3                # quanto a penalidade pesa contra o ganho

# Escopo 2: as quatro comorbidades com prevalencia suficiente para modelagem,
# todas na versao anterior ou simultanea ao diagnostico (_pre). As demais tem
# menos de 5% de prevalencia e foram declaradas inviaveis.
COMORBIDADES_MODELAVEIS = [
    "Periodontal disease",
    "Hyperthyroidism",
    "Cystitis/FLUTD",
    "Arthritis",
]

# Configuracao do sistema entregue, definida pela validacao aninhada com criterio
# RPS (gradiente|nenhum em 5 de 5 particoes). E ESTA configuracao que alimenta as
# analises descritivas -- sobrevida, decisao clinica, SHAP. Sem isso, cada
# analise leria as predicoes da configuracao que o passo [6] calhou de eleger, e
# o texto atribuiria ao sistema numeros de outro modelo.
CONFIGURACAO_SISTEMA = {"conjunto": "SEM_INDICADORES_EXAME",
                        "modelo": "gradiente", "braco": "nenhum"}

# Conjunto parcimonioso de Bradley et al. (2019): as quatro variaveis que o maior
# estudo de predicao de DRC felina selecionou a partir de 35 candidatas, sobre
# mais de cem mil gatos. Serve a duas perguntas: o sistema converge com aquele
# achado numa coorte independente, e as 14 variaveis adicionais acrescentam algo?
# A versao ESTENDIDA acrescenta as duas variaveis que o recomendador de exames
# precisa pontuar (UPC e pressao arterial): concilia a parcimonia, recomendada
# pela literatura de modelos preditivos clinicos, com a cobertura dos exames
# recomendaveis. Nenhuma delas codifica conduta de clinica.
CONJUNTO_PARCIMONIOSO_MAIS = [
    "Creatinine at diagnosis",
    "BUN at diagnosis",
    "USG at diagnosis",
    "Age",
    "UPC at diagnosis",
    "Blood pressure at first measurement",
]

CONJUNTO_PARCIMONIOSO = [
    "Creatinine at diagnosis",
    "BUN at diagnosis",
    "USG at diagnosis",
    "Age",
]

# ---------------------------------------------------------------------------
# Decisoes sobre o conjunto de entrada
# ---------------------------------------------------------------------------

# Indicadores de que o exame foi solicitado. Nao sao achado clinico: codificam a
# decisao do veterinario. Mantidos como entrada por padrao, mas isolados aqui
# para permitir a ablacao (conjunto SEM_INDICADORES_EXAME).
INDICADORES_EXAME_REALIZADO = [
    "Biochemistry",
    "Urinalyisis",
    "UPC",
    "Blood pressure measured",
    "Ocular exam",
    "Radiographs",
    "Ultrasound",
]

# Variaveis descartadas por conteudo informativo insuficiente. Os percentuais
# sao de conteudo real no subconjunto rotulado, descontadas as sentinelas.
DESCARTAR_POR_BAIXO_CONTEUDO = [
    "Previous exposure to lilies",   # 2,4% -- e ainda base do criterio de IRA
    "Previous AKI",                  # 3,3%
    "BCS 2 diagnosis (1-9)",         # 8,9%
    "Potassium at diagnosis",        # 9,8%
    "PCV at diagnosis",              # 10,6%
    "Grade of heart murmur",         # 30,9% e derivada de Heart Murmur
]

# Comorbidades anteriores ao diagnostico como PREDITORAS do estagio (escopo 1).
# Nao e circular: o alvo do escopo 1 e o estagio, nao a comorbidade. False (o
# valor usado) mantem o conjunto enxuto; True testaria a hipotese clinica de que
# o hipertireoidismo altera a creatinina por perda de massa muscular.
COMORBIDADES_PRE_COMO_ENTRADA = False

# ---------------------------------------------------------------------------
# Criterios de inclusao e exclusao
# ---------------------------------------------------------------------------

EXIGIR_IRIS = True               # manter apenas registros com estagio atribuido
EXIGIR_CREATININA = False        # aplicar como criterio separado, para medir o custo
PROP_MINIMA_PREENCHIDA = 0.50    # registro precisa de >= 50% das entradas preenchidas
PROP_MAXIMA_AUSENTE_COLUNA = 0.70  # coluna acima disso entra na lista de revisao

# ---------------------------------------------------------------------------
# Desenho de validacao (usado apenas para a aritmetica de viabilidade)
# ---------------------------------------------------------------------------

N_FOLDS = 5
N_REPETICOES = 3
FRACAO_HOLDOUT = 0.20
MIN_TESTE_POR_CLASSE = 3         # abaixo disso, holdout independente e inviavel

# ---------------------------------------------------------------------------
# Vocabulario de papeis do dicionario de variaveis
# ---------------------------------------------------------------------------

PAPEIS_VALIDOS = {
    "id",                  # identificador, nunca entra no modelo
    "entrada",             # candidata a variavel de entrada
    "alvo_iris",           # estagio IRIS
    "alvo_comorbidade",    # indicador binario de comorbidade (alvo do escopo 2)
    "desfecho",            # sobrevida, obito, causa
    "terapia",             # tratamento instituido apos o diagnostico
    "derivada",            # calculada a partir de outra coluna
    "excluir",             # vazamento estrutural, texto livre, coluna vazia
}

PAPEIS_FORA_DO_MODELO = {"id", "desfecho", "terapia", "derivada", "excluir"}


# ---------------------------------------------------------------------------
# Camada interpretativa
# ---------------------------------------------------------------------------
# Versao FIXA do modelo durante todo o desenvolvimento e a avaliacao final, como
# preve a metodologia. Escolhida antes de qualquer texto gerado:
#   - sucessor do gpt-4o-mini citado na proposta do trabalho, treinado com
#     foco em seguir instrucoes, a propriedade que a verificacao de
#     fidelidade mede;
#   - aceita temperatura 0 e semente, como a metodologia promete (os modelos
#     de raciocinio mais novos nao aceitam);
#   - tem versao com data, o que permite declarar exatamente qual modelo gerou
#     os textos.
# A chamada usa o nome curto porque o projeto da OpenAI recusava o nome com data,
# embora ele constasse dos modelos permitidos. Na execucao final, o nome curto
# serviu exatamente a versao declarada. O codigo confere, em cada chamada, a
# versao que a OpenAI serviu, e PARA se ela for outra: se o nome curto for
# apontado para uma versao nova, nenhum texto e gerado com modelo diferente do
# declarado. Conferir com: python verificar.py --api
MODELO_LLM = "gpt-4.1-mini"                   # nome usado na chamada
VERSAO_LLM = "gpt-4.1-mini-2025-04-14"        # versao declarada no texto
PRECO_LLM = (0.40, 1.60)          # US$ por milhao de tokens (entrada, saida)
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
K_TRECHOS = 5                     # trechos recuperados por consulta (metodologia: 3 a 5)
