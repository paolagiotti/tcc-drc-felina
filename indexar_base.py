"""
Indexacao da base de conhecimento da camada interpretativa (RAG).

Le os PDFs selecionados pela autora e pela orientadora, divide o texto em
trechos, representa cada trecho por vetores de um modelo multilingue de sentence
embeddings e grava um indice vetorial FAISS. Implementacao direta, sem frameworks,
para que cada etapa fique explicita.

ONDE COLOCAR OS PDFs:  <pasta do projeto>/base_conhecimento/
Os PDFs NAO vao para o repositorio publico: o indice e construido localmente.

EXTRACAO (revista depois da conferencia manual dos trechos da primeira versao):
  - ordem de leitura: a do proprio arquivo. A ordenacao geometrica usada antes
    (sort=True) intercalava linhas das duas colunas dos artigos;
  - descartados: texto girado (marcas d'agua), blocos curtos repetidos em muitas
    paginas (cabecalhos e rodapes), blocos so com numeros (eixos de graficos) e
    trechos de sumario;
  - monografias: indexadas a partir do primeiro capitulo tecnico (INICIO_TEXTO);
  - todos os documentos: indexados ate o titulo da lista de referencias,
    exclusive -- a lista e os anexos que a seguem ficam fora. Referencias sao
    titulos de artigos: parecem pertinentes a qualquer consulta e "sustentariam"
    qualquer item na verificacao lexical, sem afirmar nada;
  - siglas corrompidas pela fonte de um dos PDFs (CKd, iRiS) restauradas.
  O que foi descartado, por documento, fica registrado em meta.json.

DIVISAO EM TRECHOS. A metodologia preve divisao por secao. Deteccao de secao em
PDF generico nao e confiavel, entao os blocos de texto sao agrupados ate cerca de
1.000 caracteres, sem cruzar blocos que parecem titulo. Cada trecho guarda o
documento e a pagina de origem, para a citacao.

JANELA DO MODELO DE EMBEDDINGS. O modelo le no maximo 128 tokens
(max_seq_length); o que passa disso e cortado sem aviso. Um trecho de 1.000
caracteres tem cerca do dobro. Por isso cada trecho e representado por JANELAS
DESLIZANTES de frases inteiras: a partir de cada frase, a maior sequencia que cabe
nesse limite, medida com o proprio tokenizador do modelo. Janelas contidas na
anterior sao descartadas, o que evita janelas curtas no fim do trecho -- uma
frase generica sozinha ("A doenca renal cronica e uma condicao comum") casava com
qualquer consulta. A similaridade de um trecho com a consulta e a maior entre as
de suas janelas. O modelo de linguagem continua recebendo o trecho inteiro.

Uso:
    python indexar_base.py
    python indexar_base.py --recuperador tfidf   # SO PARA TESTE; ver abaixo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config as cfg

PASTA_PDF = cfg.RAIZ / "base_conhecimento"
SAIDA = cfg.SAIDA / "base"
TAM_TRECHO = 1000
MIN_TRECHO = 200

# Monografias: pagina do PDF (a primeira e 1) em que comeca o texto tecnico. Antes
# dela: capa, ficha catalografica, agradecimentos, epigrafe, resumos, listas e
# sumario. Conferido pagina a pagina.
INICIO_TEXTO = {
    "carvalho2024": 7,    # "1 Introducao"
    "pessoa2025": 18,     # "2. O Rim e a DRC"; o cap. 1 e o relatorio de estagio
    "ruberti2022": 21,    # "1 INTRODUCAO"
}

# Titulo da lista de referencias, sozinho na linha. O texto e indexado ate ele.
TITULO_REFERENCIAS = re.compile(
    r"^(\d+(\.\d+)*\.?\s*)?(references|refer[eê]ncias(\s+bibliogr[aá]ficas)?|"
    r"bibliografia|literature\s+cited)$", re.IGNORECASE)

# Uma das fontes de sparkes2016 codifica D, I e O maiusculos como minusculos
# ("CKd" aparece 116 vezes). Siglas afetadas, restauradas:
CORRECOES_FONTE = {"CKd": "CKD", "iRiS": "IRIS", "SdMA": "SDMA", "EPo": "EPO",
                   "ACViM": "ACVIM", "NSAid": "NSAID", "iSFM": "ISFM",
                   "KdoQi": "KDOQI"}
_RE_FONTE = re.compile(r"\b(" + "|".join(CORRECOES_FONTE) + r")(?=\d|\b)")

REPETICAO_MIN = 0.4       # cabecalho/rodape: em >= 40% das paginas (e >= 3)
_RE_NUMERICO = re.compile(r"[\d\s.,;:%<>≤≥=±*/()\[\]+\-–—'’]+")
_RE_ENTRADA_SUMARIO = re.compile(r"[A-Za-zÀ-ÿ)]\s+\d{1,3}\s*(?:[<–—]|$)")
_RE_PONTILHADO = re.compile(r"\.{5,}|…{2,}")
_RE_FRASE = re.compile(r"(?<=[.!?;])\s+")

# Palavras vazias (artigos, preposicoes, conjuncoes...) da busca lexical: as
# listas do projeto Snowball para portugues e ingles, na integra
# (snowballstem.org; algorithms/portuguese/stop.txt e english/stop.txt).
# Sem elas, "ou" ou "the" contavam como palavra em comum com a consulta.
_VAZIAS_PT = """
    de a o que e do da em um para com não uma os no se na por mais as dos
    como mas ao ele das à seu sua ou quando muito nos já eu também só pelo
    pela até isso ela entre depois sem mesmo aos seus quem nas me esse eles
    você essa num nem suas meu às minha numa pelos elas qual nós lhe deles
    essas esses pelas este dele tu te vocês vos lhes meus minhas teu tua
    teus tuas nosso nossa nossos nossas dela delas esta estes estas aquele
    aquela aqueles aquelas isto aquilo estou está estamos estão estive
    esteve estivemos estiveram estava estávamos estavam estivera estivéramos
    esteja estejamos estejam estivesse estivéssemos estivessem estiver
    estivermos estiverem hei há havemos hão houve houvemos houveram houvera
    houvéramos haja hajamos hajam houvesse houvéssemos houvessem houver
    houvermos houverem houverei houverá houveremos houverão houveria
    houveríamos houveriam sou somos são era éramos eram fui foi fomos foram
    fora fôramos seja sejamos sejam fosse fôssemos fossem for formos forem
    serei será seremos serão seria seríamos seriam tenho tem temos tém tinha
    tínhamos tinham tive teve tivemos tiveram tivera tivéramos tenha
    tenhamos tenham tivesse tivéssemos tivessem tiver tivermos tiverem terei
    terá teremos terão teria teríamos teriam
"""
_VAZIAS_EN = """
    i me my myself we our ours ourselves you your yours yourself yourselves
    he him his himself she her hers herself it its itself they them their
    theirs themselves what which who whom this that these those am is are
    was were be been being have has had having do does did doing would
    should could ought i'm you're he's she's it's we're they're i've you've
    we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll
    we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't
    don't didn't won't wouldn't shan't shouldn't can't cannot couldn't
    mustn't let's that's who's what's here's there's when's where's why's
    how's a an the and but if or because as until while of at by for with
    about against between into through during before after above below to
    from up down in out on off over under again further then once here there
    when where why how all any both each few more most other some such no
    nor not only own same so than too very
"""


def normalizar(t: str) -> str:
    """Minusculas, sem acento: usado na verificacao lexical das citacoes."""
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


PALAVRAS_VAZIAS = sorted({normalizar(w) for w in (_VAZIAS_PT + _VAZIAS_EN).split()})


def _parece_titulo(linha: str) -> bool:
    """Linha curta, sem pontuacao final, com a maioria das palavras em maiuscula."""
    l = linha.strip()
    if not l or len(l) > 90 or l.endswith((".", ",", ";", ":")):
        return False
    return bool(re.match(r"^(\d+(\.\d+)*\.?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ]", l)) and \
        (l.isupper() or sum(w[:1].isupper() for w in l.split()) >= len(l.split()) / 2)


def _parece_sumario(txt: str) -> bool:
    """Entradas "titulo + numero de pagina" ou linhas pontilhadas."""
    return (len(_RE_ENTRADA_SUMARIO.findall(txt)) >= 4
            or len(_RE_PONTILHADO.findall(txt)) >= 2)


def _chave(txt: str) -> str:
    """Forma do bloco sem numeros: o mesmo cabecalho em paginas diferentes coincide."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", txt.lower())).strip()


def _empacotar(pecas: list[str], cabe) -> list[str]:
    """Junta pecas consecutivas enquanto o resultado couber."""
    grupos, atual = [], ""
    for p in pecas:
        candidato = f"{atual} {p}".strip()
        if atual and not cabe(candidato):
            grupos.append(atual)
            atual = p
        else:
            atual = candidato
    if atual:
        grupos.append(atual)
    return grupos


def dividir(txt: str, cabe) -> list[str]:
    """Divide o texto em partes que cabem, por frases inteiras. Frase que nao cabe
    sozinha e dividida por palavras."""
    pecas = []
    for frase in _RE_FRASE.split(txt):
        pecas.extend([frase] if cabe(frase) else _empacotar(frase.split(), cabe))
    return _empacotar(pecas, cabe)


def deslizantes(txt: str, cabe) -> list[str]:
    """Janelas deslizantes de frases: a partir de cada frase, a maior sequencia
    que cabe. A janela contida na anterior (mesmo fim) e descartada; toda frase
    fica em ao menos uma janela."""
    frases = []
    for f in _RE_FRASE.split(txt):
        frases.extend([f] if cabe(f) else _empacotar(f.split(), cabe))
    out, fim_anterior = [], -1
    for i in range(len(frases)):
        janela, j = frases[i], i
        while j + 1 < len(frases) and cabe(f"{janela} {frases[j + 1]}"):
            j += 1
            janela = f"{janela} {frases[j]}"
        if j > fim_anterior:
            out.append(janela)
            fim_anterior = j
        if j == len(frases) - 1:
            break
    return out


def _blocos(pdf: Path) -> list[tuple[int, list[str]]]:
    """Blocos de texto horizontal, na ordem do arquivo, a partir de INICIO_TEXTO.
    Cada bloco e a lista de suas linhas."""
    import pymupdf
    flags = pymupdf.TEXTFLAGS_BLOCKS & ~pymupdf.TEXT_PRESERVE_LIGATURES  # "ﬁ" -> "fi"
    inicio = INICIO_TEXTO.get(pdf.stem, 1)
    out = []
    with pymupdf.open(pdf) as doc:
        for n, pagina in enumerate(doc, 1):
            if n < inicio:
                continue
            for b in pagina.get_text("dict", flags=flags)["blocks"]:
                if b["type"] != 0:
                    continue
                linhas = ["".join(s["text"] for s in l["spans"]) for l in b["lines"]
                          if abs(l["dir"][1]) < 1e-3]      # descarta texto girado
                linhas = [l for l in linhas if l.strip()]
                if linhas:
                    out.append((n, linhas))
    return out


def extrair(pdf: Path) -> tuple[list[dict], dict]:
    """Trechos do documento e o registro do que foi descartado."""
    blocos = _blocos(pdf)
    reg = {"pagina_inicial": INICIO_TEXTO.get(pdf.stem, 1),
           "referencias_a_partir_da_pagina": None,
           "descartados": Counter()}

    # 1. corte no titulo da lista de referencias (o resto do bloco, e tudo depois)
    for i, (n, linhas) in enumerate(blocos):
        pos = next((j for j, l in enumerate(linhas)
                    if TITULO_REFERENCIAS.match(l.strip())), None)
        if pos is not None:
            reg["referencias_a_partir_da_pagina"] = n
            reg["descartados"]["referencias e anexos (blocos)"] = len(blocos) - i
            blocos = blocos[:i] + ([(n, linhas[:pos])] if pos else [])
            break

    # 2. cabecalhos e rodapes: blocos curtos repetidos em muitas paginas
    paginas = defaultdict(set)
    for n, linhas in blocos:
        txt = "\n".join(linhas)
        if len(txt) <= 200:
            paginas[_chave(txt)].add(n)
    n_pag = len({n for n, _ in blocos}) or 1
    repetidos = {k for k, p in paginas.items()
                 if len(p) >= 3 and len(p) >= REPETICAO_MIN * n_pag}
    reg["cabecalhos_rodapes"] = sorted(repetidos)

    trechos, atual, pag_ini = [], "", None

    def fechar():
        nonlocal atual, pag_ini
        if len(atual.strip()) >= 50:
            trechos.append({"texto": atual.strip(), "pagina": pag_ini})
        atual, pag_ini = "", None

    for n, linhas in blocos:
        bruto = "\n".join(linhas)
        if _chave(bruto) in repetidos:
            reg["descartados"]["cabecalho ou rodape"] += 1
            continue
        if _RE_NUMERICO.fullmatch(bruto):
            reg["descartados"]["so numeros"] += 1
            continue
        txt = re.sub(r"(?<=[a-zà-ÿ])-\n(?=[a-zà-ÿ])", "", bruto)   # hifenizacao
        txt = re.sub(r"\s*\n\s*", " ", txt).strip()
        txt = _RE_FONTE.sub(lambda m: CORRECOES_FONTE[m.group(1)], txt)
        for parte in dividir(txt, lambda s: len(s) <= TAM_TRECHO):
            if _parece_titulo(parte):
                if len(atual) >= MIN_TRECHO:
                    fechar()
            elif len(atual) + len(parte) > TAM_TRECHO and len(atual) >= MIN_TRECHO:
                fechar()
            if pag_ini is None:
                pag_ini = n
            if re.search(r"[a-zà-ÿ]-$", atual) and re.match(r"[a-zà-ÿ]", parte):
                atual = atual[:-1] + parte          # palavra partida entre blocos
            else:
                atual += " " + parte
    fechar()

    antes = len(trechos)
    trechos = [t for t in trechos if not _parece_sumario(t["texto"])]
    reg["descartados"]["trechos de sumario"] = antes - len(trechos)
    for t in trechos:
        t["documento"] = pdf.stem
    reg["descartados"] = {k: v for k, v in reg["descartados"].items() if v}
    reg["trechos"] = len(trechos)
    return trechos, reg


def carregar_modelo(nome: str | None = None):
    """Modelo de sentence embeddings (config.MODELO_EMBEDDINGS)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nome or cfg.MODELO_EMBEDDINGS)


def contador_tokens(modelo):
    """Tokens que o modelo de embeddings le, com os especiais de inicio e fim.

    So conta: medir candidatos acima do limite faz parte da divisao, e nenhum texto
    acima dele vai para o modelo (janelas() confere). Por isso verbose=False, que
    cala o aviso do tokenizador sobre sequencias longas."""
    tok = modelo.tokenizer
    return lambda s: len(tok(s, add_special_tokens=True, verbose=False)["input_ids"])


def janelas(trechos: list[dict], modelo=None) -> list[dict]:
    """Janelas de cada trecho. Sem modelo (TF-IDF, so teste): o trecho inteiro."""
    if modelo is None:
        return [{"trecho": t["id"], "texto": t["texto"], "tokens": None}
                for t in trechos]
    n_tokens, limite = contador_tokens(modelo), modelo.max_seq_length
    out = []
    for t in trechos:
        for j in deslizantes(t["texto"], lambda s: n_tokens(s) <= limite):
            out.append({"trecho": t["id"], "texto": j, "tokens": n_tokens(j)})
    excedentes = [j for j in out if j["tokens"] > limite]
    if excedentes:                       # nunca truncar em silencio
        j = excedentes[0]
        raise SystemExit(f"{len(excedentes)} janela(s) acima de {limite} tokens "
                         f"(trecho {j['trecho']}: {j['texto'][:80]!r}). Verifique "
                         f"o texto extraido desse documento.")
    return out


def vetorizar(textos: list[str], recuperador: str, modelo=None):
    """Vetores normalizados dos textos: embeddings (metodologia) ou TF-IDF (teste e busca lexical)."""
    if recuperador == "embeddings":
        modelo = modelo or carregar_modelo()
        return modelo.encode(textos, normalize_embeddings=True, batch_size=32,
                             show_progress_bar=True).astype("float32"), None
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(preprocessor=normalizar, sublinear_tf=True,
                          stop_words=PALAVRAS_VAZIAS)
    m = vec.fit_transform(textos).toarray().astype("float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True).clip(1e-9)
    return m, vec


def main() -> None:
    import faiss

    ap = argparse.ArgumentParser(description="Indexa a base de conhecimento.")
    ap.add_argument("--recuperador", choices=["embeddings", "tfidf"],
                    default="embeddings",
                    help="embeddings (padrao, o da metodologia) ou tfidf, "
                         "apenas para teste em ambiente sem o modelo")
    args = ap.parse_args()

    pdfs = sorted(PASTA_PDF.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Nenhum PDF em {PASTA_PDF}. Crie a pasta e coloque os "
                         f"documentos da base de conhecimento.")
    SAIDA.mkdir(parents=True, exist_ok=True)

    print(f"[1] extraindo texto de {len(pdfs)} documentos")
    print(f"  {'documento':<18}{'trechos':>8}  {'paginas':<9} descartado")
    trechos, registro = [], {}
    for p in pdfs:
        t, reg = extrair(p)
        registro[p.stem] = reg
        fim = reg["referencias_a_partir_da_pagina"]
        pags = f"{reg['pagina_inicial']}-{fim if fim else 'fim'}"
        desc = ", ".join(f"{v} {k}" for k, v in reg["descartados"].items()) or "-"
        print(f"  {p.stem:<18}{len(t):>8}  {pags:<9} {desc}")
        trechos.extend(t)
    for i, t in enumerate(trechos):
        t["id"] = i

    if args.recuperador != "embeddings":
        print("\n  ATENCAO: recuperador TF-IDF, busca lexical, SO PARA TESTE. Nao e o")
        print("  metodo da metodologia e nao deve gerar os resultados do trabalho.\n")
        modelo = None
    else:
        modelo = carregar_modelo()
    jan = janelas(trechos, modelo)
    print(f"[2] vetorizando {len(jan)} janelas de {len(trechos)} trechos "
          f"({args.recuperador})")
    m, _ = vetorizar([j["texto"] for j in jan], args.recuperador, modelo)

    indice = faiss.IndexFlatIP(m.shape[1])     # produto interno de vetores
    indice.add(m)                              # normalizados = cosseno
    faiss.write_index(indice, str(SAIDA / "indice.faiss"))
    with open(SAIDA / "trechos.jsonl", "w", encoding="utf-8") as fh:
        for t in trechos:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(SAIDA / "janelas.jsonl", "w", encoding="utf-8") as fh:
        for j in jan:
            fh.write(json.dumps(j, ensure_ascii=False) + "\n")

    tam = [len(t["texto"]) for t in trechos]
    por_trecho = Counter(j["trecho"] for j in jan)
    tokens = [j["tokens"] for j in jan if j["tokens"] is not None]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in pdfs}
    (SAIDA / "meta.json").write_text(json.dumps({
        "recuperador": args.recuperador,
        "modelo_embeddings": cfg.MODELO_EMBEDDINGS if modelo else None,
        "limite_tokens": modelo.max_seq_length if modelo else None,
        "n_trechos": len(trechos), "n_janelas": len(jan),
        "janelas_por_trecho": {"mediana": float(np.median(list(por_trecho.values()))),
                               "maximo": max(por_trecho.values())},
        "tokens_por_janela": ({"mediana": float(np.median(tokens)), "maximo": max(tokens)}
                              if tokens else None),
        "caracteres_por_trecho": {"mediana": float(np.median(tam)),
                                  "minimo": min(tam), "maximo": max(tam)},
        "tam_trecho": TAM_TRECHO, "documentos": hashes, "extracao": registro,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[3] indice gravado em {SAIDA}")
    print(f"    trechos: {len(trechos)} | tamanho mediano {int(np.median(tam))} "
          f"caracteres [{min(tam)}-{max(tam)}]")
    if tokens:
        print(f"    janelas: {len(jan)} | ate {max(tokens)} tokens cada (limite do "
              f"modelo: {modelo.max_seq_length}) | mediana de "
              f"{np.median(list(por_trecho.values())):.0f} por trecho")
    print("    O que foi descartado de cada documento esta em meta.json.")


if __name__ == "__main__":
    main()
