"""
Camada interpretativa: segunda iteracao da traducao para o veterinario.

POR QUE EXISTE. Os textos da primeira versao explicam o MODELO, e nao o caso:
traduzem um a um os cinco fatores do SHAP, inclusive a falta de exames ("a
ausencia da creatinina aumentou a probabilidade do estagio 4") e fatores sem
sentido clinico isolado ("a ausencia de bocio"). Para o modelo, a falta de um
exame e informacao, e esse e o achado central do trabalho; para o veterinario,
e um pedido de exame. A segunda versao escreve no nivel da consulta: o estagio
e quao segura e a estimativa, os dados clinicos em que ela se apoia, o que pedir
e que comorbidades considerar.

O QUE MUDA, com o desenho fixado antes de qualquer texto da v2 ser gerado:

  1. O CODIGO, e nao o modelo de linguagem, prepara um resumo da ficha:
     - o termo da escala verbal vai ja calculado junto de cada numero;
     - ate tres dados clinicos (exames com valor, apresentacao, idade, achados
       presentes), com o efeito ja escrito ("sustenta" ou "pesa contra o
       estagio estimado"). Ficam fora sexo, estado reprodutivo, raca, achados
       negativos ou nao registrados e exames ausentes;
     - exame ausente vira incerteza e recomendacao, nunca fator;
     - so se recomenda exame com ganho esperado de ao menos LIMIAR_GANHO_BITS;
     - UPC e pressao que faltam recebem o lembrete do subestadiamento IRIS.
  2. Prompt novo. O resto nao muda: mesmas fichas, mesmo modelo e parametros. O
     RAG e o modelo puro da execucao final sao reaproveitados, sem nova chamada.
  3. Verificacao ampliada, aplicada as DUAS versoes com as mesmas regras.
  4. Leitura comparada: os 10 gatos da leitura qualitativa, com as duas versoes
     lado a lado, em ordem sorteada; a chave fica em arquivo separado.

A versao 1 continua reportada; a 2 e uma segunda iteracao, declarada como tal.

Um piloto com cinco gatos, fora da amostra da leitura, apontou tres defeitos,
corrigidos antes da execucao completa: a verificacao nao reconhecia "o estagio
estimado e IRIS 3" como mencao ao estagio; com "creatinina_medida": "sim" no
resumo, o modelo afirmou que a estimativa era "mais segura"; e o texto agrupou
duas comorbidades sob um termo que so servia a uma ("muito provaveis sao ...
(66,8%)"). Desde entao a mencao "IRIS 3" conta, os indicadores so vao ao modelo
quando pedem uma frase (para_modelo), a regra 6 proibe agrupar e a verificacao
le o termo no plural. Os nomes dos exames vao em minusculas no resumo.

Entrada: fichas e respostas da v1 da execucao final mais recente
(entregas/rodada_final_*/relatorios/camada/), nunca de saida/.

Uso:
    python camada_v2.py --simular   # sem API: grava resumos e prompts, e aplica a
                                    # verificacao nova aos textos da v1
    python camada_v2.py --piloto    # 5 gatos fora da amostra da leitura, com API
    python camada_v2.py             # os 123; pela rodada: bash rodada_camada_v2.sh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

import config as cfg
from camada_interpretativa import (LLM, ROMANOS, _RE_GRAVIDADE,
                                   selecionar_qualitativos, versao_declarada)
from indexar_base import normalizar

SAIDA = cfg.SAIDA_RELATORIOS / "camada_v2"

# ---------------------------------------------------------------------------
# Parametros do desenho -- fixados antes de qualquer texto da v2
# ---------------------------------------------------------------------------

# Exame so e recomendado se reduzir a duvida sobre o estagio em ao menos 0,1 bit:
# um vigesimo da duvida maxima entre quatro estagios (2 bits), e abaixo do menor
# ganho da bioquimica renal nos gatos que nao a tem (0,12 bit), de modo que o
# exame que define o estagio nunca fica de fora. Na v1, qualquer ganho acima de
# zero contava, e o texto recomendava exames que mudariam a duvida em 0,002 bit.
LIMIAR_GANHO_BITS = 0.10
MAX_DADOS_CLINICOS = 3          # os de maior peso, depois do filtro clinico
LIMITE_PALAVRAS = 130           # v1: 180
# Piloto: fora da amostra da leitura qualitativa e dos pilotos da v1; cobre os
# quatro estagios, com e sem creatinina, e os erros vistos na v1.
PILOTO = [192, 326, 345, 206, 177]
# A diretriz IRIS subestadia por proteinuria (UPC) e pressao arterial: esses
# exames seguem indicados mesmo quando pouco mudam o estagio.
EXAMES_SUBESTADIAMENTO = ["UPC", "Pressão arterial"]

# ---------------------------------------------------------------------------
# Classificacao das variaveis da ficha. Variavel nova sem classificacao para a
# execucao: nada entra no texto sem decisao explicita.
# ---------------------------------------------------------------------------

# Exames: com valor, sao dado clinico; "ausente", viram incerteza e recomendacao.
EXAMES = {
    "Creatinina": "creatinina de {}",
    "Ureia": "ureia de {}",
    "Fósforo": "fósforo de {}",
    "Densidade urinária": "densidade urinária de {}",
    "Proteinúria (fita)": "proteinúria de {} na fita",
    "UPC": "UPC de {}",
    "Pressão arterial": "pressão arterial de {}",
}
APRESENTACAO = {
    "sinais clínicos": "apresentação com sinais clínicos",
    "check-up geriátrico": "atendimento de check-up geriátrico",
    "perfil pré-anestésico": "atendimento para perfil pré-anestésico",
}
MOTIVO_SEM_INFORMACAO = {"outro", "não registrado", "ausente"}
# Achados so entram quando PRESENTES. Achado negativo ou nao registrado so faria
# sentido com o mecanismo que o explica (a "ausencia de bocio", por exemplo), e o
# mecanismo nao esta na ficha.
ACHADOS = {
    "Bócio": "bócio", "Sopro cardíaco": "sopro cardíaco",
    "Ritmo de galope": "ritmo de galope", "Arritmia": "arritmia",
    "Anestesia prévia": "anestesia prévia",
}
FARMACOS = {"NSAID": "exposição prévia a fármaco anti-inflamatório não esteroidal"}
NEGATIVOS = {"não", "não registrado", "nenhum registrado", "não testado", "ausente"}
# Descrevem o animal ou o registro, nao o quadro clinico.
NAO_CLINICOS = {"Sexo", "Estado reprodutivo", "Raça"}
EFEITO = {"a favor": "sustenta o estágio estimado",
          "contra": "pesa contra o estágio estimado"}

# ---------------------------------------------------------------------------
# Instrucoes ao modelo
# ---------------------------------------------------------------------------

SISTEMA_V2 = """Você é a camada de verbalização de um sistema de apoio à decisão \
clínica para doença renal crônica felina. Recebe um resumo já interpretado da \
saída do sistema e escreve, para o médico veterinário, um texto curto e prático: \
o estágio estimado e quão segura é a estimativa, os dados clínicos em que ela se \
apoia, o que pedir para esclarecer e que comorbidades considerar.

Como ler o resumo:
- "estagio_estimado" e "probabilidade": o estágio IRIS estimado, com o termo e o \
número já calculados. "segundo_estagio", quando existe, é o outro estágio com \
probabilidade de ao menos 20%.
- "dados_clinicos": até três dados do paciente que mais pesaram na estimativa, \
cada um com o efeito já escrito.
- "creatinina_medida": aparece, com o valor "não", só quando a creatinina — o \
marcador que define o estágio IRIS — não foi medida.
- "falta_de_exames_pesou": aparece, com o valor "sim", só quando a falta de \
exames pesou na estimativa.
- "exames_recomendados": exames que mais reduziriam a dúvida sobre o estágio, do \
mais útil para o menos útil. "exames_para_subestadiamento": exames que faltam e \
que a diretriz IRIS usa no subestadiamento.
- "comorbidades": comorbidades com probabilidade de ao menos 20%, com o termo e o \
número já calculados.

Regras:
1. Use somente o que está no resumo. Não acrescente valores, exames, diagnósticos, \
condições nem recomendações.
2. Comece pelo estágio estimado, com o termo e o número exatamente como no resumo. \
Se houver "segundo_estagio", mencione-o na mesma frase, também com termo e número.
3. Em uma frase, diga em que dados clínicos a estimativa se apoia, com os efeitos \
como estão no resumo. Não cite nenhum outro dado do paciente. Se a lista estiver \
vazia, omita essa frase.
4. Se o resumo trouxer "creatinina_medida", diga que a estimativa foi feita sem \
creatinina, o marcador que define o estágio IRIS, e que por isso é menos segura. \
Se trouxer "falta_de_exames_pesou", diga que parte da estimativa reflete o \
padrão de registro dos atendimentos usados para desenvolver o sistema, e não \
achados do animal. Se esses campos não estiverem no resumo, não comente a \
segurança da estimativa nem a falta de exames. Nunca escreva que a falta de um \
exame aumentou, diminuiu, sustentou ou definiu a probabilidade de um estágio.
5. Recomende os exames de "exames_recomendados", na ordem, dizendo que o primeiro \
é o que mais esclareceria o estágio. Se a lista estiver vazia, diga que exames \
adicionais pouco mudariam o estágio estimado. Se "exames_para_subestadiamento" \
não estiver vazia, diga que esses exames seguem indicados para o \
subestadiamento. Não recomende nenhum outro exame. Escreva os nomes dos exames \
como estão no resumo.
6. Cite cada comorbidade da lista com o seu próprio termo e número, exatamente \
como no resumo; não agrupe comorbidades sob um termo só. Se a lista estiver \
vazia, diga que as comorbidades avaliadas são pouco prováveis.
7. Termine com uma frase dizendo que o resultado é um apoio à avaliação do médico \
veterinário e não a substitui. Nunca o apresente como diagnóstico.
8. Não mencione modelos, algoritmos, SHAP ou termos de aprendizado de máquina.
9. Escreva em português, em texto corrido, com no máximo {LIMITE} palavras.

Responda em JSON, com exatamente estes campos:
{"texto": "...",
 "estagio_citado": <inteiro>,
 "exames_recomendados_citados": [<nomes exatamente como no resumo>]}""".replace(
    "{LIMITE}", str(LIMITE_PALAVRAS))


# ---------------------------------------------------------------------------
# Resumo da ficha: tudo o que o texto pode dizer, decidido aqui
# ---------------------------------------------------------------------------

def _pct(p: float) -> float:
    """Probabilidade (0 a 1) em porcentagem com uma casa."""
    return round(100 * p, 1)


def escala(x: float) -> str:
    """Termo da escala verbal (a mesma da v1) para uma porcentagem."""
    if x > 70:
        return "muito provável"
    if x >= 40:
        return "provável"
    if x >= 20:
        return "possível"
    return "pouco provável"


def termo(p: float) -> str:
    """Probabilidade com o termo da escala: 'provável (67,8%)'."""
    x = _pct(p)
    return f"{escala(x)} ({x:.1f}%)".replace(".", ",")


def _numero_pt(variavel: str, valor: str) -> str:
    """Virgula decimal; a densidade urinaria fica como se escreve na clinica."""
    if variavel == "Densidade urinária":
        return valor
    return re.sub(r"(\d)\.(\d)", r"\1,\2", valor)


def dado_clinico(x: dict) -> str | None:
    """O fator em linguagem clinica, ou None quando fica fora do texto."""
    v, val = x["variavel"], x["valor"]
    if v in NAO_CLINICOS:
        return None
    if v in EXAMES:
        return None if val == "ausente" else EXAMES[v].format(_numero_pt(v, val))
    if v == "Idade":
        return None if val == "ausente" else f"idade de {val}"
    if v == "Motivo da apresentação":
        if val in MOTIVO_SEM_INFORMACAO:
            return None
        if val not in APRESENTACAO:
            raise SystemExit(f"Motivo da apresentacao sem traducao: {val!r}. "
                             f"Acrescente-o em APRESENTACAO (camada_v2.py).")
        return APRESENTACAO[val]
    if v in ACHADOS:
        if val in NEGATIVOS:
            return None
        if val != "sim":
            raise SystemExit(f"Valor inesperado para {v}: {val!r}.")
        return ACHADOS[v]
    if v == "Fármacos nefrotóxicos":
        if val in NEGATIVOS:
            return None
        if val not in FARMACOS:
            raise SystemExit(f"Farmaco sem traducao: {val!r}. Acrescente-o em "
                             f"FARMACOS (camada_v2.py).")
        return FARMACOS[val]
    raise SystemExit(f"Variavel sem classificacao na v2: {v!r}. Classifique-a no "
                     f"inicio de camada_v2.py antes de gerar qualquer texto.")


def resumo(f: dict) -> dict:
    """O que o modelo de linguagem recebe na v2."""
    e = f["estagio_predito"]
    probs = {int(k): p for k, p in f["probabilidades"].items()}
    outros = sorted(((k, p) for k, p in probs.items() if k != e), key=lambda t: -t[1])
    dados = [{"dado": d, "efeito": EFEITO[x["direcao"]]}
             for x in f["fatores"] if (d := dado_clinico(x)) is not None]
    ausentes = [x["variavel"] for x in f["fatores"]
                if x["variavel"] in EXAMES and x["valor"] == "ausente"]
    rec = [x["exame"] for x in f["recomendacao_exames"]
           if x["ganho_bits"] >= LIMIAR_GANHO_BITS]
    r = {"estagio_estimado": e, "probabilidade": termo(probs[e])}
    if outros and _pct(outros[0][1]) >= 20:
        r["segundo_estagio"] = {"estagio": outros[0][0],
                                "probabilidade": termo(outros[0][1])}
    r.update({
        "dados_clinicos": dados[:MAX_DADOS_CLINICOS],
        # dado do atendimento (a creatinina foi medida?), e nao do rotulo: o
        # estagio registrado continua fora de tudo o que o modelo le
        "creatinina_medida": "não" if f["_avaliacao"]["sem_creatinina"] else "sim",
        "falta_de_exames_pesou": "sim" if ausentes else "não",
        "exames_recomendados": rec,
        "exames_para_subestadiamento": [x for x in EXAMES_SUBESTADIAMENTO
                                        if x in f["exames_faltantes"] and x not in rec],
        "comorbidades": [{"nome": c["nome"], "probabilidade": termo(c["probabilidade"])}
                         for c in f["comorbidades"] if _pct(c["probabilidade"]) >= 20],
    })
    return r


# nome do exame no texto (a ficha usa maiuscula, como rotulo)
NOME_NO_TEXTO = {"Bioquímica renal": "bioquímica renal", "Urinálise": "urinálise",
                 "UPC": "UPC", "Pressão arterial": "pressão arterial"}


def para_modelo(rs: dict) -> dict:
    """O resumo como o modelo o recebe. Os dois indicadores so vao quando pedem
    uma frase: no piloto, com "creatinina_medida": "sim" no resumo, o modelo
    escreveu que a medida "torna a estimativa mais segura" -- afirmacao que o
    resumo nao sustenta."""
    m = {k: v for k, v in rs.items() if k not in ("creatinina_medida", "falta_de_exames_pesou",
                                                  "exames_recomendados",
                                                  "exames_para_subestadiamento", "comorbidades")}
    if rs["creatinina_medida"] == "não":
        m["creatinina_medida"] = "não"
    if rs["falta_de_exames_pesou"] == "sim":
        m["falta_de_exames_pesou"] = "sim"
    m["exames_recomendados"] = [NOME_NO_TEXTO[e] for e in rs["exames_recomendados"]]
    m["exames_para_subestadiamento"] = [NOME_NO_TEXTO[e] for e in rs["exames_para_subestadiamento"]]
    m["comorbidades"] = rs["comorbidades"]
    return m


# ---------------------------------------------------------------------------
# Verificacao automatica -- as mesmas regras para as duas versoes
# ---------------------------------------------------------------------------

# como o texto pode nomear cada exame recomendavel (normalizado, sem acento)
APELIDOS_EXAME = {
    "Bioquímica renal": ["bioquimica", "creatinina", "ureia", "fosforo"],
    "Urinálise": ["urinalise", "exame de urina", "densidade urinaria", "proteinuria"],
    "UPC": ["upc", "relacao proteina"],
    "Pressão arterial": ["pressao arterial", "pressao sistolica"],
}
# dados do paciente que o texto pode citar (normalizados)
TERMOS_PACIENTE = [
    "creatinina", "ureia", "fosforo", "densidade urinaria", "proteinuria", "upc",
    "pressao arterial", "idade", "sinais clinicos", "check-up", "pre-anestesico",
    "bocio", "sopro", "ritmo de galope", "arritmia", "anestesia",
    "anti-inflamatorio", "farmaco", "bioquimica", "urinalise",
]
_RE_NAO_CLINICO = re.compile(
    r"\bsexo\b|\bmacho\b|\bfemea\b|estado reprodutivo|castrad|\binteir[oa]\b|"
    r"\braca\b|crossbreed|purebreed|neutered|\bentire\b")
# termo da escala junto do numero; "o mais provavel" e comparacao, nao escala
_RE_TERMO_NUMERO = re.compile(
    r"(?<!mais )(muito provavel|pouco provavel|provavel|possivel)\W{0,3}"
    r"(?:\(|com probabilidade de |com |de )?(\d{1,3}(?:[.,]\d+)?)\s*%")
# termo no plural aplicado a uma lista: "muito provaveis sao cistite (84,5%) e
# doenca periodontal (66,8%)" -- cada numero da lista precisa caber no termo
_RE_TERMO_PLURAL = re.compile(r"(?<!mais )(muito provaveis|pouco provaveis|provaveis|possiveis)")
_RE_QUALQUER_TERMO = re.compile(r"provave(?:l|is)|possive(?:l|is)|[.;]")
_SINGULAR = {"muito provaveis": "muito provavel", "pouco provaveis": "pouco provavel",
             "provaveis": "provavel", "possiveis": "possivel"}
# estagio mencionado: "estagio 3", "estagio IRIS 3" ou "IRIS 3"
_RE_ESTAGIO_V2 = re.compile(r"(?:est[aá]gios?|estadiamento)\s+(?:iris\s+)?(\d|iv|i{1,3})\b|"
                            r"\biris\s+(\d|iv|i{1,3})\b", re.IGNORECASE)
# falta de EXAME: a palavra de ausencia ligada ao nome do exame ("ausencia da
# creatinina", "falta desse exame", "a pressao arterial nao foi avaliada"). A
# ausencia de um achado ("ausencia de bocio") nao e falta de exame, e "sem exames
# laboratoriais", sem artigo, e contexto, nao causa.
_EXAME = (r"(?:creatinina|ureia|fosforo|densidade urinaria|proteinuria|upc|"
          r"pressao arterial|bioquimica|urinalise)")
_ART = r"(?:(?:d[aoe]s?|desse|deste|dessa|desta|de|o|a|os|as)\s+)"
_RE_AUSENCIA = re.compile(
    r"\b(?:falta|faltam|faltou|ausencia)\s+(?:"
    + _ART + r"?"
    # um item coordenado antes do exame: "ausencia do estado reprodutivo e da
    # densidade urinaria" (mas nao "ausencia de bocio e a densidade urinaria")
    + r"(?:[a-z\- ]{1,25}?\s+e\s+(?:(?:d[aoe]s?|de)\s+))?"
    + r"(?:exames?\s+" + _ART + r"?)?" + _EXAME
    + r"|" + _ART + r"exames?\b)"
    + r"|" + _EXAME + r"\s+(?:ausentes?|nao (?:foi|foram) (?:realizad|medid|aferid|avaliad))")
_RE_SEM_EXAME = re.compile(r"\bsem\s+(?:[ao]s?\s+)?(?:exames?\s+(?:de\s+)?)?" + _EXAME)
# efeito sobre a probabilidade de um estagio (o verbo que a regra 4 proibe para a
# falta de exame, e que so os dados clinicos do resumo podem ter)
_RE_EFEITO = re.compile(
    r"(?:aument|diminu|elev|reduz)\w*\s+(?:a |essa |esta |sua )?(?:probabilidade|chance)|"
    r"sustent\w*\s+(?:o|a|esse|este)\s+(?:estagio|estimativa)|pes\w*\s+contra|"
    r"favorec\w*\s+o\s+estagio")
_RE_RECOMENDA = re.compile(r"recomend|\bpeca\b|\bpecam\b|\bpedir\b|solicit|esclarec|suger|indicad")
_RE_POUCO = re.compile(r"pouco (?:mud|acrescent|contribu|alter|esclarec)")
_RE_PARADA = re.compile(r"pouco (?:mudaria|mudariam|acrescentaria|acrescentariam|contribuiria|"
                        r"contribuiriam|alteraria|alterariam)|nao (?:mudaria|acrescentaria)")
_RE_SEM = re.compile(r"\bsem\b|falt|ausen|nao (?:foi|ha|tem|houve) |nao medid")
_RE_PADRAO = re.compile(r"padrao de registro|registro dos atendimentos|padrao dos atendimentos")
_RE_APOIO = re.compile(r"apoio|auxili|complement|decis[aã]o|substitu")


def _estagios(texto: str) -> list[int]:
    """Estagios mencionados, na ordem em que aparecem."""
    out = []
    for m in _RE_ESTAGIO_V2.finditer(texto):
        g = (m.group(1) or m.group(2)).lower()
        out.append(int(g) if g.isdigit() else ROMANOS[g])
    return out


def _trocas_de_escala(tn: str) -> int:
    """Termo da escala que nao cabe no numero, no singular e no plural."""
    def fora(termo: str, numero: str) -> bool:
        return normalizar(escala(float(numero.replace(",", ".")))) != termo
    n = sum(fora(m.group(1), m.group(2)) for m in _RE_TERMO_NUMERO.finditer(tn))
    for m in _RE_TERMO_PLURAL.finditer(tn):
        fim = _RE_QUALQUER_TERMO.search(tn, m.end())
        trecho = tn[m.end():fim.start() if fim else len(tn)]
        n += sum(fora(_SINGULAR[m.group(1)], x)
                 for x in re.findall(r"(\d{1,3}(?:[.,]\d+)?)\s*%", trecho))
    return n


def _frases(texto: str) -> list[str]:
    """Frases do texto, normalizadas (minusculas, sem acento)."""
    return [normalizar(x) for x in re.split(r"(?<=[.!?;])\s+", texto) if x.strip()]


def _tem(termo: str, frase: str) -> bool:
    """Termo como palavra inteira: "idade" nao conta dentro de "probabilidade"."""
    return re.search(rf"(?<!\w){re.escape(termo)}(?!\w)", frase) is not None


def _exames_citados(frase: str) -> set[str]:
    """Exames recomendaveis citados na frase, por qualquer dos seus apelidos."""
    return {e for e, apelidos in APELIDOS_EXAME.items()
            if any(_tem(a, frase) for a in apelidos)}


def verificar(texto: str | None, rs: dict, f: dict) -> dict:
    """Medidas de um texto contra o resumo (rs) e a ficha (f) do mesmo gato."""
    if texto is None:
        return {"erro": True}
    tn = normalizar(texto)
    frases = _frases(texto)
    base_texto = normalizar(" ".join(d["dado"] for d in rs["dados_clinicos"]))
    rec = rs["exames_recomendados"]

    # estagio: mencionado; e o texto nao abre com outro
    mencoes = _estagios(texto)

    # escala verbal: termo e numero precisam concordar
    trocas = _trocas_de_escala(tn)

    # porcentagens do texto correspondem a alguma probabilidade da ficha
    probs = [100 * p for p in f["probabilidades"].values()] + \
            [100 * c["probabilidade"] for c in f["comorbidades"]]
    pcts = [float(p.replace(",", ".")) for p in re.findall(r"(\d{1,3}(?:[.,]\d+)?)\s*%", texto)]
    pct_ruins = [p for p in pcts if not any(abs(p - q) <= 1.5 for q in probs)]

    # falta de exame descrita como causa de efeito sobre a probabilidade
    ausencia_causa = sum(1 for fr in frases if _RE_AUSENCIA.search(fr) and _RE_EFEITO.search(fr))

    # efeito atribuido a dado do paciente que nao esta no resumo. O exame citado
    # como ausente ("sem creatinina") nao conta aqui: isso e a medida anterior.
    fora = set()
    for fr in frases:
        if _RE_EFEITO.search(fr):
            resto = _RE_SEM_EXAME.sub(" ", _RE_AUSENCIA.sub(" ", fr))
            fora |= {t for t in TERMOS_PACIENTE if _tem(t, resto) and not _tem(t, base_texto)}

    # recomendacao. Na mesma frase podem estar o lembrete do subestadiamento e o
    # sinal de parada: os exames citados neles nao contam como recomendados.
    na_recomendacao = set()
    for fr in frases:
        if not _RE_RECOMENDA.search(fr):
            continue
        citados = _exames_citados(fr)
        if "subestadiamento" in fr:
            citados -= set(rs["exames_para_subestadiamento"])
        if _RE_POUCO.search(fr):
            citados &= set(rec)
        na_recomendacao |= citados
    primeiro_citado = (not rec) or rec[0] in na_recomendacao

    sem_creatinina = rs["creatinina_medida"] == "não"
    incerteza_dita = any("creatinina" in fr and _RE_SEM.search(fr) for fr in frases)
    # comentario sem base no resumo: seguranca da estimativa com creatinina medida,
    # ou falta de exames quando ela nao pesou
    sem_base = ((not sem_creatinina and re.search(r"\bsegur", tn) is not None)
                or (rs["falta_de_exames_pesou"] == "não"
                    and re.search(r"falta de exames|ausencia de exames|influencia da falta|"
                                  r"padrao de registro", tn) is not None))
    return {
        "erro": False,
        "estagio_correto": rs["estagio_estimado"] in mencoes,
        "primeiro_estagio_errado": bool(mencoes) and mencoes[0] != rs["estagio_estimado"],
        "escala_trocada": trocas,
        "porcentagens_sem_correspondencia": len(pct_ruins),
        "ausencia_como_causa": ausencia_causa,
        "efeito_fora_do_resumo": len(fora),
        "fator_nao_clinico_citado": len(set(_RE_NAO_CLINICO.findall(tn))),
        "comentario_fora_do_resumo": sem_base,
        "fator_descrito_por_gravidade": sum(1 for fr in frases if _RE_GRAVIDADE.search(fr)
                                            and any(_tem(t, fr) for t in TERMOS_PACIENTE)),
        "sem_creatinina": sem_creatinina,
        "incerteza_omitida": sem_creatinina and not incerteza_dita,
        "falta_de_exames_pesou": rs["falta_de_exames_pesou"] == "sim",
        "padrao_de_registro_omitido": (rs["falta_de_exames_pesou"] == "sim"
                                       and not _RE_PADRAO.search(tn)),
        "recomendacao_fora_da_lista": len(na_recomendacao - set(rec)),
        "recomendacao_omitida": not primeiro_citado,
        "parada_omitida": (not rec) and not _RE_PARADA.search(tn),
        "subestadiamento_omitido": (bool(rs["exames_para_subestadiamento"])
                                    and "subestadiamento" not in tn),
        "linguagem_de_apoio": bool(_RE_APOIO.search(tn)),
        "palavras": len(texto.split()),
        "acima_do_limite": len(texto.split()) > LIMITE_PALAVRAS,
    }


# (medida, rotulo, tipo): "pct" = % dos textos; "cont" = textos com ao menos uma
# ocorrencia; "cond" = textos em que ocorre, entre os que a regra alcanca
MEDIDAS = [
    ("estagio_correto", "estagio estimado mencionado", "pct"),
    ("primeiro_estagio_errado", "texto abre com outro estagio", "pct"),
    ("linguagem_de_apoio", "frase de apoio ao veterinario", "pct"),
    ("escala_trocada", "termo da escala trocado", "cont"),
    ("porcentagens_sem_correspondencia", "porcentagem sem correspondencia", "cont"),
    ("ausencia_como_causa", "falta de exame descrita como causa", "cont"),
    ("efeito_fora_do_resumo", "efeito atribuido fora do resumo", "cont"),
    ("fator_nao_clinico_citado", "sexo, estado reprodutivo ou raca citados", "cont"),
    ("comentario_fora_do_resumo", "seguranca ou falta comentadas sem base", "cont"),
    ("fator_descrito_por_gravidade", "fator descrito como mais/menos grave", "cont"),
    ("incerteza_omitida", "sem creatinina e sem dizer isso", "cond:sem_creatinina"),
    ("padrao_de_registro_omitido", "falta pesou e sem a ressalva", "cond:falta_de_exames_pesou"),
    ("recomendacao_fora_da_lista", "exame recomendado fora da lista", "cont"),
    ("recomendacao_omitida", "exame principal nao recomendado", "cont"),
    ("parada_omitida", "sem exame util e sem dizer isso", "cont"),
    ("subestadiamento_omitido", "sem o lembrete do subestadiamento", "cont"),
    ("acima_do_limite", f"acima de {LIMITE_PALAVRAS} palavras", "cont"),
]


def sintese(medidas: list[dict]) -> dict:
    """Resumo por medida: proporcao, ou numero de textos em que ocorre."""
    d = pd.DataFrame([m for m in medidas if not m["erro"]])
    out = {"textos": len(d), "com_erro_da_api": sum(m["erro"] for m in medidas)}
    if d.empty:
        return out
    for col, _, tipo in MEDIDAS:
        if tipo == "pct":
            out[col] = round(float(d[col].mean()), 4)
        elif tipo.startswith("cond:"):
            alvo = d[d[tipo[5:]]]
            out[col] = {"textos": int((alvo[col] > 0).sum()), "de": len(alvo)}
        else:
            out[col] = int((d[col] > 0).sum())
    out["palavras"] = {"mediana": float(d.palavras.median()),
                       "min": int(d.palavras.min()), "max": int(d.palavras.max())}
    return out


def imprimir_comparacao(v1: dict, v2: dict | None) -> None:
    """Tabela das medidas da verificacao, versao 1 ao lado da versao 2."""
    def fmt(x, tipo):
        if x is None:
            return "-"
        if tipo == "pct":
            return f"{x:.1%}"
        if tipo.startswith("cond:"):
            return f"{x['textos']} de {x['de']}"
        return str(x)
    print(f"    {'medida (textos)':<44} {'v1':>10} {'v2':>10}")
    for col, rot, tipo in MEDIDAS:
        print(f"    {rot:<44} {fmt(v1.get(col), tipo):>10} "
              f"{fmt(v2.get(col) if v2 else None, tipo):>10}")
    p1 = v1.get("palavras", {})
    p2 = (v2 or {}).get("palavras", {})
    print(f"    {'palavras (mediana)':<44} {p1.get('mediana', '-'):>10} "
          f"{p2.get('mediana', '-'):>10}")


# ---------------------------------------------------------------------------
# Entrada e saida
# ---------------------------------------------------------------------------

def _sha(p: Path) -> str:
    """SHA-256 de um arquivo."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    """Caminho relativo a raiz do projeto, quando possivel."""
    return str(p.relative_to(cfg.RAIZ)) if p.is_relative_to(cfg.RAIZ) else str(p)


def origem_padrao() -> Path:
    """A execucao final mais recente em entregas/."""
    rodadas = sorted((cfg.RAIZ / "entregas").glob("rodada_final_*"))
    if not rodadas:
        raise SystemExit("Nenhuma rodada final em entregas/. A v2 le as fichas e as "
                         "respostas da v1 de la.")
    return rodadas[-1]


def carregar(origem: Path) -> tuple[list[dict], dict[int, dict]]:
    """Fichas e respostas da v1 de uma execucao final, conferindo que cobrem os mesmos gatos."""
    pasta = origem / "relatorios" / "camada"
    arq_f, arq_r = pasta / "fichas.jsonl", pasta / "respostas.jsonl"
    for a in (arq_f, arq_r):
        if not a.exists():
            raise SystemExit(f"Nao encontrado: {a}")
    fichas = [json.loads(l) for l in open(arq_f, encoding="utf-8")]
    v1 = {r["id"]: r for r in (json.loads(l) for l in open(arq_r, encoding="utf-8"))}
    if {f["id"] for f in fichas} != set(v1):
        raise SystemExit("Fichas e respostas da v1 nao cobrem os mesmos gatos.")
    return fichas, v1


def _texto_v1(r: dict) -> str | None:
    """Texto da traducao v1, ou None se a chamada falhou."""
    t = r["traducao"]
    return None if "_erro" in t else t.get("texto", "")


def _rag_md(r: dict) -> list[str]:
    """Itens da contextualizacao em Markdown, com a marca da verificacao."""
    linhas = []
    for it in r["rag"].get("itens", []):
        ok = "✓" if it.get("_sustentado") else ("~" if it.get("_citacao_valida") else "✗")
        linhas.append(f"- {ok} **{it.get('condicao')}** "
                      f"({', '.join(it.get('trechos', []))}) — {it.get('justificativa')}")
    return linhas


def gravar_textos(destino: Path, fichas: list[dict], v1: dict, v2: dict | None,
                  resumos: dict, med1: dict, med2: dict | None) -> None:
    """Leitura corrida: resumo, texto v1, texto v2 e o que a verificacao acusou."""
    ln = ["# Textos para o veterinário — versão 1 e versão 2\n"]
    for f in fichas:
        i = f["id"]
        ln += [f"\n---\n\n## Gato {i}\n",
               f"*Registrado:* estágio {f['_avaliacao']['estagio_registrado']}; "
               f"{'sem' if f['_avaliacao']['sem_creatinina'] else 'com'} creatinina.\n",
               "**Resumo enviado na v2:**\n",
               "```json\n" + json.dumps(para_modelo(resumos[i]), ensure_ascii=False, indent=1)
               + "\n```\n",
               "**Versão 1:**\n", f"> {_texto_v1(v1[i]) or '(erro)'}\n"]
        acusa1 = [k for k, v in med1[i].items() if k not in ("erro", "palavras", "sem_creatinina",
                  "falta_de_exames_pesou", "estagio_correto", "linguagem_de_apoio") and v]
        ln.append(f"*Verificação:* {', '.join(acusa1) or 'nada'}\n")
        if v2 is not None:
            t2 = v2[i]["resposta"].get("texto") if "_erro" not in v2[i]["resposta"] else None
            acusa2 = [k for k, v in med2[i].items() if k not in ("erro", "palavras",
                      "sem_creatinina", "falta_de_exames_pesou", "estagio_correto",
                      "linguagem_de_apoio") and v]
            falta = [k for k in ("estagio_correto", "linguagem_de_apoio")
                     if not med2[i].get(k, True)]
            ln += ["**Versão 2:**\n", f"> {t2 or '(erro)'}\n",
                   f"*Verificação:* {', '.join(acusa2 + [f'sem {k}' for k in falta]) or 'nada'}\n"]
    (destino / "textos.md").write_text("\n".join(ln), encoding="utf-8")


def gravar_leitura_comparada(destino: Path, fichas: list[dict], v1: dict, v2: dict) -> dict:
    """Os 10 gatos da leitura qualitativa, com as duas versoes em ordem sorteada."""
    amostra = selecionar_qualitativos(fichas)
    rng = random.Random(cfg.SEMENTE)
    chave = {}
    ln = ["# Leitura comparada — camada interpretativa, versões 1 e 2\n",
          "Os mesmos 10 gatos da leitura qualitativa. Para cada um, os textos **A** e "
          "**B** são as duas versões da tradução, em ordem sorteada. A chave está em "
          "`chave_leitura.json`: abra só depois de terminar.\n",
          "Critérios: princípios de Xiao et al. (2025) e compreensibilidade (Wang et "
          "al., 2023). Marque de 1 (não atende) a 5 (atende plenamente).\n"]
    for f in amostra:
        i = f["id"]
        ordem = ["v1", "v2"]
        rng.shuffle(ordem)
        chave[str(i)] = {"A": ordem[0], "B": ordem[1]}
        textos = {"v1": _texto_v1(v1[i]),
                  "v2": None if "_erro" in v2[i]["resposta"] else v2[i]["resposta"].get("texto")}
        ln += [f"\n---\n\n## Gato {i}\n",
               f"*Gabarito, só para a avaliadora:* estágio registrado "
               f"{f['_avaliacao']['estagio_registrado']}; "
               f"{'sem' if f['_avaliacao']['sem_creatinina'] else 'com'} creatinina.\n",
               "### Texto A\n", (textos[ordem[0]] or "(erro)") + "\n",
               "### Texto B\n", (textos[ordem[1]] or "(erro)") + "\n",
               "### Contextualização por literatura (RAG) — igual nas duas versões\n",
               *_rag_md(v1[i]),
               "\n### Avaliação\n",
               "| Critério | A (1–5) | B (1–5) |", "|---|---|---|",
               "| **Centralidade do profissional:** o texto se apresenta como apoio, não como veredito? | | |",
               "| **Explicabilidade:** os fatores que motivaram o resultado estão inteligíveis? | | |",
               "| **Transparência:** a incerteza e as limitações ficam claras? | | |",
               "| **Compreensibilidade:** um clínico generalista entenderia sem ajuda? | | |",
               "| **Correção clínica:** há alguma afirmação clinicamente incorreta? (anotar) | | |",
               "| **Utilidade:** qual dos dois você levaria para a consulta? (A, B ou tanto faz) | | |",
               "\nObservações:\n"]
    (destino / "leitura_comparada.md").write_text("\n".join(ln), encoding="utf-8")
    (destino / "chave_leitura.json").write_text(
        json.dumps(chave, indent=2, ensure_ascii=False), encoding="utf-8")
    return chave


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Camada interpretativa, versao 2 da traducao.")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--simular", action="store_true",
                      help="sem API: grava resumos e prompts, e verifica os textos da v1")
    modo.add_argument("--piloto", action="store_true",
                      help=f"com API, so os gatos {PILOTO}")
    ap.add_argument("--origem", type=Path, default=None,
                    help="pasta da rodada final (padrao: a mais recente em entregas/)")
    args = ap.parse_args()

    origem = args.origem.resolve() if args.origem else origem_padrao()
    fichas, v1 = carregar(origem)
    pasta_v1 = origem / "relatorios" / "camada"
    amostra = [f["id"] for f in selecionar_qualitativos(fichas)]
    if set(PILOTO) & set(amostra):
        raise SystemExit(f"Gatos do piloto na amostra da leitura: {set(PILOTO) & set(amostra)}.")
    if not set(PILOTO) <= {f["id"] for f in fichas}:
        raise SystemExit("Gato do piloto ausente das fichas.")

    destino = SAIDA / ("simulacao" if args.simular else "piloto" if args.piloto else "")
    destino.mkdir(parents=True, exist_ok=True)
    alvo = [f for f in fichas if f["id"] in PILOTO] if args.piloto else fichas

    resumos = {f["id"]: resumo(f) for f in fichas}
    med1 = {f["id"]: verificar(_texto_v1(v1[f["id"]]), resumos[f["id"]], f) for f in fichas}
    print(f"[1] origem: {_rel(origem)} | {len(fichas)} fichas | "
          f"{len(alvo)} a processar")
    print(f"    limiar de ganho {LIMIAR_GANHO_BITS} bit | ate {MAX_DADOS_CLINICOS} dados "
          f"clinicos | ate {LIMITE_PALAVRAS} palavras")
    rs = list(resumos.values())
    print(f"    resumos: {sum(not r['dados_clinicos'] for r in rs)} sem dado clinico; "
          f"{sum(not r['exames_recomendados'] for r in rs)} sem exame recomendado; "
          f"{sum(r['creatinina_medida'] == 'não' for r in rs)} sem creatinina; "
          f"{sum(bool(r['exames_para_subestadiamento']) for r in rs)} com lembrete "
          f"do subestadiamento")

    if args.simular:
        (destino / "prompts").mkdir(exist_ok=True)
        for f in fichas:
            i = f["id"]
            (destino / "prompts" / f"gato_{i}.md").write_text(
                f"# Gato {i}\n\n## Sistema\n\n{SISTEMA_V2}\n\n## Usuário\n\n```json\n"
                f"{json.dumps(para_modelo(resumos[i]), ensure_ascii=False, indent=1)}\n```\n\n"
                f"## Texto da v1, para comparação\n\n{_texto_v1(v1[i]) or '(erro)'}\n",
                encoding="utf-8")
        with open(destino / "resumos.jsonl", "w", encoding="utf-8") as fh:
            for i, r in resumos.items():
                fh.write(json.dumps({"id": i, "resumo": r}, ensure_ascii=False) + "\n")
        s1 = sintese(list(med1.values()))
        (destino / "verificacao_v1.json").write_text(
            json.dumps(s1, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[2] MODO SIMULACAO: nenhuma chamada a API.")
        print(f"    prompts em {destino / 'prompts'}")
        print("\n[3] verificacao nova aplicada aos textos da v1 (123):")
        imprimir_comparacao(s1, None)
        return

    llm = LLM()
    v2 = {}
    for n, f in enumerate(alvo, 1):
        i = f["id"]
        enviado = para_modelo(resumos[i])
        u = "Resumo do paciente:\n" + json.dumps(enviado, ensure_ascii=False, indent=1)
        resp = llm.json(SISTEMA_V2, u)
        texto = None if "_erro" in resp else resp.get("texto", "")
        v2[i] = {"id": i, "resumo": resumos[i], "enviado": enviado, "resposta": resp,
                 "verificacao": verificar(texto, resumos[i], f)}
        print(f"  gato {n}/{len(alvo)} | custo acumulado US$ {llm.custo():.4f}", end="\r")
    print(" " * 60, end="\r")

    with open(destino / "respostas_v2.jsonl", "w", encoding="utf-8") as fh:
        for r in v2.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ids = [f["id"] for f in alvo]
    s1 = sintese([med1[i] for i in ids])
    s2 = sintese([v2[i]["verificacao"] for i in ids])
    med2 = {i: v2[i]["verificacao"] for i in ids}
    gravar_textos(destino, alvo, v1, v2, resumos, med1, med2)
    print(f"[2] VERIFICACAO, mesmas regras nas duas versoes ({len(ids)} gatos)")
    imprimir_comparacao(s1, s2)

    resumo_exec = {
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "modo": "piloto" if args.piloto else "completo",
        "origem": _rel(origem),
        "sha256_entradas": {"fichas.jsonl": _sha(pasta_v1 / "fichas.jsonl"),
                            "respostas.jsonl": _sha(pasta_v1 / "respostas.jsonl")},
        "modelo": cfg.MODELO_LLM, "versao_declarada": versao_declarada(),
        "modelos_servidos": llm.modelos_servidos,
        "chamadas": llm.chamadas, "chamadas_com_erro": llm.erros,
        "parametros": {"limiar_ganho_bits": LIMIAR_GANHO_BITS,
                       "max_dados_clinicos": MAX_DADOS_CLINICOS,
                       "limite_palavras": LIMITE_PALAVRAS},
        "gatos": ids,
        "verificacao": {"v1": s1, "v2": s2},
        "tokens": [llm.tokens_in, llm.tokens_out], "custo_usd": round(llm.custo(), 4),
    }
    (destino / "resumo_v2.json").write_text(
        json.dumps(resumo_exec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[3] custo: {llm.tokens_in:,} tokens de entrada, {llm.tokens_out:,} de "
          f"saida -> US$ {llm.custo():.4f}")
    print(f"    versao servida pela OpenAI: {llm.modelos_servidos}")
    if llm.erros:
        print(f"    ATENCAO: {llm.erros} de {llm.chamadas} chamadas falharam.")

    if not args.piloto:
        gravar_leitura_comparada(destino, fichas, v1, v2)
        print(f"\n[4] leitura comparada: 10 gatos em {destino / 'leitura_comparada.md'}")
        print("    (a chave A/B esta em chave_leitura.json; nao abrir antes da leitura)")
    print(f"\nsaidas em {destino}")
    print(f"    textos lado a lado: {destino / 'textos.md'}")


if __name__ == "__main__":
    main()
