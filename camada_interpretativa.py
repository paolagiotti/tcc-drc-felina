"""
Camada interpretativa: traducao estruturada e contextualizacao por literatura.

DOIS PAPEIS, como na metodologia:

  1. TRADUCAO ESTRUTURADA. O modelo de linguagem recebe a ficha do gato -- saidas
     dos tres componentes e fatores SHAP -- e escreve um texto curto para o
     veterinario. So verbaliza: nao calcula nem altera nenhuma predicao.

  2. CONTEXTUALIZACAO POR LITERATURA (RAG). Busca os trechos mais pertinentes na
     base de conhecimento e pede de 3 a 5 complicacoes ou condicoes associadas,
     cada uma com citacao obrigatoria do trecho que a sustenta. Item sem citacao
     valida e suprimido.

     A BUSCA e feita por TEMAS construidos a partir do perfil do paciente, como
     preve a metodologia, e nao pelo texto do perfil inteiro: o estagio estimado,
     cada comorbidade ao menos "possivel" e os achados clinicos presentes. Os k
     trechos sao escolhidos alternando entre os temas, na ordem de similaridade de
     cada um, sem repeticao. As consultas nao repetem os termos do dominio ("doenca
     renal cronica em gatos"): a base inteira trata disso, entao esses termos nao
     distinguem um trecho de outro e puxam a busca para os mais genericos.
     Consultas de estagio e de comorbidade levam tambem o termo original em
     ingles (da diretriz e do conjunto de dados): a base mistura os dois idiomas.

     BUSCA HIBRIDA. Cada consulta ordena os trechos duas vezes -- pela
     similaridade semantica (embeddings) e pela lexical (TF-IDF, palavras em
     comum, sem as palavras vazias da lista Snowball, como "ou" e "the") -- e as
     duas ordens sao combinadas por fusao de postos reciprocos
     (Cormack et al., 2009): pontuacao = 1/(60 + posto semantico) + 1/(60 +
     posto lexical). A busca semantica sozinha nao achava termos raros: para
     "doenca periodontal" e "artrite" nao trouxe nenhum trecho sobre elas, e para
     "hipertireoidismo" trouxe um sobre hiperparatireoidismo; a lexical, nos
     mesmos trechos, acha os que citam o termo.

     Esse desenho veio de conferencias sem custo, feitas antes de qualquer texto
     gerado: com o perfil inteiro como consulta, os 123 gatos recuperavam poucos
     trechos distintos, concentrados no resumo e na metodologia de uma revisao,
     e a busca por hipertireoidismo nao trazia nenhum trecho sobre ele; os temas
     sem os termos do dominio e a busca hibrida ampliaram a cobertura, e a lista
     de palavras vazias corrigiu o posto lexical de trechos genericos.

AVALIACAO, com precedente em Jin et al. (2025), que avaliou o AVA (a) pela
acuracia com que o modelo usa a informacao estruturada e (b) contra modelos de
linguagem SEM a base de conhecimento:

  (a) FIDELIDADE da traducao -- verificacao automatica de que estagio,
      probabilidades, fatores, comorbidades e exames citados estao na ficha;
  (b) FUNDAMENTACAO do RAG contra o MODELO PURO -- a mesma pergunta com e sem os
      trechos; mede-se quantos itens tem sustentacao verificavel na base;
  (c) LEITURA QUALITATIVA de 10 casos, com os principios de Xiao et al. (2025)
      e o criterio de compreensibilidade de Wang et al. (2023).

Uso:
    python camada_interpretativa.py --simular          # sem API: grava prompts
    python camada_interpretativa.py --limite 5         # 5 gatos, com API
    python camada_interpretativa.py                    # os 123
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as cfg
from figuras import NOMES_PT
from indexar_base import normalizar

SAIDA = cfg.SAIDA_RELATORIOS / "camada"
BASE = cfg.SAIDA / "base"


# ---------------------------------------------------------------------------
# Instrucoes ao modelo
# ---------------------------------------------------------------------------

SISTEMA_TRADUCAO = """Você é a camada de verbalização de um sistema de apoio à \
decisão clínica para doença renal crônica felina. Recebe a saída estruturada de \
três componentes preditivos e escreve um texto curto para o médico veterinário.

Como ler a ficha:
- "estagio_predito" é o estágio IRIS estimado; "probabilidades", a probabilidade \
de cada estágio.
- Em "fatores", "direcao" diz se o fator AUMENTOU ("a favor") ou DIMINUIU \
("contra") a probabilidade do estágio estimado. Não diz se o fator aponta para \
estágio mais grave ou menos grave.
- "exames_disponiveis" e "exames_faltantes" servem só para a recomendação de \
exames; não são fatores.

Regras:
1. Use SOMENTE as informações da ficha. Não acrescente valores, diagnósticos, \
exames, condições ou recomendações que não estejam nela.
2. Expresse cada probabilidade em termo linguístico seguido do número: acima de \
70% "muito provável"; de 40% a 70% "provável"; de 20% a 40% "possível"; abaixo \
de 20% "pouco provável".
3. Descreva cada fator citado pelo efeito sobre o estágio estimado — por \
exemplo, "a idade aumentou a probabilidade do estágio 3". Não escreva que um \
fator favorece estágios mais ou menos graves, nem que atua a favor ou contra a \
progressão da doença.
4. Quando um fator tiver valor "ausente", não o trate como achado do paciente: \
escreva que, nos dados usados para desenvolver o sistema, a falta desse exame \
aumentou (se "a favor") ou diminuiu (se "contra") a probabilidade do estágio \
estimado, e que isso reflete o padrão de registro dos atendimentos, não uma \
característica do animal.
5. Fale apenas dos fatores listados em "fatores". Não atribua efeito sobre o \
estágio a nenhum outro exame ou variável.
6. Recomende apenas exames com "reduz_duvida": true, na ordem da ficha. Se não \
houver nenhum, diga que exames adicionais pouco acrescentariam ao estadiamento.
7. Termine com uma frase dizendo que o resultado é um apoio à avaliação do médico \
veterinário e não a substitui. Nunca o apresente como diagnóstico.
8. Não mencione SHAP, modelos, algoritmos ou termos de aprendizado de máquina.
9. Escreva em português, em no máximo 180 palavras.

Responda em JSON, com exatamente estes campos:
{"texto": "...",
 "estagio_citado": <inteiro>,
 "fatores_citados": [<nomes exatamente como na ficha>],
 "comorbidades_citadas": [<nomes exatamente como na ficha>],
 "exames_recomendados_citados": [<nomes exatamente como na ficha>]}"""

SISTEMA_RAG = """Você auxilia o médico veterinário com contextualização \
fundamentada em literatura sobre doença renal crônica felina. Recebe o perfil do \
paciente e TRECHOS de documentos de referência, identificados como T1, T2 etc.

Liste de 3 a 5 complicações ou condições concomitantes da doença renal crônica \
felina pertinentes a este paciente, em ordem de pertinência clínica.

Regras:
1. Cada item DEVE citar ao menos um trecho que o sustente, pelo identificador.
2. Não liste item que os trechos não sustentem. Não use conhecimento externo.
3. Não faça diagnóstico: trata-se de contextualização.

Responda em JSON:
{"itens": [{"condicao": "<nome em português>", "termo_en": "<nome em inglês>",
  "trechos": ["T1"], "justificativa": "<uma frase>"}]}"""

SISTEMA_PURO = """Você auxilia o médico veterinário com contextualização sobre \
doença renal crônica felina. Recebe o perfil do paciente.

Liste de 3 a 5 complicações ou condições concomitantes da doença renal crônica \
felina pertinentes a este paciente, em ordem de pertinência clínica. Não faça \
diagnóstico: trata-se de contextualização.

Responda em JSON:
{"itens": [{"condicao": "<nome em português>", "termo_en": "<nome em inglês>",
  "justificativa": "<uma frase>"}]}"""


def ficha_para_modelo(f: dict) -> dict:
    """Remove o que o modelo NAO pode ver: o gabarito da avaliacao."""
    return {k: v for k, v in f.items() if not k.startswith("_")}


def perfil(f: dict) -> str:
    """Perfil do paciente em uma frase, para a contextualizacao e o modelo puro."""
    comorb = ", ".join(f"{c['nome']} ({c['probabilidade']:.0%})"
                       for c in f["comorbidades"][:2]) or "não estimadas"
    achados = "; ".join(f"{x['variavel']}: {x['valor']}" for x in f["fatores"]
                        if x["valor"] != "ausente") or "sem valores laboratoriais"
    return (f"Gato, {f['idade']}, {f['sexo']}. Estágio IRIS estimado: "
            f"{f['estagio_predito']} (probabilidade "
            f"{f['probabilidades'][str(f['estagio_predito'])]:.0%}). "
            f"Comorbidades mais prováveis: {comorb}. Achados: {achados}.")


# ---------------------------------------------------------------------------
# Recuperacao
# ---------------------------------------------------------------------------

# Fatores que descrevem o paciente ou o atendimento, nao um achado clinico: ficam
# no perfil que o modelo le, mas fora da consulta de busca.
NAO_CLINICOS = {"Idade", "Sexo", "Estado reprodutivo", "Raça", "Motivo da apresentação"}
# Valores que dizem que o achado NAO esta presente: fora da busca, porque o vetor da
# frase ignora a negacao ("Sopro cardiaco: nao" buscaria trechos sobre sopro).
NEGATIVOS = {"não", "não registrado", "nenhum registrado", "não testado"}
# Consulta de cada comorbidade: o nome em portugues e o termo original do conjunto
# de dados (config.COMORBIDADES_MODELAVEIS), porque a base mistura os dois idiomas.
# Artrite leva "osteoartrite", o termo da literatura felina (e o unico da base);
# FLUTD vai por extenso.
TERMO_BUSCA = {
    "doença periodontal": "Doença periodontal (periodontal disease).",
    "hipertireoidismo": "Hipertireoidismo (hyperthyroidism).",
    "cistite/DTUIF": "Cistite ou doença do trato urinário inferior felino (cystitis, "
                     "feline lower urinary tract disease, FLUTD).",
    "artrite": "Artrite ou osteoartrite (arthritis, osteoarthritis).",
}
RRF_K = 60        # constante da fusao de postos reciprocos (Cormack et al., 2009)
# Comorbidade entra na busca a partir de "possivel" (20%), a mesma escala dos textos.
COMORBIDADE_MIN = 0.20
# Descricao de cada estagio nos comentarios da diretriz (IRIS, 2026). O numero
# sozinho quase nao muda o vetor da frase: "estagio 2" e "estagio 4" recuperariam
# os mesmos trechos.
DESCRICAO_ESTAGIO = {   # (portugues, original da diretriz)
    1: ("creatinina normal com outra anormalidade renal: baixa concentração "
        "urinária, proteinúria renal ou alteração à imagem",
        "normal creatinine with another renal abnormality: inadequate urine "
        "concentrating ability, renal proteinuria or abnormal imaging"),
    2: ("azotemia renal discreta, com sinais clínicos geralmente leves ou ausentes",
        "mild renal azotemia; clinical signs usually mild or absent"),
    3: ("azotemia renal moderada, com sinais extrarrenais possivelmente presentes",
        "moderate renal azotemia; many extrarenal signs may be present"),
    4: ("risco crescente de sinais clínicos sistêmicos e de crises urêmicas",
        "increasing risk of systemic clinical signs and uremic crises"),
}


def consultas(f: dict) -> list[tuple[str, str]]:
    """Temas de busca do paciente: (rotulo, texto da consulta)."""
    e = f["estagio_predito"]
    pt, en = DESCRICAO_ESTAGIO[e]
    temas = [("estágio", f"Complicações no estágio IRIS {e}: {pt} (IRIS stage {e}: {en}).")]
    for c in f["comorbidades"][:2]:
        if c["probabilidade"] >= COMORBIDADE_MIN:
            if c["nome"] not in TERMO_BUSCA:
                raise SystemExit(f"Comorbidade sem consulta definida: {c['nome']!r}. "
                                 f"Acrescente-a em TERMO_BUSCA.")
            temas.append((c["nome"], TERMO_BUSCA[c["nome"]]))
    achados = [f"{x['variavel']}: {x['valor']}" for x in f["fatores"]
               if x["valor"] != "ausente" and x["valor"] not in NEGATIVOS
               and x["variavel"] not in NAO_CLINICOS]
    if achados:
        temas.append(("achados", "Achados: " + "; ".join(achados) + "."))
    return temas


def recuperar(base: "Base", temas: list[tuple[str, str]], k: int) -> list[dict]:
    """k trechos, alternando entre os temas pela ordem de similaridade de cada um."""
    listas = [(rotulo, base.buscar(texto, k)) for rotulo, texto in temas]
    escolhidos, vistos = [], set()
    for posicao in range(k):
        for rotulo, lista in listas:
            if len(escolhidos) == k:
                return escolhidos
            if posicao < len(lista) and lista[posicao]["id"] not in vistos:
                vistos.add(lista[posicao]["id"])
                escolhidos.append({**lista[posicao], "tema": rotulo})
    return escolhidos


def _idioma(texto: str) -> str:
    """Ingles ou portugues, pela contagem de palavras funcionais."""
    en = len(re.findall(r"\b(the|and|of|with|is|are|in)\b", texto.lower()))
    pt = len(re.findall(r"\b(de|da|do|com|em|que|os|as|na|no)\b", texto.lower()))
    return "inglês" if en > pt else "português"


def diagnostico_recuperacao(recuperacoes: list[list[dict]],
                            buscas: list[list[tuple[str, str]]], base: "Base") -> dict:
    """O que a busca entrega: trechos por consulta, por documento e por idioma."""
    todos = [t for r in recuperacoes for t in r]
    freq = pd.Series([t["id"] for t in todos]).value_counts()
    idioma_base = pd.Series([_idioma(t["texto"]) for t in base.trechos])
    # consultas fixas (estagio e comorbidades): que trechos cada uma trouxe
    por_consulta: dict[str, dict] = {}
    for temas, recup in zip(buscas, recuperacoes):
        for rotulo, texto in temas:
            if rotulo == "achados":
                continue
            d = por_consulta.setdefault(texto, {"gatos": 0, "trechos": {}})
            d["gatos"] += 1
            for t in recup:
                if t["tema"] == rotulo:
                    d["trechos"][t["id"]] = (f"{t['documento']}, p. {t['pagina']}: "
                                             f"{t['texto'][:70]}")
    achados = [t for t in todos if t["tema"] == "achados"]
    return {
        "busca": f"hibrida: semantica + lexical, fusao de postos reciprocos (k = {RRF_K})",
        "gatos": len(recuperacoes), "recuperacoes": len(todos),
        "trechos_distintos": int(freq.size), "trechos_na_base": len(base.trechos),
        "por_documento": pd.Series([t["documento"] for t in todos])
                           .value_counts().to_dict(),
        "idioma": {"recuperados": pd.Series([_idioma(t["texto"]) for t in todos])
                                    .value_counts(normalize=True).round(3).to_dict(),
                   "base": idioma_base.value_counts(normalize=True).round(3).to_dict()},
        "por_consulta": {c: {"gatos": d["gatos"], "trechos": list(d["trechos"].values())}
                         for c, d in sorted(por_consulta.items())},
        "achados": {"recuperacoes": len(achados),
                    "trechos_distintos": len({t["id"] for t in achados})},
        "mais_frequentes": [{"vezes": int(n), "documento": base.trechos[i]["documento"],
                             "pagina": base.trechos[i]["pagina"],
                             "inicio": base.trechos[i]["texto"][:90]}
                            for i, n in freq.head(10).items()],
    }


class Base:
    """Base de conhecimento indexada: busca hibrida (semantica e lexical) nos trechos."""
    def __init__(self):
        import faiss
        if not (BASE / "indice.faiss").exists():
            raise SystemExit("Base nao indexada. Rode antes: python indexar_base.py")
        self.meta = json.loads((BASE / "meta.json").read_text(encoding="utf-8"))
        if "n_janelas" not in self.meta:
            raise SystemExit("Indice da versao anterior (sem janelas). Rode de novo: "
                             "python indexar_base.py")
        self.indice = faiss.read_index(str(BASE / "indice.faiss"))
        self.trechos = [json.loads(l) for l in
                        open(BASE / "trechos.jsonl", encoding="utf-8")]
        janelas = [json.loads(l) for l in open(BASE / "janelas.jsonl", encoding="utf-8")]
        self.janela_trecho = [j["trecho"] for j in janelas]
        if self.indice.ntotal != len(janelas):
            raise SystemExit("indice.faiss e janelas.jsonl nao correspondem. Rode de "
                             "novo: python indexar_base.py")
        self.texto_norm = [normalizar(t["texto"]) for t in self.trechos]
        from indexar_base import vetorizar
        # parte lexical da busca hibrida: TF-IDF dos trechos, reajustado a cada
        # carga (deterministico; nada serializado)
        self.lex_m, self.lex_vec = vetorizar([t["texto"] for t in self.trechos], "tfidf")
        if self.meta["recuperador"] == "embeddings":
            from indexar_base import carregar_modelo, contador_tokens
            self.modelo = carregar_modelo(self.meta["modelo_embeddings"])
            self.n_tokens = contador_tokens(self.modelo)
            self.vec = None
        else:
            print("  ATENCAO: base indexada com TF-IDF, so para teste; nao e o "
                  "metodo da metodologia.")
            self.modelo = None
            # reajuste deterministico sobre as mesmas janelas: reproduz os vetores
            # do indice sem depender de objeto serializado
            _, self.vec = vetorizar([j["texto"] for j in janelas], "tfidf")

    def _semantica(self, consulta: str) -> np.ndarray:
        """Similaridade de cada trecho: a da sua melhor janela."""
        if self.modelo is not None:
            n = self.n_tokens(consulta)
            if n > self.modelo.max_seq_length:      # nunca truncar em silencio
                raise SystemExit(f"Consulta com {n} tokens, acima do limite do modelo "
                                 f"({self.modelo.max_seq_length}): {consulta[:80]!r}")
            q = self.modelo.encode([consulta], normalize_embeddings=True)
        else:
            q = self.vec.transform([consulta]).toarray()
            q /= np.linalg.norm(q).clip(1e-9)
        sim, ids = self.indice.search(q.astype("float32"), self.indice.ntotal)
        melhor = np.full(len(self.trechos), -np.inf)
        for s, j in zip(sim[0], ids[0]):
            i = self.janela_trecho[j]
            if s > melhor[i]:
                melhor[i] = s
        return melhor

    def _lexical(self, consulta: str) -> np.ndarray:
        """Similaridade lexical (TF-IDF, cosseno) de cada trecho com a consulta."""
        q = self.lex_vec.transform([consulta]).toarray()[0]
        return self.lex_m @ (q / max(np.linalg.norm(q), 1e-9))

    def buscar(self, consulta: str, k: int) -> list[dict]:
        """k trechos pela fusao de postos reciprocos das buscas semantica e lexical.
        Na lexical so entram trechos com alguma palavra da consulta."""
        sem, lex = self._semantica(consulta), self._lexical(consulta)
        posto_sem = np.empty(len(sem), dtype=int)
        posto_sem[np.argsort(-sem, kind="stable")] = np.arange(1, len(sem) + 1)
        pont = 1.0 / (RRF_K + posto_sem)
        posto_lex = np.zeros(len(lex), dtype=int)
        com = np.flatnonzero(lex > 0)
        posto_lex[com[np.argsort(-lex[com], kind="stable")]] = np.arange(1, len(com) + 1)
        pont[com] += 1.0 / (RRF_K + posto_lex[com])
        ordem = np.lexsort((-sem, -pont))[:k]       # empate: a maior semantica
        return [{**self.trechos[i], "similaridade": float(sem[i]),
                 "lexical": round(float(lex[i]), 4), "posto_semantico": int(posto_sem[i]),
                 "posto_lexical": int(posto_lex[i]) or None, "rrf": float(pont[i])}
                for i in ordem]

    def contem(self, termo: str, ids: list[int] | None = None) -> bool:
        """Verificacao lexical: o termo aparece no(s) trecho(s)?"""
        alvo = self.texto_norm if ids is None else [self.texto_norm[i] for i in ids]
        return any(_termo_em(termo, t) for t in alvo)


def _termo_em(termo: str, texto: str) -> bool:
    """Termo inteiro, ou todas as suas palavras significativas, no texto."""
    t = normalizar(termo or "").strip()
    if not t:
        return False
    if t in texto:
        return True
    palavras = [p for p in re.findall(r"[a-z0-9]+", t) if len(p) >= 4]
    return bool(palavras) and all(p in texto for p in palavras)


# ---------------------------------------------------------------------------
# Modelo de linguagem
# ---------------------------------------------------------------------------

def _transitorio(e: Exception) -> bool:
    """Erro que pode passar sozinho: limite de requisicoes, rede, falha do servidor.

    Todo o resto -- chave invalida, modelo nao liberado, parametro recusado, falta
    de credito -- se repetiria nas centenas de chamadas seguintes. Esses param a
    execucao na hora, em vez de gerar um arquivo inteiro de respostas com erro.
    """
    import openai
    if isinstance(e, openai.RateLimitError):
        return getattr(e, "code", None) != "insufficient_quota"
    return isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                          openai.InternalServerError))


def diagnosticar(e: Exception, modelo: str) -> str:
    """Traduz o erro da API em instrucao."""
    import openai
    if isinstance(e, openai.AuthenticationError):
        return ("chave recusada: foi apagada ou esta errada. Crie outra no painel da "
                "OpenAI e defina de novo a variavel de ambiente OPENAI_API_KEY.")
    if isinstance(e, openai.RateLimitError) and getattr(e, "code", None) == "insufficient_quota":
        return "sem credito na conta. Adicione em Settings -> Billing."
    if isinstance(e, (openai.PermissionDeniedError, openai.NotFoundError)):
        return (f"o projeto da OpenAI nao tem acesso a {modelo}. No painel, abra o "
                f"projeto -> Limits -> modelos permitidos e libere exatamente "
                f"'{modelo}' (liberar so o nome curto nao basta).")
    if isinstance(e, openai.BadRequestError):
        return f"a API recusou a requisicao: {str(e)[:200]}"
    return f"{type(e).__name__}: {str(e)[:200]}"


def _requisicao(cliente, modelo: str, sistema: str, usuario: str):
    """A chamada usada em TODA a camada: mesmos parametros no teste e no uso."""
    return cliente.chat.completions.create(
        model=modelo, temperature=0, seed=cfg.SEMENTE,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": sistema},
                  {"role": "user", "content": usuario}])


def testar_acesso(cliente=None, modelo: str | None = None) -> tuple[bool, str]:
    """Chamada minima com os parametros reais. Devolve (ok, mensagem)."""
    modelo = modelo or cfg.MODELO_LLM
    if cliente is None:
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY nao definida no ambiente."
        from openai import OpenAI
        cliente = OpenAI()
    try:
        r = _requisicao(cliente, modelo, 'Responda apenas com o JSON {"ok": true}.',
                        "teste")
        json.loads(r.choices[0].message.content)
    except json.JSONDecodeError:
        return False, f"{modelo} respondeu, mas nao em JSON valido."
    except Exception as e:  # noqa: BLE001 -- traduzido em instrucao abaixo
        return False, diagnosticar(e, modelo)
    if modelo == cfg.MODELO_LLM and r.model != versao_declarada():
        return False, versao_divergente(r.model)
    return True, f"{modelo} acessivel; versao servida: {r.model}"


def versao_declarada() -> str:
    """Versao do modelo de linguagem declarada no texto (config.VERSAO_LLM)."""
    return getattr(cfg, "VERSAO_LLM", cfg.MODELO_LLM)


def versao_divergente(servida: str) -> str:
    """Mensagem de interrupcao quando a API serve versao diferente da declarada."""
    return (f"a OpenAI serviu '{servida}', e nao a versao declarada "
            f"'{versao_declarada()}'. O nome '{cfg.MODELO_LLM}' passou a apontar "
            f"para outra versao. Nenhum texto deve ser gerado ate decidir se a "
            f"versao declarada no TCC muda (config.VERSAO_LLM).")


class LLM:
    """Cliente do modelo de linguagem: testa o acesso antes de gastar e contabiliza tokens."""
    def __init__(self):
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY nao definida. Veja o README, ou "
                             "rode com --simular para gerar so os prompts.")
        self.cliente = OpenAI()
        self.tokens_in = self.tokens_out = 0
        self.erros = 0
        self.chamadas = 0
        self.modelos_servidos: dict[str, int] = {}
        # Teste antes de gastar: se o modelo nao estiver acessivel, para aqui,
        # e nao depois de centenas de chamadas com erro.
        ok, msg = testar_acesso(self.cliente)
        print(f"    API: {msg}")
        if not ok:
            raise SystemExit("Execucao interrompida antes de qualquer gasto.")

    def json(self, sistema: str, usuario: str) -> dict:
        """Uma chamada com resposta em JSON; repete so erros transitorios, ate tres vezes."""
        self.chamadas += 1
        for tentativa in range(3):
            try:
                r = _requisicao(self.cliente, cfg.MODELO_LLM, sistema, usuario)
            except Exception as e:  # noqa: BLE001 -- classificado abaixo
                if not _transitorio(e):
                    raise SystemExit(f"\nExecucao interrompida: "
                                     f"{diagnosticar(e, cfg.MODELO_LLM)}") from e
                if tentativa == 2:
                    self.erros += 1
                    return {"_erro": f"{type(e).__name__} apos 3 tentativas"}
                time.sleep(5 * (tentativa + 1))
                continue
            self.tokens_in += r.usage.prompt_tokens
            self.tokens_out += r.usage.completion_tokens
            self.modelos_servidos[r.model] = self.modelos_servidos.get(r.model, 0) + 1
            if r.model != versao_declarada():
                raise SystemExit(f"\nExecucao interrompida: {versao_divergente(r.model)}")
            try:
                return json.loads(r.choices[0].message.content)
            except json.JSONDecodeError:
                self.erros += 1
                return {"_erro": "resposta nao e JSON valido"}

    def custo(self) -> float:
        """Custo acumulado em dolares, pelos precos de config.PRECO_LLM."""
        return (self.tokens_in * cfg.PRECO_LLM[0] + self.tokens_out * cfg.PRECO_LLM[1]) / 1e6


# ---------------------------------------------------------------------------
# Verificacao automatica
# ---------------------------------------------------------------------------

ROMANOS = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
VARIAVEIS_CONHECIDAS = sorted(set(NOMES_PT.values()), key=len, reverse=True)
# "estagio 2", "estagio IRIS 2", "estagios 3 e 4", "estadiamento 4", "estagio IV"
_RE_ESTAGIO = re.compile(r"(?:est[aá]gios?|estadiamento)\s+(?:iris\s+)?(\d|iv|i{1,3})\b",
                         re.IGNORECASE)
# efeito de fator descrito como gravidade, o que a regra 3 proibe: a direcao do
# fator e relativa ao estagio estimado, nao a gravidade
_RE_GRAVIDADE = re.compile(r"mais grave|menos grave|mais avancad|menos avancad|avanco|progress")
_RE_RECOMENDACAO = re.compile(r"recomend|reduzir|acrescent")


def _cita(nome: str, frase_normalizada: str) -> bool:
    """Nome como palavra inteira: "idade" nao conta dentro de "probabilidade"."""
    return re.search(rf"(?<!\w){re.escape(normalizar(nome))}(?!\w)",
                     frase_normalizada) is not None


def _estagios_no_texto(texto: str) -> list[int]:
    """Estagios mencionados, na ordem em que aparecem."""
    out = []
    for m in _RE_ESTAGIO.finditer(texto):
        g = m.group(1).lower()
        out.append(int(g) if g.isdigit() else ROMANOS[g])
    return out


def fidelidade(f: dict, r: dict) -> dict:
    """Verificacao automatica da traducao: o texto diz o que a ficha diz?"""
    if "_erro" in r:
        return {"erro": True}
    texto = r.get("texto", "")
    tn = normalizar(texto)
    frases = [normalizar(x) for x in re.split(r"(?<=[.!?])\s+", texto)]
    nomes_fatores = {x["variavel"] for x in f["fatores"]}
    comorbs = {c["nome"] for c in f["comorbidades"]}
    exames_ok = {e["exame"] for e in f["recomendacao_exames"] if e["reduz_duvida"]}
    exames_nao = {e["exame"] for e in f["recomendacao_exames"] if not e["reduz_duvida"]}
    exames = [normalizar(e) for e in
              set(f["exames_disponiveis"]) | set(f["exames_faltantes"]) | exames_ok | exames_nao]

    # estagio mencionado NO TEXTO, e nao so no campo autodeclarado; mencionar
    # outro estagio com a probabilidade da ficha e fiel -- erro e ABRIR com outro
    mencoes = _estagios_no_texto(texto)

    # porcentagens do texto devem corresponder a alguma probabilidade da ficha
    probs = [100 * p for p in f["probabilidades"].values()] + \
            [100 * c["probabilidade"] for c in f["comorbidades"]]
    pcts = [float(p.replace(",", ".")) for p in
            re.findall(r"(\d{1,3}(?:[.,]\d+)?)\s*%", texto)]
    pct_ruins = [p for p in pcts if not any(abs(p - q) <= 1.5 for q in probs)]

    # variavel citada que NAO esta entre os fatores. Exame so e aceito na frase da
    # recomendacao; fora dela, falar de exame que nao e fator e atribuir-lhe papel
    # que a ficha nao da (ex.: "a ausencia de UPC esteve associada a casos graves")
    fora = set()
    for fr in frases:
        for v in VARIAVEIS_CONHECIDAS:
            if v in nomes_fatores or not _cita(v, fr):
                continue
            if _RE_RECOMENDACAO.search(fr) and any(normalizar(v) in e for e in exames):
                continue
            fora.add(v)

    # fator descrito como "mais grave / menos grave / progressao" (regra 3)
    por_gravidade = sum(1 for fr in frases if _RE_GRAVIDADE.search(fr)
                        and any(_cita(v, fr) for v in nomes_fatores))

    cit_fat = set(r.get("fatores_citados", []))
    cit_com = set(r.get("comorbidades_citadas", []))
    cit_exa = set(r.get("exames_recomendados_citados", []))
    return {
        "erro": False,
        "estagio_correto": f["estagio_predito"] in mencoes,
        "primeiro_estagio_errado": bool(mencoes) and mencoes[0] != f["estagio_predito"],
        "outros_estagios_mencionados": len(set(mencoes) - {f["estagio_predito"]}),
        "estagio_autodeclarado_ok": r.get("estagio_citado") == f["estagio_predito"],
        "fatores_citados": len(cit_fat),
        "fatores_fora_da_ficha": len(cit_fat - nomes_fatores),
        "variaveis_fora_dos_fatores": len(fora),
        "fator_descrito_por_gravidade": por_gravidade,
        "comorbidades_fora_da_ficha": len(cit_com - comorbs),
        "exame_nao_informativo_recomendado": len(cit_exa & exames_nao),
        "exames_fora_da_ficha": len(cit_exa - exames_ok - exames_nao),
        "porcentagens": len(pcts),
        "porcentagens_sem_correspondencia": len(pct_ruins),
        "palavras": len(texto.split()),
        "linguagem_de_apoio": bool(re.search(r"apoio|auxili|complement|decis[aã]o|substitu", tn)),
    }


def fundamentacao(r: dict, recuperados: list[dict], base: Base) -> dict:
    """Verificacao do RAG: cada item cita um trecho valido, e o trecho sustenta o termo?"""
    if "_erro" in r:
        return {"erro": True}
    rotulo = {f"T{i+1}": t["id"] for i, t in enumerate(recuperados)}
    itens = r.get("itens", [])
    validos, sustentados = 0, 0
    for it in itens:
        ids = [rotulo[t] for t in it.get("trechos", []) if t in rotulo]
        it["_citacao_valida"] = bool(ids)
        it["_sustentado"] = bool(ids) and (base.contem(it.get("condicao", ""), ids)
                                            or base.contem(it.get("termo_en", ""), ids))
        validos += it["_citacao_valida"]
        sustentados += it["_sustentado"]
    return {"erro": False, "itens": len(itens), "citacao_valida": validos,
            "sustentado_no_trecho": sustentados,
            "suprimidos": len(itens) - validos}


def verificabilidade(r: dict, base: Base) -> dict:
    """Para o modelo puro: o item aparece em ALGUM lugar da base?"""
    if "_erro" in r:
        return {"erro": True}
    itens = r.get("itens", [])
    ok = sum(base.contem(i.get("condicao", "")) or base.contem(i.get("termo_en", ""))
             for i in itens)
    return {"erro": False, "itens": len(itens), "verificavel_na_base": ok}


# ---------------------------------------------------------------------------

def selecionar_qualitativos(fichas: list[dict], n: int = 10) -> list[dict]:
    """Amostra deterministica: dois por estagio, mais dois sem creatinina."""
    escolha = []
    for e in (1, 2, 3, 4):
        g = [f for f in fichas if f["_avaliacao"]["estagio_registrado"] == e]
        escolha += g[:2]
    resto = [f for f in fichas if f["_avaliacao"]["sem_creatinina"] and f not in escolha]
    return (escolha + resto)[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Camada interpretativa.")
    ap.add_argument("--simular", action="store_true",
                    help="grava os prompts, sem chamar a API")
    ap.add_argument("--limite", type=int, default=None,
                    help="processar apenas os N primeiros gatos")
    args = ap.parse_args()
    SAIDA.mkdir(parents=True, exist_ok=True)

    arq = SAIDA / "fichas.jsonl"
    if not arq.exists():
        raise SystemExit("Fichas nao encontradas. Rode antes: python fichas_pacientes.py")
    fichas = [json.loads(l) for l in open(arq, encoding="utf-8")]
    if args.limite:
        fichas = fichas[:args.limite]

    print(f"[1] {len(fichas)} fichas | modelo {cfg.MODELO_LLM} "
          f"(versao declarada {versao_declarada()}) | k = {cfg.K_TRECHOS}")
    base = Base()
    print(f"    base: {base.meta['n_trechos']} trechos ({base.meta['n_janelas']} janelas) "
          f"de {len(base.meta['documentos'])} documentos ({base.meta['recuperador']})")

    llm = None if args.simular else LLM()
    if args.simular:
        (SAIDA / "prompts").mkdir(exist_ok=True)
        print("    MODO SIMULACAO: nenhuma chamada a API.")

    registros, recuperacoes, buscas = [], [], []
    for n, f in enumerate(fichas, 1):
        entrada = json.dumps(ficha_para_modelo(f), ensure_ascii=False, indent=1)
        paciente = perfil(f)
        temas = consultas(f)
        recup = recuperar(base, temas, cfg.K_TRECHOS)
        recuperacoes.append(recup)
        buscas.append(temas)
        contexto = "\n\n".join(f"[T{i+1}] ({t['documento']}, p. {t['pagina']})\n{t['texto']}"
                               for i, t in enumerate(recup))
        u_trad = f"Ficha do paciente:\n{entrada}"
        u_rag = f"Perfil do paciente:\n{paciente}\n\nTrechos:\n{contexto}"
        u_puro = f"Perfil do paciente:\n{paciente}"

        if args.simular:
            lista_buscas = "\n".join(f"- **{r}:** {c}" for r, c in temas)
            origem = "\n".join(f"- T{i+1}: tema *{t['tema']}*; posto semântico "
                               f"{t['posto_semantico']} (similaridade "
                               f"{t['similaridade']:.3f}), posto lexical "
                               f"{t['posto_lexical'] or '—'}" for i, t in enumerate(recup))
            (SAIDA / "prompts" / f"gato_{f['id']}.md").write_text(
                f"# Gato {f['id']}\n\n## Papel 1 — tradução\n\n### Sistema\n\n"
                f"{SISTEMA_TRADUCAO}\n\n### Usuário\n\n```json\n{entrada}\n```\n\n"
                f"## Papel 2 — contextualização (RAG)\n\n### Consultas de busca "
                f"(não vão para o modelo)\n\n{lista_buscas}\n\n{origem}\n\n"
                f"### Usuário\n\nPerfil do paciente:\n{paciente}\n\n"
                f"### Trechos recuperados\n\n{contexto}\n",
                encoding="utf-8")
            continue

        r_trad = llm.json(SISTEMA_TRADUCAO, u_trad)
        r_rag = llm.json(SISTEMA_RAG, u_rag)
        r_puro = llm.json(SISTEMA_PURO, u_puro)
        registros.append({
            "id": f["id"], "_avaliacao": f["_avaliacao"],
            "traducao": r_trad, "rag": r_rag, "puro": r_puro,
            "consultas": temas,
            "recuperados": [{"id": t["id"], "documento": t["documento"],
                             "pagina": t["pagina"], "tema": t["tema"],
                             "similaridade": round(t["similaridade"], 4),
                             "posto_semantico": t["posto_semantico"],
                             "posto_lexical": t["posto_lexical"]} for t in recup],
            "fidelidade": fidelidade(f, r_trad),
            "fundamentacao": fundamentacao(r_rag, recup, base),
            "verificabilidade_puro": verificabilidade(r_puro, base),
            "verificabilidade_rag": verificabilidade(r_rag, base),
        })
        print(f"  gato {n}/{len(fichas)} | custo acumulado US$ {llm.custo():.4f}",
              end="\r")

    diag = diagnostico_recuperacao(recuperacoes, buscas, base)
    (SAIDA / "recuperacao.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.simular:
        print(f"\n[2] prompts gravados em {SAIDA / 'prompts'}")
        print("    Leia dois ou tres: e exatamente o que o modelo vai receber.")
    else:
        print(" " * 60, end="\r")
    print(f"[R] recuperacao: {diag['trechos_distintos']} trechos distintos em "
          f"{diag['recuperacoes']} recuperacoes ({diag['trechos_na_base']} na base)")
    print("    por documento: " + ", ".join(f"{d} {n}" for d, n in
                                            diag["por_documento"].items()))
    print("    em ingles: " + f"{diag['idioma']['recuperados'].get('inglês', 0):.0%} dos "
          f"recuperados, {diag['idioma']['base'].get('inglês', 0):.0%} da base")
    print(f"    detalhes em {SAIDA / 'recuperacao.json'}")
    if args.simular:
        return

    with open(SAIDA / "respostas.jsonl", "w", encoding="utf-8") as fh:
        for r in registros:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ------------------ resumo da avaliacao automatica ------------------
    fid = pd.DataFrame([r["fidelidade"] for r in registros])
    fid = fid[~fid.erro]
    print("[2] FIDELIDADE da traducao (o texto diz o que a ficha diz?)")
    for col, rot in [("estagio_correto", "estagio predito mencionado no texto"),
                     ("primeiro_estagio_errado", "texto abre com outro estagio"),
                     ("linguagem_de_apoio", "texto se apresenta como apoio")]:
        print(f"    {rot:<42} {fid[col].mean():.1%}")
    print(f"    {'menciona outros estagios (descritivo)':<42} "
          f"{(fid.outros_estagios_mencionados > 0).mean():.1%}")
    for col, rot in [("fatores_fora_da_ficha", "fatores citados fora da ficha"),
                     ("variaveis_fora_dos_fatores", "variaveis fora dos fatores"),
                     ("fator_descrito_por_gravidade", "fator descrito como mais/menos grave"),
                     ("comorbidades_fora_da_ficha", "comorbidades fora da ficha"),
                     ("exame_nao_informativo_recomendado", "exame nao informativo recomendado"),
                     ("porcentagens_sem_correspondencia", "porcentagens sem correspondencia")]:
        tot = int(fid[col].sum())
        print(f"    {rot:<42} {tot} ocorrencias em {int((fid[col] > 0).sum())} textos")
    print(f"    palavras por texto: mediana {fid.palavras.median():.0f}")

    fun = pd.DataFrame([r["fundamentacao"] for r in registros])
    fun = fun[~fun.erro]
    vp = pd.DataFrame([r["verificabilidade_puro"] for r in registros])
    vp = vp[~vp.erro]
    vr = pd.DataFrame([r["verificabilidade_rag"] for r in registros])
    vr = vr[~vr.erro]
    print("\n[3] FUNDAMENTACAO: RAG contra modelo puro (como em Jin et al., 2025)")
    print(f"    RAG  itens: {int(fun.itens.sum())} | citacao valida "
          f"{fun.citacao_valida.sum() / max(fun.itens.sum(), 1):.1%} | "
          f"sustentado no trecho citado "
          f"{fun.sustentado_no_trecho.sum() / max(fun.itens.sum(), 1):.1%}")
    print(f"    RAG  verificavel em algum ponto da base: "
          f"{vr.verificavel_na_base.sum() / max(vr.itens.sum(), 1):.1%}")
    print(f"    PURO verificavel em algum ponto da base: "
          f"{vp.verificavel_na_base.sum() / max(vp.itens.sum(), 1):.1%}")
    print("    (verificacao lexical: o termo precisa aparecer no trecho; e um")
    print("     criterio conservador, que subestima sinonimos e parafrases)")

    print(f"\n[4] custo: {llm.tokens_in:,} tokens de entrada, {llm.tokens_out:,} de "
          f"saida -> US$ {llm.custo():.4f}")
    print(f"    versao servida pela OpenAI: {llm.modelos_servidos}")
    if llm.erros:
        taxa = llm.erros / max(llm.chamadas, 1)
        print(f"    ATENCAO: {llm.erros} de {llm.chamadas} chamadas falharam ({taxa:.0%}).")
        if taxa > 0.05:
            print("    Acima de 5%: rode de novo antes de usar estes numeros.")

    resumo = {
        "executado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo": cfg.MODELO_LLM, "versao_declarada": versao_declarada(),
        "modelos_servidos": llm.modelos_servidos,
        "chamadas": llm.chamadas, "chamadas_com_erro": llm.erros,
        "k": cfg.K_TRECHOS, "n": len(registros),
        "recuperador": base.meta["recuperador"],
        "recuperacao": {c: diag[c] for c in ("trechos_distintos", "recuperacoes",
                                             "por_documento", "idioma")},
        "fidelidade": fid.mean(numeric_only=True).round(4).to_dict(),
        "fundamentacao_rag": fun.sum(numeric_only=True).to_dict(),
        "verificabilidade": {"rag": vr.sum(numeric_only=True).to_dict(),
                             "puro": vp.sum(numeric_only=True).to_dict()},
        "tokens": [llm.tokens_in, llm.tokens_out], "custo_usd": round(llm.custo(), 4),
    }
    (SAIDA / "resumo_avaliacao.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")

    # ------------------ amostra para leitura qualitativa ------------------
    por_id = {r["id"]: r for r in registros}
    amostra = [f for f in selecionar_qualitativos(fichas) if f["id"] in por_id]
    linhas = ["# Leitura qualitativa — camada interpretativa\n",
              "Critérios: princípios de Xiao et al. (2025) aplicáveis a cada texto, "
              "e compreensibilidade (Wang et al., 2023). Marque de 1 (não atende) "
              "a 5 (atende plenamente).\n"]
    for f in amostra:
        r = por_id[f["id"]]
        linhas += [f"\n---\n\n## Gato {f['id']}\n",
                   f"*Gabarito, só para a avaliadora:* estágio registrado "
                   f"{f['_avaliacao']['estagio_registrado']}; "
                   f"{'sem' if f['_avaliacao']['sem_creatinina'] else 'com'} creatinina.\n",
                   "### Texto para o veterinário\n",
                   r["traducao"].get("texto", "(erro)") + "\n",
                   "### Contextualização por literatura (RAG)\n"]
        for it in r["rag"].get("itens", []):
            ok = "✓" if it.get("_sustentado") else ("~" if it.get("_citacao_valida") else "✗")
            linhas.append(f"- {ok} **{it.get('condicao')}** "
                          f"({', '.join(it.get('trechos', []))}) — {it.get('justificativa')}")
        linhas += ["\n### Avaliação\n",
                   "| Critério | Nota (1–5) |", "|---|---|",
                   "| **Centralidade do profissional:** o texto se apresenta como apoio, não como veredito? | |",
                   "| **Explicabilidade:** os fatores que motivaram o resultado estão inteligíveis? | |",
                   "| **Transparência:** a incerteza e as limitações ficam claras? | |",
                   "| **Compreensibilidade:** um clínico generalista entenderia sem ajuda? | |",
                   "| **Correção clínica:** há alguma afirmação clinicamente incorreta? (anotar) | |",
                   "\nObservações:\n"]
    (SAIDA / "revisao_qualitativa.md").write_text("\n".join(linhas), encoding="utf-8")
    print(f"\n[5] amostra para leitura qualitativa: {len(amostra)} casos em "
          f"{SAIDA / 'revisao_qualitativa.md'}")
    print(f"\nsaidas em {SAIDA}")


if __name__ == "__main__":
    main()
