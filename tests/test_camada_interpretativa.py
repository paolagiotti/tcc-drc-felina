"""Camada interpretativa, versao 1: consultas de busca, recuperacao alternada e
verificacao automatica da traducao. Nada aqui chama a API nem carrega o indice."""
import numpy as np
import pytest

import camada_interpretativa as ci
import config as cfg
from indexar_base import normalizar


# ---------------------------------------------------------------- utilidades

def test_ficha_para_modelo_remove_o_gabarito(ficha):
    enviada = ci.ficha_para_modelo(ficha)
    assert "_avaliacao" not in enviada and "estagio_predito" in enviada


def test_perfil_resume_o_paciente(ficha):
    p = ci.perfil(ficha)
    assert "Estágio IRIS estimado: 3 (probabilidade 62%)" in p
    assert "doença periodontal (55%)" in p and "hipertireoidismo (22%)" in p
    assert "cistite" not in p                       # so as duas mais provaveis
    assert "Creatinina: 310 µmol/L" in p and "UPC" not in p   # valor ausente fica fora


@pytest.mark.parametrize("texto, esperado", [
    ("O estágio IRIS 3 é o mais provável.", [3]),
    ("estágio IRIS 2, com estágios 3 e 4 possíveis", [2, 3]),
    ("Estadiamento IV; estágio ii antes", [4, 2]),
    ("sem menção", []),
    # limitacao da v1, corrigida na v2: o numero precisa vir logo apos a palavra
    ("O estágio estimado é 3.", []),
])
def test_estagios_no_texto(texto, esperado):
    assert ci._estagios_no_texto(texto) == esperado


def test_termo_em_texto_normalizado():
    t = normalizar("A doença periodontal é frequente em gatos idosos.")
    assert ci._termo_em("Doença periodontal", t)
    assert ci._termo_em("periodontal disease", t) is False
    assert ci._termo_em("gatos idosos doença", t)        # todas as palavras com 4+ letras
    assert ci._termo_em("", t) is False


def test_cita_exige_palavra_inteira():
    assert ci._cita("idade", normalizar("a idade pesou"))
    assert not ci._cita("idade", normalizar("a probabilidade subiu"))


def test_idioma():
    assert ci._idioma("The cat is in the house with the dog") == "inglês"
    assert ci._idioma("O gato está na casa com o cão que mora no bairro") == "português"


# -------------------------------------------------------------------- busca

def test_consultas_cobrem_estagio_comorbidades_e_achados(ficha):
    temas = ci.consultas(ficha)
    rotulos = [r for r, _ in temas]
    assert rotulos == ["estágio", "doença periodontal", "hipertireoidismo", "achados"]
    assert "IRIS stage 3" in temas[0][1]
    achados = temas[-1][1]
    assert "Creatinina: 310 µmol/L" in achados
    assert "Sexo" not in achados and "Bócio" not in achados and "UPC" not in achados


def test_consultas_ignoram_comorbidade_pouco_provavel(ficha):
    ficha["comorbidades"][1]["probabilidade"] = 0.19
    rotulos = [r for r, _ in ci.consultas(ficha)]
    assert "hipertireoidismo" not in rotulos


def test_comorbidade_sem_consulta_definida_interrompe(ficha):
    ficha["comorbidades"][0]["nome"] = "diabetes"
    with pytest.raises(SystemExit):
        ci.consultas(ficha)


class _BaseFalsa:
    """Devolve listas fixas por consulta, para testar a alternancia entre temas."""
    def __init__(self, listas):
        self.listas = listas
        self.trechos = [{"id": i, "texto": f"trecho {i}", "documento": "d", "pagina": 1}
                        for i in range(10)]

    def buscar(self, consulta, k):
        return [{"id": i, "texto": f"trecho {i}", "documento": "d", "pagina": 1}
                for i in self.listas[consulta]][:k]

    def contem(self, termo, ids=None):
        alvo = self.trechos if ids is None else [self.trechos[i] for i in ids]
        return any(normalizar(termo) in t["texto"] for t in alvo)


def test_recuperacao_alterna_entre_temas_sem_repetir():
    base = _BaseFalsa({"q1": [1, 2, 3, 4, 5], "q2": [1, 6, 7, 8, 9]})
    escolhidos = ci.recuperar(base, [("a", "q1"), ("b", "q2")], 5)
    assert [t["id"] for t in escolhidos] == [1, 2, 6, 3, 7]
    assert [t["tema"] for t in escolhidos] == ["a", "a", "b", "a", "b"]


def test_recuperacao_com_lista_curta_completa_com_o_outro_tema():
    base = _BaseFalsa({"q1": [1], "q2": [2, 3, 4, 5, 6]})
    ids = [t["id"] for t in ci.recuperar(base, [("a", "q1"), ("b", "q2")], 4)]
    assert ids == [1, 2, 3, 4]


# ------------------------------------------------------------- verificacao

TEXTO_FIEL = (
    "O estágio estimado é o estágio 3, provável (62%). A creatinina de 310 µmol/L e a "
    "idade de 14 anos aumentaram a probabilidade do estágio 3. Nos dados usados para "
    "desenvolver o sistema, a falta da UPC diminuiu a probabilidade do estágio estimado, "
    "o que reflete o padrão de registro dos atendimentos. Entre as comorbidades, a doença "
    "periodontal é provável (55%) e o hipertireoidismo é possível (22%). Recomenda-se a "
    "UPC, seguida da pressão arterial. O resultado é um apoio à avaliação do médico "
    "veterinário e não a substitui."
)


def _resposta(texto, **campos):
    r = {"texto": texto, "estagio_citado": 3,
         "fatores_citados": ["Creatinina", "Idade", "UPC"],
         "comorbidades_citadas": ["doença periodontal", "hipertireoidismo"],
         "exames_recomendados_citados": ["UPC", "Pressão arterial"]}
    r.update(campos)
    return r


def test_fidelidade_de_texto_fiel(ficha):
    f = ci.fidelidade(ficha, _resposta(TEXTO_FIEL))
    assert f["erro"] is False
    assert f["estagio_correto"] and not f["primeiro_estagio_errado"]
    assert f["estagio_autodeclarado_ok"]
    assert f["fatores_fora_da_ficha"] == 0
    assert f["variaveis_fora_dos_fatores"] == 0
    assert f["fator_descrito_por_gravidade"] == 0
    assert f["comorbidades_fora_da_ficha"] == 0
    assert f["exames_fora_da_ficha"] == 0 and f["exame_nao_informativo_recomendado"] == 0
    assert f["porcentagens"] == 3 and f["porcentagens_sem_correspondencia"] == 0
    assert f["linguagem_de_apoio"]


def test_fidelidade_acusa_desvios(ficha):
    texto = ("O estágio IRIS 4 é o estimado (estágio 3 com 62%). A ureia favoreceu estágios mais "
             "graves, e a creatinina indica doença mais avançada. Há 90% de chance de "
             "diabetes. Recomenda-se o hemograma.")
    r = _resposta(texto, estagio_citado=4, fatores_citados=["Ureia", "Creatinina"],
                  comorbidades_citadas=["diabetes"], exames_recomendados_citados=["Hemograma"])
    f = ci.fidelidade(ficha, r)
    assert f["primeiro_estagio_errado"] and f["estagio_correto"]      # o 3 aparece depois
    assert not f["estagio_autodeclarado_ok"]
    assert f["fatores_fora_da_ficha"] == 1                            # ureia
    assert f["variaveis_fora_dos_fatores"] == 1                       # ureia no texto
    assert f["fator_descrito_por_gravidade"] == 1
    assert f["comorbidades_fora_da_ficha"] == 1
    assert f["exames_fora_da_ficha"] == 1
    assert f["porcentagens_sem_correspondencia"] == 1                 # 90%
    assert not f["linguagem_de_apoio"]


def test_fidelidade_exame_nao_informativo(ficha):
    ficha["recomendacao_exames"][1]["reduz_duvida"] = False
    f = ci.fidelidade(ficha, _resposta(TEXTO_FIEL))
    assert f["exame_nao_informativo_recomendado"] == 1


def test_fidelidade_com_erro_da_api(ficha):
    assert ci.fidelidade(ficha, {"_erro": "x"}) == {"erro": True}


def test_fundamentacao_conta_citacoes_validas_e_sustentadas():
    base = _BaseFalsa({})
    base.trechos[0]["texto"] = "a anemia e frequente na doenca renal cronica"
    recuperados = [base.trechos[0], base.trechos[1]]
    r = {"itens": [
        {"condicao": "anemia", "termo_en": "anemia", "trechos": ["T1"]},      # valida e sustentada
        {"condicao": "hipertensão", "termo_en": "hypertension", "trechos": ["T2"]},  # valida, nao sustentada
        {"condicao": "hipocalemia", "termo_en": "hypokalemia", "trechos": ["T9"]},   # citacao invalida
    ]}
    f = ci.fundamentacao(r, recuperados, base)
    assert f == {"erro": False, "itens": 3, "citacao_valida": 2,
                 "sustentado_no_trecho": 1, "suprimidos": 1}
    assert r["itens"][0]["_sustentado"] and not r["itens"][1]["_sustentado"]


def test_verificabilidade_no_modelo_puro():
    base = _BaseFalsa({})
    base.trechos[3]["texto"] = "hyperphosphatemia is common"
    r = {"itens": [{"condicao": "hiperfosfatemia", "termo_en": "hyperphosphatemia"},
                   {"condicao": "gota", "termo_en": "gout"}]}
    assert ci.verificabilidade(r, base) == {"erro": False, "itens": 2, "verificavel_na_base": 1}


def test_selecao_qualitativa_dois_por_estagio_mais_sem_creatinina():
    fichas = []
    for e in (1, 2, 3, 4):
        for i in range(3):
            fichas.append({"id": e * 10 + i,
                           "_avaliacao": {"estagio_registrado": e, "sem_creatinina": i == 2}})
    amostra = ci.selecionar_qualitativos(fichas)
    assert len(amostra) == 10
    assert [f["_avaliacao"]["estagio_registrado"] for f in amostra[:8]] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert all(f["_avaliacao"]["sem_creatinina"] for f in amostra[8:])


def test_prompts_congelados_tem_as_regras_essenciais():
    assert "Use SOMENTE as informações da ficha" in ci.SISTEMA_TRADUCAO
    assert "180 palavras" in ci.SISTEMA_TRADUCAO
    assert "Cada item DEVE citar ao menos um trecho" in ci.SISTEMA_RAG
    assert "Não use conhecimento externo" in ci.SISTEMA_RAG
    assert "Não faça diagnóstico" in ci.SISTEMA_PURO


def test_versao_declarada_vem_da_configuracao():
    assert ci.versao_declarada() == cfg.VERSAO_LLM
    assert cfg.VERSAO_LLM in ci.versao_divergente("outro-modelo")
