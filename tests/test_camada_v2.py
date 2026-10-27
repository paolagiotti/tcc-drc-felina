"""Camada interpretativa, versao 2: escala verbal, resumo preparado pelo codigo e
verificacao automatica dos textos."""
import pytest

import camada_v2 as v2


# ------------------------------------------------------------ escala verbal

@pytest.mark.parametrize("pct, termo", [
    (95, "muito provável"), (70.1, "muito provável"),
    (70, "provável"), (40, "provável"),
    (39.9, "possível"), (20, "possível"),
    (19.9, "pouco provável"), (0, "pouco provável"),
])
def test_escala(pct, termo):
    assert v2.escala(pct) == termo


def test_termo_com_numero_e_virgula_decimal():
    assert v2.termo(0.678) == "provável (67,8%)"
    assert v2.termo(0.05) == "pouco provável (5,0%)"


def test_numero_em_portugues_preserva_densidade_urinaria():
    assert v2._numero_pt("Creatinina", "2.5 mg/dL") == "2,5 mg/dL"
    assert v2._numero_pt("Densidade urinária", "1.020") == "1.020"


# ------------------------------------------------------------- dado clinico

@pytest.mark.parametrize("fator, esperado", [
    ({"variavel": "Creatinina", "valor": "310 µmol/L"}, "creatinina de 310 µmol/L"),
    ({"variavel": "Creatinina", "valor": "ausente"}, None),
    ({"variavel": "Idade", "valor": "14 anos"}, "idade de 14 anos"),
    ({"variavel": "Sexo", "valor": "fêmea"}, None),
    ({"variavel": "Raça", "valor": "Crossbreed"}, None),
    ({"variavel": "Bócio", "valor": "sim"}, "bócio"),
    ({"variavel": "Bócio", "valor": "não"}, None),
    ({"variavel": "Bócio", "valor": "não registrado"}, None),
    ({"variavel": "Motivo da apresentação", "valor": "sinais clínicos"},
     "apresentação com sinais clínicos"),
    ({"variavel": "Motivo da apresentação", "valor": "outro"}, None),
    ({"variavel": "Fármacos nefrotóxicos", "valor": "NSAID"},
     "exposição prévia a fármaco anti-inflamatório não esteroidal"),
    ({"variavel": "Fármacos nefrotóxicos", "valor": "não"}, None),
])
def test_dado_clinico(fator, esperado):
    assert v2.dado_clinico(fator) == esperado


@pytest.mark.parametrize("fator", [
    {"variavel": "Variável nova", "valor": "x"},
    {"variavel": "Bócio", "valor": "talvez"},
    {"variavel": "Motivo da apresentação", "valor": "emergência"},
])
def test_dado_clinico_sem_classificacao_interrompe(fator):
    with pytest.raises(SystemExit):
        v2.dado_clinico(fator)


# ------------------------------------------------------------------ resumo

def test_resumo_da_ficha(ficha):
    r = v2.resumo(ficha)
    assert r["estagio_estimado"] == 3
    assert r["probabilidade"] == "provável (62,0%)"
    assert r["segundo_estagio"] == {"estagio": 2, "probabilidade": "possível (25,0%)"}
    assert r["dados_clinicos"] == [
        {"dado": "creatinina de 310 µmol/L", "efeito": "sustenta o estágio estimado"},
        {"dado": "idade de 14 anos", "efeito": "sustenta o estágio estimado"},
    ]
    assert r["creatinina_medida"] == "sim"
    assert r["falta_de_exames_pesou"] == "sim"            # a UPC ausente e fator
    assert r["exames_recomendados"] == ["UPC"]            # 0,02 bit fica abaixo do limiar
    assert r["exames_para_subestadiamento"] == ["Pressão arterial"]
    assert [c["nome"] for c in r["comorbidades"]] == ["doença periodontal", "hipertireoidismo"]


def test_resumo_sem_segundo_estagio_nem_falta_de_exames(ficha):
    ficha["probabilidades"] = {"1": 0.05, "2": 0.10, "3": 0.80, "4": 0.05}
    ficha["fatores"] = [f for f in ficha["fatores"] if f["valor"] != "ausente"]
    r = v2.resumo(ficha)
    assert "segundo_estagio" not in r
    assert r["falta_de_exames_pesou"] == "não"
    assert r["probabilidade"] == "muito provável (80,0%)"


def test_resumo_limita_os_dados_clinicos(ficha):
    ficha["fatores"] = [{"variavel": "Creatinina", "valor": "310 µmol/L", "direcao": "a favor"},
                        {"variavel": "Ureia", "valor": "20 mmol/L", "direcao": "a favor"},
                        {"variavel": "Fósforo", "valor": "2 mmol/L", "direcao": "contra"},
                        {"variavel": "Idade", "valor": "14 anos", "direcao": "a favor"},
                        {"variavel": "Bócio", "valor": "sim", "direcao": "a favor"}]
    r = v2.resumo(ficha)
    assert len(r["dados_clinicos"]) == v2.MAX_DADOS_CLINICOS
    assert r["dados_clinicos"][2]["efeito"] == "pesa contra o estágio estimado"


def test_para_modelo_so_leva_os_indicadores_que_pedem_frase(ficha):
    rs = v2.resumo(ficha)
    m = v2.para_modelo(rs)
    assert "creatinina_medida" not in m                    # medida: nao vai
    assert m["falta_de_exames_pesou"] == "sim"
    assert m["exames_recomendados"] == ["UPC"]
    assert m["exames_para_subestadiamento"] == ["pressão arterial"]   # minusculas
    rs["creatinina_medida"] = "não"
    assert v2.para_modelo(rs)["creatinina_medida"] == "não"


# ------------------------------------------------------------- verificacao

TEXTO_V2 = (
    "O estágio estimado é IRIS 3, provável (62,0%), com o estágio 2 possível (25,0%). "
    "A estimativa se apoia na creatinina de 310 µmol/L e na idade de 14 anos, que "
    "sustentam o estágio estimado. Parte da estimativa reflete o padrão de registro dos "
    "atendimentos usados para desenvolver o sistema, e não achados do animal. "
    "Recomenda-se a UPC, o exame que mais esclareceria o estágio; a pressão arterial "
    "segue indicada para o subestadiamento. Entre as comorbidades, a doença periodontal "
    "é provável (55,0%) e o hipertireoidismo é possível (22,0%). O resultado é um apoio "
    "à avaliação do médico veterinário e não a substitui."
)


def test_verificacao_de_texto_conforme(ficha):
    rs = v2.resumo(ficha)
    m = v2.verificar(TEXTO_V2, rs, ficha)
    assert m["erro"] is False
    assert m["estagio_correto"] and not m["primeiro_estagio_errado"]
    assert m["escala_trocada"] == 0
    assert m["porcentagens_sem_correspondencia"] == 0
    assert m["ausencia_como_causa"] == 0
    assert m["efeito_fora_do_resumo"] == 0
    assert m["fator_nao_clinico_citado"] == 0
    assert m["comentario_fora_do_resumo"] is False
    assert m["fator_descrito_por_gravidade"] == 0
    assert m["incerteza_omitida"] is False and m["padrao_de_registro_omitido"] is False
    assert m["recomendacao_fora_da_lista"] == 0 and m["recomendacao_omitida"] is False
    assert m["parada_omitida"] is False and m["subestadiamento_omitido"] is False
    assert m["linguagem_de_apoio"] and not m["acima_do_limite"]


def test_verificacao_acusa_desvios(ficha):
    rs = v2.resumo(ficha)
    texto = ("O estágio IRIS 2 é o estimado; o estágio 3 é muito provável (62,0%). "
             "A ausência da creatinina aumentou a probabilidade do estágio. "
             "A ureia elevada sustenta o estágio estimado, e o animal é uma fêmea castrada. "
             "O quadro é mais grave pela creatinina. Recomenda-se o hemograma. "
             "A estimativa é segura.")
    m = v2.verificar(texto, rs, ficha)
    assert m["primeiro_estagio_errado"] and m["estagio_correto"]
    assert m["escala_trocada"] == 1                       # 62% nao e "muito provavel"
    assert m["ausencia_como_causa"] == 1
    assert m["efeito_fora_do_resumo"] == 1                # ureia nao esta no resumo
    assert m["fator_nao_clinico_citado"] >= 1             # femea, castrada
    assert m["fator_descrito_por_gravidade"] == 1
    assert m["comentario_fora_do_resumo"] is True         # "segura" com creatinina medida
    assert m["recomendacao_omitida"] is True              # a UPC nao foi recomendada
    assert m["padrao_de_registro_omitido"] is True
    assert m["subestadiamento_omitido"] is True
    assert not m["linguagem_de_apoio"]


def test_verificacao_sem_creatinina_exige_a_ressalva(ficha):
    ficha["_avaliacao"]["sem_creatinina"] = True
    rs = v2.resumo(ficha)
    assert rs["creatinina_medida"] == "não"
    omite = "O estágio IRIS 3 é provável (62,0%). Recomenda-se a UPC."
    assert v2.verificar(omite, rs, ficha)["incerteza_omitida"] is True
    diz = ("O estágio IRIS 3 é provável (62,0%). A estimativa foi feita sem creatinina, "
           "o marcador que define o estágio IRIS, e por isso é menos segura.")
    assert v2.verificar(diz, rs, ficha)["incerteza_omitida"] is False


def test_verificacao_parada_quando_nao_ha_exame_util(ficha):
    ficha["recomendacao_exames"] = [{"exame": "UPC", "ganho_bits": 0.01, "reduz_duvida": True}]
    rs = v2.resumo(ficha)
    assert rs["exames_recomendados"] == []
    sem = "O estágio IRIS 3 é provável (62,0%)."
    assert v2.verificar(sem, rs, ficha)["parada_omitida"] is True
    com = "O estágio IRIS 3 é provável (62,0%). Exames adicionais pouco mudariam o estágio estimado."
    assert v2.verificar(com, rs, ficha)["parada_omitida"] is False


def test_trocas_de_escala_no_plural():
    tn = "muito provaveis sao cistite (84,5%) e doenca periodontal (66,8%)."
    assert v2._trocas_de_escala(tn) == 1
    assert v2._trocas_de_escala("provavel (55,0%) e possivel (22,0%).") == 0
    assert v2._trocas_de_escala("o mais provavel (62,0%)") == 0     # comparacao, nao escala


def test_exames_citados_por_apelido():
    assert v2._exames_citados("recomenda-se a bioquimica e a pressao sistolica") == \
        {"Bioquímica renal", "Pressão arterial"}
    assert v2._exames_citados("nada a pedir") == set()


def test_sintese_das_medidas(ficha):
    rs = v2.resumo(ficha)
    medidas = [v2.verificar(TEXTO_V2, rs, ficha), {"erro": True}]
    s = v2.sintese(medidas)
    assert s["textos"] == 1 and s["com_erro_da_api"] == 1
    assert s["estagio_correto"] == 1.0
    assert s["escala_trocada"] == 0
    assert s["incerteza_omitida"] == {"textos": 0, "de": 0}
    assert s["padrao_de_registro_omitido"] == {"textos": 0, "de": 1}


def test_prompt_v2_congelado():
    assert f"no máximo {v2.LIMITE_PALAVRAS} palavras" in v2.SISTEMA_V2
    assert "{LIMITE}" not in v2.SISTEMA_V2
    assert "não agrupe comorbidades sob um termo só" in v2.SISTEMA_V2
    assert v2.LIMIAR_GANHO_BITS == 0.10
