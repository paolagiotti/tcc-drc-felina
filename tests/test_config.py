"""Coerencia interna da configuracao e do dicionario de variaveis."""
import pandas as pd

import config as cfg


def test_limiares_iris_sao_contiguos():
    lim = cfg.LIMIARES_IRIS_UMOL
    assert list(lim) == [1, 2, 3, 4]
    assert lim[1] == (None, 140.0)
    assert lim[4][1] is None
    for e in (1, 2, 3):
        assert lim[e][1] == lim[e + 1][0]


def test_configuracao_do_sistema_e_coerente():
    cs = cfg.CONFIGURACAO_SISTEMA
    assert cs["conjunto"] == cfg.CONJUNTO_SISTEMA
    assert cs["modelo"] in {"logistica", "floresta", "gradiente"}
    assert cs["braco"] in {"peso", "smote", "nenhum"}


def test_exames_recomendaveis_e_custos_batem():
    exames = set(cfg.EXAMES_RECOMENDAVEIS)
    assert sorted(cfg.ORDEM_FIXA_EXAMES) == sorted(exames)
    assert set(cfg.CUSTO_EXAMES) == exames
    for custos in cfg.CUSTO_EXAMES.values():
        assert set(custos) == {"custo", "invasividade", "indisponibilidade"}
        assert all(0.0 <= v <= 1.0 for v in custos.values())
    assert len(cfg.PESOS_CUSTO) == 3
    assert 0.0 <= cfg.LAMBDA_CUSTO <= 1.0


def test_variaveis_dos_exames_tem_faixa_de_plausibilidade():
    for variaveis in cfg.EXAMES_RECOMENDAVEIS.values():
        for v in variaveis:
            if v == "Dipstick protein at diagnosis":   # ordinal, sem faixa
                continue
            assert v in cfg.FAIXAS_PLAUSIVEIS, v


def test_comorbidades_modelaveis_tem_qualificador_temporal():
    achados = {a for a, _ in cfg.PARES_TEMPORAIS}
    assert set(cfg.COMORBIDADES_MODELAVEIS) <= achados


def test_conjuntos_parcimoniosos():
    assert set(cfg.CONJUNTO_PARCIMONIOSO) < set(cfg.CONJUNTO_PARCIMONIOSO_MAIS)
    assert cfg.COL_CREATININA in cfg.CONJUNTO_PARCIMONIOSO
    assert {cfg.COL_UPC, cfg.COL_PAS} <= set(cfg.CONJUNTO_PARCIMONIOSO_MAIS)
    assert not set(cfg.DESCARTAR_POR_BAIXO_CONTEUDO) & set(cfg.CONJUNTO_PARCIMONIOSO_MAIS)


def test_papeis():
    assert cfg.PAPEIS_FORA_DO_MODELO < cfg.PAPEIS_VALIDOS
    assert {"entrada", "alvo_iris", "alvo_comorbidade"} <= cfg.PAPEIS_VALIDOS


def test_versao_do_modelo_de_linguagem():
    assert cfg.VERSAO_LLM.startswith(cfg.MODELO_LLM)
    assert cfg.K_TRECHOS in (3, 4, 5)
    assert len(cfg.PRECO_LLM) == 2


def test_desenho_de_validacao():
    assert cfg.N_FOLDS == 5 and cfg.N_REPETICOES == 3
    assert 0 < cfg.FRACAO_HOLDOUT < 1
    assert 0 < cfg.PROP_MINIMA_PREENCHIDA <= 1


def test_dicionario_de_variaveis_e_coerente_com_a_configuracao():
    dic = pd.read_csv(cfg.DICIONARIO)
    assert {"coluna", "papel", "tipo_inferido"} <= set(dic.columns)
    assert dic.coluna.is_unique
    assert set(dic.papel.dropna()) <= cfg.PAPEIS_VALIDOS
    papel = dict(zip(dic.coluna, dic.papel))
    assert papel[cfg.COL_ID] == "id"
    assert papel[cfg.COL_IRIS] == "alvo_iris"
    for col in (cfg.COL_CREATININA, cfg.COL_UPC, cfg.COL_PAS):
        assert papel[col] == "entrada", col
    for c in cfg.COMORBIDADES_MODELAVEIS:
        assert papel[c] == "alvo_comorbidade", c
    for achado, quando in cfg.PARES_TEMPORAIS:
        assert papel[achado] in {"entrada", "alvo_comorbidade"}, achado
        # o qualificador temporal nunca e entrada: so serve para derivar o `_pre`
        assert papel[quando] in {"derivada", "excluir"}, quando
    for c in cfg.INDICADORES_EXAME_REALIZADO + cfg.DESCARTAR_POR_BAIXO_CONTEUDO:
        assert papel[c] == "entrada", c
    for col in cfg.COLUNAS_DATA + [cfg.COL_IRA]:
        assert col in papel, col
    # o dicionario foi revisado a mao: a coluna papel nao e a sugestao automatica
    assert "papel_sugerido" in dic.columns
    assert (dic.papel != dic.papel_sugerido).any()
