"""Etapas da triagem sobre um quadro sintetico com os nomes reais das colunas."""
import numpy as np
import pandas as pd
import pytest

import config as cfg
import triagem as tr


def _quadro() -> pd.DataFrame:
    """Oito registros com as colunas que os criterios e as derivacoes usam."""
    df = pd.DataFrame({
        cfg.COL_ID: range(1, 9),
        cfg.COL_IRIS: [1, 2, 3, 4, np.nan, 2, 3, np.nan],
        cfg.COL_CREATININA: [100.0, 200.0, 5000.0, 600.0, np.nan, 180.0, 300.0, 20.0],
        "BUN at diagnosis": [10.0, 12.0, 30.0, 40.0, np.nan, 11.0, np.nan, 9.0],
        "Date of diagnosis": ["05/02/2013"] * 6 + ["05/02/2011", "05/02/2013"],
        "Censored date": ["18/09/2014"] * 5 + ["01/01/2012", "18/09/2014", "18/09/2014"],
        "Date hypertension diagnosed": [np.nan] * 8,
        cfg.COL_IRA: [False] * 4 + [True] + [False] * 3,
        "Heart Murmur": ["TRUE", "TRUE", "FALSE", "Not recorded", "TRUE", "FALSE", "TRUE", "FALSE"],
        "HM first recorded": ["Before CKD diagnosis", "After CKD diagnosis", np.nan, np.nan,
                              "At CKD diagnosis", np.nan, "Before CKD diagnosis", np.nan],
    })
    for achado, quando in cfg.PARES_TEMPORAIS:
        if achado not in df.columns:
            df[achado] = "FALSE"
            df[quando] = np.nan
    return df


ENTRADAS = [cfg.COL_CREATININA, "BUN at diagnosis", "Heart Murmur"]


def test_cascata_registra_removidos_e_restantes():
    c = tr.Cascata(10)
    c.registrar("criterio a", 10, 7)
    c.registrar("criterio b", 7, 7)
    t = c.tabela()
    assert t.removidos.tolist() == [0, 3, 0] and t.restantes.tolist() == [10, 7, 7]


def test_faixas_de_plausibilidade_convertem_em_ausente():
    df, tab = tr.aplicar_faixas(_quadro())
    linha = tab.set_index("coluna").loc[cfg.COL_CREATININA]
    assert linha.n_convertidos_para_ausente == 2          # 5000 e 20 umol/L
    assert np.isnan(df.loc[2, cfg.COL_CREATININA]) and np.isnan(df.loc[7, cfg.COL_CREATININA])
    assert df.loc[3, cfg.COL_CREATININA] == 600.0


def test_achado_pre_diagnostico_descarta_o_posterior():
    df, tab = tr.derivar_achados_pre_diagnostico(_quadro())
    assert "Heart Murmur_pre" in df.columns
    # TRUE antes ou no diagnostico: registros 1, 5 e 7; o registro 2 e posterior
    assert df["Heart Murmur_pre"].tolist() == [True, False, False, False, True, False, True, False]
    linha = tab.set_index("achado").loc["Heart Murmur"]
    assert linha.positivos_brutos == 4 and linha.posteriores_ao_diagnostico == 1
    assert linha.positivos_retidos == 3


def test_datas_inconsistentes_exigem_datetime():
    df = _quadro()
    with pytest.raises(TypeError):
        tr.datas_inconsistentes(df)
    df = tr.converter_tipos(df, pd.DataFrame({"coluna": [], "tipo_inferido": []}))
    ruim = tr.datas_inconsistentes(df)
    assert ruim.tolist() == [False] * 5 + [True, False, False]


def test_diagnostico_fora_da_janela():
    fora = tr.diagnostico_fora_da_janela(_quadro())
    assert fora.tolist() == [False] * 6 + [True, False]


def test_marcar_ira_booleana_e_textual():
    df = _quadro()
    assert tr.marcar_ira(df).tolist() == [False] * 4 + [True] + [False] * 3
    df[cfg.COL_IRA] = ["Positive", "Negative", "Not tested", "", "positive", "x", "y", "z"]
    assert tr.marcar_ira(df).tolist() == [True, False, False, False, True, False, False, False]


def test_criterios_de_inclusao_na_ordem():
    df = tr.converter_tipos(_quadro(), pd.DataFrame({"coluna": [], "tipo_inferido": []}))
    coorte, casc = tr.aplicar_criterios(df, ENTRADAS)
    t = casc.tabela().set_index("criterio")
    assert t.loc["Registros brutos", "restantes"] == 8
    assert t.loc["sem estagio IRIS no diagnostico", "removidos"] == 2     # 5 e 8
    assert t.loc["datas inconsistentes", "removidos"] == 1                # 6
    assert t.loc["diagnostico fora da janela do estudo", "removidos"] == 1  # 7
    assert t.loc["injuria renal aguda associada", "removidos"] == 0       # 5 ja saiu
    assert sorted(coorte[cfg.COL_ID]) == [1, 2, 3, 4]
    # a coorte das comorbidades nao exige o estagio
    coorte2, _ = tr.aplicar_criterios(df, ENTRADAS, exigir_iris=False)
    assert sorted(coorte2[cfg.COL_ID]) == [1, 2, 3, 4, 8]


def test_cobertura_informativa_desconta_sentinelas():
    cob = tr.cobertura_informativa(_quadro(), ["Heart Murmur", "BUN at diagnosis"])
    c = cob.set_index("coluna")
    assert c.loc["Heart Murmur", "n_sentinela"] == 1
    assert c.loc["Heart Murmur", "n_informativo"] == 7
    assert c.loc["BUN at diagnosis", "n_ausente"] == 2


def test_coausencia_jaccard():
    df = pd.DataFrame({"a": [np.nan, np.nan, 1.0, 1.0],
                       "b": [np.nan, 1.0, np.nan, 1.0],
                       "c": [np.nan, np.nan, 1.0, 1.0]})
    j = tr.coausencia(df, ["a", "b", "c"])
    assert j.loc["a", "c"] == 1.0
    assert j.loc["a", "b"] == pytest.approx(1 / 3, abs=1e-3)      # arredondado a 3 casas
    assert j.loc["a", "a"] == 1.0


def test_decisoes_de_entrada_trocam_pelo_derivado_e_descartam():
    df, _ = tr.derivar_achados_pre_diagnostico(_quadro())
    entradas = [cfg.COL_CREATININA, "Heart Murmur", "PCV at diagnosis", "BUN at diagnosis"]
    finais = tr.aplicar_decisoes_de_entrada(entradas, df)
    assert "Heart Murmur_pre" in finais and "Heart Murmur" not in finais
    assert "PCV at diagnosis" not in finais             # descartada por baixo conteudo
    assert finais[0] == cfg.COL_CREATININA              # a ordem e preservada


def test_indicadores_de_ausencia_so_para_colunas_com_ausencia_parcial():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0], "c": [np.nan, np.nan]})
    df, criadas = tr.derivar_indicadores_ausencia(df, ["a", "b", "c"])
    assert criadas == ["a_ausente"]
    assert df.a_ausente.tolist() == [False, True]


def test_conjuntos_de_variaveis():
    entradas = [cfg.COL_CREATININA, cfg.COL_UPC, cfg.COL_PAS, "Age", "Biochemistry"]
    conj = tr.montar_conjuntos(entradas)
    assert conj["FULL"] == entradas
    assert conj["SEM_CREATININA"] == [cfg.COL_UPC, cfg.COL_PAS, "Age", "Biochemistry"]
    assert conj["SEM_SUBESTADIAMENTO"] == [cfg.COL_CREATININA, "Age", "Biochemistry"]
    assert conj["SEM_CREATININA_NEM_SUBESTADIAMENTO"] == ["Age", "Biochemistry"]
    assert conj["SEM_INDICADORES_EXAME"] == [cfg.COL_CREATININA, cfg.COL_UPC, cfg.COL_PAS, "Age"]
    assert conj["DEFINIDOR_ESTAGIO"] == [cfg.COL_CREATININA]


def test_aritmetica_do_desenho():
    y = pd.Series([1] * 60 + [2] * 40 + [3] * 20 + [4] * 18)
    r = tr.aritmetica_desenho(y)
    assert r["menor_classe"] == 18
    assert r["n_teste_menor_classe_por_fold"] == [3, 4]       # 18 em 5 folds
    assert r["folds_desbalanceados"] is True
    assert r["k_neighbors_smote_recomendado"] == 3
    assert r["holdout_menor_classe"] == 3
    assert r["holdout_independente_viavel"] == (3 >= cfg.MIN_TESTE_POR_CLASSE)


def test_comparacao_rotulado_vs_nao_rotulado_exclui_desfecho():
    df = _quadro()
    dic = pd.DataFrame({"coluna": ["Heart Murmur", "BUN at diagnosis", "Death"],
                        "papel": ["entrada", "entrada", "desfecho"]})
    df["Death"] = [True, False, True, False, True, False, True, False]
    comp = tr.comparar_rotulado_vs_nao_rotulado(df, dic)
    assert "Death" not in set(comp.variavel)            # desfecho fica fora
    assert "Date of diagnosis" not in set(comp.variavel)
    assert "Heart Murmur" in set(comp.variavel)
    # numericas so entram com ao menos cinco valores em cada grupo
    assert cfg.COL_CREATININA not in set(comp.variavel)
