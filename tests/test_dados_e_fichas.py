"""Inventario do arquivo de origem (sobre um CSV sintetico) e montagem das fichas."""
import hashlib

import numpy as np
import pandas as pd
import pytest

import config as cfg
import fichas_pacientes as fp
import inventario as inv


# --------------------------------------------------------------- inventario

def test_hash_do_arquivo(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"a,b\n1,2\n")
    assert inv.hash_arquivo(p) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()


def test_carregar_bruto_le_ausentes_e_limpa_textos(tmp_path):
    p = tmp_path / "Survival_data.csv"
    p.write_text("ID, Breed ,Creatinine at diagnosis,Note\n"
                 "1,Crossbreed ,140.0,  TRUE\n"
                 "2,NA,-,\n"
                 "3,Purebreed,ND,Not recorded\n", encoding="utf-8")
    df = inv.carregar_bruto(p)
    assert list(df.columns) == ["ID", "Breed", "Creatinine at diagnosis", "Note"]
    assert df.Breed.tolist()[0] == "Crossbreed"             # espacos removidos
    assert pd.isna(df.Breed.iloc[1])                          # "NA" e ausente
    assert df["Creatinine at diagnosis"].isna().tolist() == [False, True, True]
    assert df.Note.iloc[2] == "Not recorded"                  # sentinela NAO vira ausente


@pytest.mark.parametrize("valores, tipo", [
    ([0, 1, 1, 0], "binaria"),
    ([1, 2, 3, 2], "ordinal_ou_categorica"),
    ([1.5, 2.25, 3.0, 10.0], "numerica"),
    (["1.5", "2.0", "3", "4.5"], "numerica_como_texto"),
    (["05/02/2013", "19/03/2012", "30/01/2012", "19/06/2012"], "data"),
    (["a", "b", "a", "c"], "categorica"),
    ([np.nan] * 4, "vazia"),
])
def test_inferir_tipo(valores, tipo):
    assert inv.inferir_tipo(pd.Series(valores)) == tipo


@pytest.mark.parametrize("nome, papel", [
    ("ID", "id"),
    ("Iris stage at diagnosis", "alvo_iris"),
    ("Death", "desfecho"),
    ("Censored date", "entrada"),         # a heuristica nao reconhece: fica para a revisao
    ("Benazepril prescribed", "terapia"),
    ("Hyperthyroidism", "alvo_comorbidade"),
    ("Clinical comments", "excluir"),
    ("Creatinine at diagnosis", "entrada"),
])
def test_sugerir_papel(nome, papel):
    assert inv.sugerir_papel(nome, pd.Series([1.0, 2.0])) == papel


def test_coluna_vazia_vira_excluir():
    assert inv.sugerir_papel("Qualquer coisa", pd.Series([np.nan, np.nan])) == "excluir"


def test_perfilar_uma_linha_por_coluna():
    df = pd.DataFrame({"ID": [1, 2], "Age": [10.0, np.nan], "Notes": ["x", "y"]})
    t = inv.perfilar(df)
    assert t.coluna.tolist() == ["ID", "Age", "Notes"]
    assert t.set_index("coluna").loc["Age", "pct_ausente"] == 50.0
    assert (t.papel == t.papel_sugerido).all()


# ------------------------------------------------------------------- fichas

def test_valor_da_variavel_em_portugues():
    df = pd.DataFrame({"Creatinine at diagnosis": [310.0, np.nan],
                       "USG at diagnosis": [1.0184, 1.02],
                       "Sex": ["Female", "Male"],
                       "Goitre": ["TRUE", "Not recorded"],
                       "Age": [14.0, 9.0]}, index=[7, 8])
    assert fp._valor(df, 7, "Creatinine at diagnosis") == "310 µmol/L"
    assert fp._valor(df, 8, "Creatinine at diagnosis") == "ausente"
    assert fp._valor(df, 7, "USG at diagnosis") == "1.018"
    assert fp._valor(df, 7, "Sex") == "fêmea" and fp._valor(df, 8, "Sex") == "macho"
    assert fp._valor(df, 7, "Goitre") == "sim" and fp._valor(df, 8, "Goitre") == "não registrado"
    assert fp._valor(df, 7, "Age") == "14 anos"
    assert fp._valor(df, 7, "Inexistente") == "ausente"


def test_nomes_em_portugues_cobrem_as_variaveis_do_sistema():
    from figuras import NOMES_PT
    for v in cfg.CONJUNTO_PARCIMONIOSO_MAIS:
        assert v in NOMES_PT, v
    assert set(fp.COMORB_PT) == set(cfg.COMORBIDADES_MODELAVEIS)
    for e, variaveis in cfg.EXAMES_RECOMENDAVEIS.items():
        for v in variaveis:
            assert v in NOMES_PT, (e, v)
