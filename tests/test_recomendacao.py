"""Componente de recomendacao: metricas de ordenacao das comorbidades e
mecanismo de reducao esperada de entropia dos exames."""
import numpy as np
import pandas as pd
import pytest

import config as cfg
import escopo2_comorbidades as e2
import escopo3_recomendador as e3


# ------------------------------------------------------------ comorbidades

def test_metricas_de_ranqueamento_calculadas_a_mao():
    Y = pd.DataFrame({"a": [1, 1, 0], "b": [0, 1, 0], "c": [1, 0, 0]})
    P = pd.DataFrame({"a": [0.9, 0.2, 0.5], "b": [0.1, 0.8, 0.4], "c": [0.5, 0.5, 0.3]})
    r = e2.metricas_ranking(Y, P)
    assert r["n_com_alguma"] == 2                 # o terceiro gato nao tem comorbidade
    # gato 1: ordem a, c, b -> presentes nas posicoes 1 e 2: P@1 = 1, AP = 1
    # gato 2: ordem b, c, a -> presentes nas posicoes 1 e 3: P@1 = 1, AP = (1 + 2/3) / 2
    assert r["p_at_1"] == 1.0
    assert r["map"] == pytest.approx((1.0 + (1 + 2 / 3) / 2) / 2)
    assert r["recall_at_1"] == pytest.approx(0.5)
    assert r["recall_at_2"] == pytest.approx((1.0 + 0.5) / 2)
    assert r["recall_at_3"] == 1.0
    assert 0 < r["ndcg"] <= 1


def test_ndcg_perfeito_e_pior():
    Y = pd.DataFrame({"a": [1], "b": [0], "c": [0]})
    assert e2.metricas_ranking(Y, pd.DataFrame({"a": [0.9], "b": [0.1], "c": [0.2]}))["ndcg"] == 1.0
    pior = e2.metricas_ranking(Y, pd.DataFrame({"a": [0.1], "b": [0.9], "c": [0.5]}))
    assert pior["ndcg"] == pytest.approx(1 / np.log2(4))
    assert pior["p_at_1"] == 0.0


def test_ranqueamento_sem_comorbidade_nao_pontua():
    Y = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
    P = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]})
    assert e2.metricas_ranking(Y, P) == {}


def test_metricas_por_alvo_pulam_alvo_constante():
    Y = pd.DataFrame({"a": [1, 0, 1, 0], "b": [0, 0, 0, 0]})
    P = pd.DataFrame({"a": [0.9, 0.1, 0.8, 0.2], "b": [0.1, 0.1, 0.1, 0.1]})
    r = e2.metricas_por_alvo(Y, P, {"a": 0.5, "b": 0.0})
    assert r["auc_a"] == 1.0 and "auc_b" not in r
    assert r["brier_a"] < r["brier_piso_a"]


def test_estrato_limita_em_dois():
    df = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 0], "c": [1, 0, 0]})
    assert e2._estrato(df, ["a", "b", "c"]).tolist() == [2, 1, 0]


def test_classificadores_independentes_com_ausentes_nativos():
    rng = np.random.default_rng(3)
    n = 60
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                      "cat": np.where(rng.random(n) < 0.5, "a", "b")})
    X.loc[rng.random(n) < 0.3, "x1"] = np.nan
    Y = pd.DataFrame({"c1": (X.x2 > 0).astype(int), "c2": (rng.random(n) < 0.4).astype(int)})
    P = e2.ajustar_independente(X.iloc[:40], Y.iloc[:40], X.iloc[40:], ["x1", "x2", "cat"])
    assert list(P.columns) == ["c1", "c2"] and len(P) == 20
    assert ((P >= 0) & (P <= 1)).all().all()


# ------------------------------------------------------------------- exames

def test_entropia_em_bits():
    assert e3.entropia(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(2.0)
    assert e3.entropia(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert e3.entropia(np.array([[0.5, 0.5, 0, 0], [1, 0, 0, 0]])).tolist() == pytest.approx([1.0, 0.0], abs=1e-6)


def test_rps_de_uma_predicao():
    assert e3.rps(np.array([1, 0, 0, 0]), 1) == 0.0
    assert e3.rps(np.array([0.5, 0, 0, 0.5]), 1) == pytest.approx(0.25)
    assert e3.rps(np.array([0.5, 0.5, 0, 0]), 1) == pytest.approx(0.25 / 3)


def test_mascarar_esconde_os_exames_nao_mantidos():
    exames = {"A": ["a1", "a2"], "B": ["b1"]}
    x = pd.Series({"a1": 1.0, "a2": 2.0, "b1": 3.0, "z": 4.0})
    m = e3.mascarar(x, exames, {"B"})
    assert np.isnan(m.a1) and np.isnan(m.a2) and m.b1 == 3.0 and m.z == 4.0
    assert x.a1 == 1.0                       # o original nao muda


def test_penalidade_e_ordenacao_com_custo():
    for e in cfg.EXAMES_RECOMENDAVEIS:
        assert 0 <= e3.penalidade(e) <= sum(cfg.PESOS_CUSTO)
    ganhos = {"UPC": 0.30, "Pressão arterial": 0.29, "Bioquímica renal": 0.05}
    assert e3.ordenar(ganhos, False) == ["UPC", "Pressão arterial", "Bioquímica renal"]
    com = e3.ordenar(ganhos, True)
    assert set(com) == set(ganhos) and len(com) == 3
    # a penalidade so reordena quando os ganhos sao proximos: a pressao, mais
    # barata e nao invasiva, passa a frente da UPC
    assert com[0] == "Pressão arterial"


class _PipeFalso:
    """Modelo de brinquedo: certeza do estagio 3 quando a creatinina existe,
    duvida total quando falta."""
    classes_ = np.array([1, 2, 3, 4])

    def __init__(self, colunas):
        self.colunas = colunas

    def predict_proba(self, X):
        out = np.full((len(X), 4), 0.25)
        tem = X["creat"].notna().to_numpy()
        out[tem] = [0.0, 0.0, 1.0, 0.0]
        return out


def _recomendador():
    colunas = ["creat", "ureia", "pas"]
    exames = {"Bioquímica": ["creat", "ureia"], "Pressão": ["pas"]}
    Xtr = pd.DataFrame({"creat": [100.0, 200.0, 300.0, 500.0, 150.0, 260.0, 450.0, 120.0],
                        "ureia": [10.0, 12.0, 20.0, 40.0, 11.0, 21.0, 35.0, 10.5],
                        "pas": [120.0, 140.0, 160.0, 180.0, 130.0, 150.0, 170.0, 125.0]})
    ytr = pd.Series([1, 2, 3, 4, 2, 3, 4, 1])
    return e3.Recomendador(_PipeFalso(colunas), colunas, Xtr, ytr, exames,
                           amostras=3, semente=0), exames


def test_banco_de_valores_por_exame_e_estagio():
    rec, exames = _recomendador()
    for e in exames:
        for s in e3.CLASSES:
            assert not rec.banco[(e, s)].empty
            assert len(rec.banco[(e, s)]) <= 3


def test_prob_na_ordem_das_classes():
    rec, _ = _recomendador()
    p = rec.prob(pd.DataFrame({"creat": [np.nan, 300.0], "ureia": [np.nan, 20.0],
                               "pas": [np.nan, 150.0]}))
    assert p[0].tolist() == [0.25] * 4 and p[1].tolist() == [0, 0, 1, 0]


def test_ganho_esperado_do_exame_que_resolve_a_duvida():
    rec, _ = _recomendador()
    x = pd.Series({"creat": np.nan, "ureia": np.nan, "pas": 150.0})
    g = rec.ganho_esperado(x, ["Bioquímica"])
    # sem creatinina a duvida e de 2 bits; com qualquer valor simulado cai a zero
    assert g["Bioquímica"] == pytest.approx(2.0)
    x_pas = pd.Series({"creat": np.nan, "ureia": np.nan, "pas": np.nan})
    assert rec.ganho_esperado(x_pas, ["Pressão"])["Pressão"] == pytest.approx(0.0)


def test_trajetoria_mede_a_cada_passo():
    rec, exames = _recomendador()
    x = pd.Series({"creat": 300.0, "ureia": 20.0, "pas": 150.0})
    t = e3.trajetoria(rec, x, 3, ["Pressão", "Bioquímica"], exames)
    assert [p["passo"] for p in t] == [0, 1, 2]
    assert t[0]["entropia"] == pytest.approx(2.0) and t[1]["entropia"] == pytest.approx(2.0)
    assert t[2]["entropia"] == pytest.approx(0.0, abs=1e-6)
    assert t[2]["acerto"] == 1 and t[2]["rps"] == 0.0
