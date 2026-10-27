"""Metricas, splits compartilhados e comparacao pareada."""
import numpy as np
import pandas as pd
import pytest

import avaliacao as av
import config as cfg


@pytest.fixture(autouse=True)
def quatro_classes():
    av.definir_classes(pd.Series([1, 2, 3, 4]))


def test_acuracia_adjacente():
    assert av.acuracia_adjacente([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert av.acuracia_adjacente([1, 2, 3, 4], [2, 3, 4, 1]) == 0.75
    assert av.acuracia_adjacente([1, 4], [4, 1]) == 0.0


def test_metricas_perfeitas():
    y = [1, 2, 3, 4, 2, 3]
    m = av.metricas(y, y)
    assert m["acuracia_balanceada"] == 1.0
    assert m["kappa_quadratico"] == 1.0
    assert m["f1_macro"] == 1.0
    for c in (1, 2, 3, 4):
        assert m[f"sensibilidade_e{c}"] == 1.0 and m[f"precisao_e{c}"] == 1.0


def test_metricas_classe_ausente_no_fold_vira_nan():
    m = av.metricas([1, 2, 2, 3], [1, 2, 3, 3])
    assert np.isnan(m["sensibilidade_e4"])
    assert m["sensibilidade_e2"] == 0.5


def test_classes_binarias():
    av.definir_classes(pd.Series([1, 2]))
    m = av.metricas([1, 1, 2, 2], [1, 2, 2, 2])
    assert "sensibilidade_e3" not in m
    assert m["sensibilidade_e1"] == 0.5 and m["sensibilidade_e2"] == 1.0


def test_brier_e_rps_de_predicao_perfeita_sao_zero():
    proba = np.eye(4)[[0, 1, 2, 3]]
    m = av.metricas_probabilidade([1, 2, 3, 4], proba, [1, 2, 3, 4])
    assert m["brier_multiclasse"] == 0.0 and m["rps"] == 0.0
    assert m["brier_piso_prevalencia"] > 0 and m["rps_piso_prevalencia"] > 0


def test_rps_penaliza_mais_o_erro_distante():
    """Mesma massa de probabilidade: no estagio vizinho custa menos que no oposto."""
    real = [1]
    vizinho = np.array([[0.5, 0.5, 0.0, 0.0]])
    oposto = np.array([[0.5, 0.0, 0.0, 0.5]])
    r_viz = av.metricas_probabilidade(real, vizinho, [1, 2, 3, 4])
    r_opo = av.metricas_probabilidade(real, oposto, [1, 2, 3, 4])
    # Brier ignora a ordem: igual nos dois; RPS respeita a ordem
    assert r_viz["brier_multiclasse"] == pytest.approx(r_opo["brier_multiclasse"])
    assert r_viz["rps"] < r_opo["rps"]
    # valores a mao: acumuladas (0,5; 0,5; 0,5) contra (1; 1; 1) -> 3 x 0,25 / 3
    assert r_opo["rps"] == pytest.approx(0.25)
    assert r_viz["rps"] == pytest.approx((0.25 + 0 + 0) / 3)


def test_brier_uniforme():
    proba = np.full((2, 4), 0.25)
    m = av.metricas_probabilidade([1, 3], proba, [1, 2, 3, 4])
    # (0,75^2 + 3 x 0,25^2) = 0,75
    assert m["brier_multiclasse"] == pytest.approx(0.75)


def test_splits_sao_deterministicos_e_estratificados():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=100)})
    y = pd.Series([1] * 40 + [2] * 30 + [3] * 20 + [4] * 10)
    a = av.materializar_splits(X, y)
    b = av.materializar_splits(X, y)
    assert len(a) == cfg.N_FOLDS * cfg.N_REPETICOES
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
    for tr, te in a:
        assert len(set(tr) & set(te)) == 0
        assert y.iloc[te].value_counts().loc[4] == 2      # 10 da classe 4 em 5 folds


def test_consolidar_oof_usa_a_moda():
    oof = pd.DataFrame({"rep0": [1, 2, 3], "rep1": [1, 3, 3], "rep2": [2, 3, 4]}, dtype=float)
    assert av.consolidar_oof(oof).tolist() == [1, 3, 3]


def test_resumir_mediana_e_faixa():
    por_fold = pd.DataFrame({"fold": [0, 1, 2], "repeticao": [0, 0, 0],
                             "acc": [0.5, 0.7, 0.6], "vazia": [np.nan] * 3})
    r = av.resumir(por_fold)
    assert r["acc"] == 0.6 and r["acc_min"] == 0.5 and r["acc_max"] == 0.7
    assert np.isnan(r["vazia"])


def test_comparar_pareado():
    a = pd.DataFrame({"m": [0.6, 0.7, 0.8]})
    b = pd.DataFrame({"m": [0.5, 0.7, 0.6]})
    d = av.comparar_pareado(a, b, "m")
    assert d["n_folds"] == 3 and d["folds_a_favor"] == 2 and d["folds_contra"] == 0
    assert d["delta_mediano"] == pytest.approx(0.1)
    assert d["inclui_zero"] is True          # ha um empate


def test_matriz_confusao_tem_as_quatro_classes():
    m = av.matriz_confusao([1, 2, 2], [1, 3, 2])
    assert m.shape == (4, 4)
    assert m.loc["real_e2", "pred_e3"] == 1 and m.loc["real_e4"].sum() == 0


def test_avaliar_roda_validacao_cruzada_com_pipeline_simples():
    from sklearn.dummy import DummyClassifier
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"x": rng.normal(size=60)})
    y = pd.Series([1] * 20 + [2] * 20 + [3] * 10 + [4] * 10)
    av.definir_classes(y)
    splits = av.materializar_splits(X, y, n_splits=5, n_repeats=2)
    por_fold, oof = av.avaliar(lambda y_tr: DummyClassifier(strategy="prior"), X, y, splits)
    assert len(por_fold) == 10
    assert set(por_fold.repeticao) == {0, 1}
    assert oof.shape == (60, 2) and not oof.isna().any().any()
    assert "rps" in por_fold.columns
