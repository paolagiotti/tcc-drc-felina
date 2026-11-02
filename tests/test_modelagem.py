"""Regra IRIS, adaptador do XGBoost e pipeline sem vazamento."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import config as cfg
import modelagem as mdl


# ----------------------------------------------------------------- regra IRIS

@pytest.mark.parametrize("creatinina, estagio", [
    (50.0, 1), (139.9, 1),
    (140.0, 2),            # limite superior do estagio 1 e exclusivo
    (200.0, 2), (250.0, 2),
    (250.1, 3), (440.0, 3),
    (440.1, 4), (1200.0, 4),
])
def test_regra_iris(creatinina, estagio):
    assert mdl.regra_iris(creatinina) == estagio


def test_regra_iris_ausente_devolve_ausente():
    assert np.isnan(mdl.regra_iris(np.nan))


def test_regra_iris_segue_os_limiares_da_configuracao():
    lim = cfg.LIMIARES_IRIS_UMOL
    assert mdl.regra_iris(lim[1][1] - 0.01) == 1
    assert mdl.regra_iris(lim[2][1]) == 2
    assert mdl.regra_iris(lim[3][1]) == 3
    assert mdl.regra_iris(lim[3][1] + 0.01) == 4


def test_aplicar_regra_iris_em_dataframe():
    df = pd.DataFrame({cfg.COL_CREATININA: [100.0, np.nan, 500.0]})
    r = mdl.aplicar_regra_iris(df)
    assert r.iloc[0] == 1 and np.isnan(r.iloc[1]) and r.iloc[2] == 4


# --------------------------------------------------------------- dados sinteticos

def _dados(n=80, semente=0):
    """Conjunto pequeno com rotulos 1 a 4, uma numerica com ausentes e uma categorica."""
    rng = np.random.default_rng(semente)
    y = pd.Series(rng.integers(1, 5, size=n))
    x1 = y * 100 + rng.normal(0, 30, size=n)
    x1[rng.random(n) < 0.2] = np.nan
    x2 = rng.normal(size=n)
    cat = pd.Series(np.where(rng.random(n) < 0.5, "a", "b"))
    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    return X, y


def test_separar_tipos():
    X, _ = _dados()
    num, cat = mdl.separar_tipos(X, ["x1", "x2", "cat", "inexistente"])
    assert num == ["x1", "x2"] and cat == ["cat"]


# ------------------------------------------------------------- preprocessador

def test_preprocessador_nativo_deixa_o_ausente_passar():
    X, _ = _dados()
    prep = mdl.construir_preprocessador(["x1", "x2"], ["cat"], "nativo")
    Z = prep.fit_transform(X)
    assert np.isnan(Z[:, 0]).sum() == X.x1.isna().sum()


def test_preprocessador_imputar_nao_deixa_ausente():
    X, _ = _dados()
    prep = mdl.construir_preprocessador(["x1", "x2"], ["cat"], "imputar")
    Z = prep.fit_transform(X)
    assert not np.isnan(Z).any()
    assert Z.shape[1] == 2 + 2        # duas numericas + duas categorias codificadas


def test_preprocessador_indicador_acrescenta_coluna_de_ausencia():
    X, _ = _dados()
    base = mdl.construir_preprocessador(["x1", "x2"], ["cat"], "imputar").fit_transform(X)
    ind = mdl.construir_preprocessador(["x1", "x2"], ["cat"], "indicador").fit_transform(X)
    assert ind.shape[1] == base.shape[1] + 1     # so x1 tem ausentes


@pytest.mark.parametrize("braco", ["dobro", "smote"])
def test_braco_desconhecido_e_recusado(braco):
    with pytest.raises(ValueError):
        mdl.construir_pipeline(mdl.estimadores("nenhum")["floresta"], ["x1"], [], braco)


def test_estimadores_disponiveis():
    for braco in ("peso", "nenhum"):
        assert list(mdl.estimadores(braco)) == ["floresta", "gradiente"]
    assert mdl.estimadores("peso")["floresta"].class_weight == "balanced"
    assert mdl.estimadores("nenhum")["floresta"].class_weight is None
    assert mdl.estimadores("peso")["gradiente"].class_weight == "balanced"
    assert mdl.estimadores("nenhum")["gradiente"].class_weight is None


# ---------------------------------------------------------------- XGBOrdinal

def test_xgbordinal_aceita_rotulos_de_1_a_4():
    X, y = _dados()
    est = mdl.estimadores("nenhum")["gradiente"]
    pipe = mdl.construir_pipeline(est, ["x1", "x2"], ["cat"], "nenhum", "nativo")
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert set(np.unique(pred)) <= {1, 2, 3, 4}
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(pipe.classes_) == [1, 2, 3, 4]


def test_xgbordinal_ponderacao_muda_o_ajuste():
    X, y = _dados()
    y = y.where(y != 4, 1)                    # classe 4 some; 1 fica desbalanceada
    y.iloc[:3] = 4
    com = mdl.estimadores("peso")["gradiente"].fit(X[["x1", "x2"]].fillna(0), y)
    sem = mdl.estimadores("nenhum")["gradiente"].fit(X[["x1", "x2"]].fillna(0), y)
    assert not np.allclose(com.predict_proba(X[["x1", "x2"]].fillna(0)),
                           sem.predict_proba(X[["x1", "x2"]].fillna(0)))


def test_xgbordinal_e_clonavel_pelo_sklearn():
    est = mdl.XGBOrdinal(class_weight="balanced", max_depth=2, n_estimators=5)
    copia = clone(est)
    assert copia.get_params()["max_depth"] == 2
    assert copia.get_params()["class_weight"] == "balanced"
    copia.set_params(max_depth=3, class_weight=None)
    assert copia.get_params()["max_depth"] == 3 and copia.class_weight is None


@pytest.mark.parametrize("tratamento", ["imputar", "indicador"])
def test_pipeline_com_imputacao_ajusta_e_prediz(tratamento):
    X, y = _dados()
    est = mdl.estimadores("peso")["floresta"]
    pipe = mdl.construir_pipeline(est, ["x1", "x2"], ["cat"], "peso", tratamento)
    pipe.fit(X, y)
    assert set(np.unique(pipe.predict(X))) <= {1, 2, 3, 4}


def test_xgboost_nao_depende_da_escala():
    """Por que nao ha padronizacao: no modelo do sistema, padronizar as
    variaveis nao muda nenhuma probabilidade, nem fora da amostra. (Na
    floresta, o limiar fica entre dois valores, e o arredondamento pode mudar o
    lado de um valor que cai exatamente nele.)"""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(3)
    n = 160
    y = pd.Series(rng.integers(1, 5, size=n))
    X = pd.DataFrame({                      # valores em grade, como os exames
        "idade": rng.integers(2, 22, size=n).astype(float),
        "creatinina": (y * 90 + rng.integers(-40, 40, size=n)).astype(float),
        "densidade": np.round(1.045 - y * 0.007 + rng.normal(0, 0.004, n), 3)})
    tr, te = np.arange(120), np.arange(120, n)
    esc = StandardScaler().fit(X.iloc[tr])
    Xp = pd.DataFrame(esc.transform(X), columns=X.columns)
    p = mdl.estimadores("nenhum")["gradiente"].fit(X.iloc[tr], y.iloc[tr]) \
        .predict_proba(X.iloc[te])
    pp = mdl.estimadores("nenhum")["gradiente"].fit(Xp.iloc[tr], y.iloc[tr]) \
        .predict_proba(Xp.iloc[te])
    assert np.array_equal(p, pp)
