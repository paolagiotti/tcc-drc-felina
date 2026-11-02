"""Preparacao do texto da base de conhecimento: normalizacao, deteccao de titulos
e de sumarios, divisao em trechos e busca por palavras (TF-IDF). Sem PDFs."""
import numpy as np
import pytest

import indexar_base as ib


def test_normalizar_tira_acento_e_caixa():
    assert ib.normalizar("Doença Renal Crônica") == "doenca renal cronica"
    assert ib.normalizar("IRIS") == "iris"


def test_palavras_vazias_das_duas_linguas():
    assert "ou" in ib.PALAVRAS_VAZIAS and "the" in ib.PALAVRAS_VAZIAS
    assert "nao" in ib.PALAVRAS_VAZIAS                 # normalizada, sem acento
    assert "creatinina" not in ib.PALAVRAS_VAZIAS


def test_palavras_vazias_combinam_com_o_tokenizador():
    """As contracoes da lista inglesa (can't, isn't) entram como as partes que o
    tokenizador produz (can, isn); sem isso, o scikit-learn avisa que a lista e
    o tokenizador nao combinam, e essas partes entram na busca."""
    import warnings
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        _, vec = ib.vetorizar_tfidf(["The cat can't eat and isn't well.",
                                     "O gato não come e está fraco."])
    assert not [a for a in avisos if "stop_words" in str(a.message)]
    vocab = set(vec.vocabulary_)
    assert {"can", "isn", "nao", "esta"}.isdisjoint(vocab)
    assert {"cat", "eat", "well", "gato", "come", "fraco"} <= vocab


@pytest.mark.parametrize("linha, titulo", [
    ("2. O Rim e a DRC", True),
    ("INTRODUÇÃO", True),
    ("Fisiopatologia da Doença Renal Crônica", True),
    ("a creatinina sérica permanece o marcador mais utilizado.", False),
    ("A creatinina sérica permanece o marcador mais utilizado em gatos com DRC.", False),
    ("Resultados:", False),
    ("", False),
])
def test_parece_titulo(linha, titulo):
    assert ib._parece_titulo(linha) is titulo


def test_parece_sumario():
    assert ib._parece_sumario("Introdução 7 – Objetivos 9 – Metodologia 12 – Resultados 20")
    assert ib._parece_sumario("Introdução ........ 7\nMétodos ........ 9")
    assert not ib._parece_sumario("A doença renal crônica afeta de 30% a 40% dos gatos idosos.")


def test_chave_ignora_numeros_e_espacos():
    assert ib._chave("Veterinary Record   | 2019 | p. 526") == ib._chave("Veterinary Record | 2014 | p. 7")


def test_titulo_da_lista_de_referencias():
    for l in ("REFERÊNCIAS", "Referências bibliográficas", "7. References", "Bibliografia",
              "Literature cited"):
        assert ib.TITULO_REFERENCIAS.match(l), l
    assert not ib.TITULO_REFERENCIAS.match("Referências citadas no texto")


def test_correcao_das_siglas_corrompidas():
    txt = "the CKd stage by iRiS and SdMA values; CKd4 is advanced"
    corrigido = ib._RE_FONTE.sub(lambda m: ib.CORRECOES_FONTE[m.group(1)], txt)
    assert corrigido == "the CKD stage by IRIS and SDMA values; CKD4 is advanced"


def test_empacotar_e_dividir_respeitam_o_limite():
    cabe = lambda s: len(s) <= 20
    partes = ib.dividir("Uma frase curta. Outra frase curta. Esta frase aqui e bem mais longa que o limite.", cabe)
    assert all(len(p) <= 20 for p in partes)
    assert " ".join(partes).replace("  ", " ") == (
        "Uma frase curta. Outra frase curta. Esta frase aqui e bem mais longa que o limite.")


@pytest.mark.filterwarnings("ignore:Your stop_words may be inconsistent")
def test_vetorizar_tfidf_normaliza_e_ignora_palavras_vazias():
    textos = ["a doença periodontal é frequente ou comum",
              "the periodontal disease is common",
              "hipertireoidismo felino no estágio 3"]
    m, vec = ib.vetorizar_tfidf(textos)
    assert m.shape[0] == 3 and np.allclose(np.linalg.norm(m, axis=1), 1.0)
    vocab = set(vec.vocabulary_)
    assert "periodontal" in vocab and "ou" not in vocab and "the" not in vocab
    # termo de um caractere (o numero do estagio) fica fora da busca
    assert "estagio" in vocab and "3" not in vocab
    # a palavra em comum aproxima os dois primeiros textos
    assert m[0] @ m[1] > m[0] @ m[2] == pytest.approx(0.0)
