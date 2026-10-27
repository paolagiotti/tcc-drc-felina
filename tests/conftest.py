"""Configuracao comum dos testes.

Os scripts ficam na raiz do projeto e se importam pelo nome (import config),
entao a raiz entra no caminho de busca. Nenhum teste le o arquivo de origem,
chama a API da OpenAI ou carrega o modelo de embeddings: tudo roda sobre dados
sinteticos pequenos.
"""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture
def ficha() -> dict:
    """Ficha de um gato no formato gravado por fichas_pacientes.py."""
    return {
        "id": 101,
        "_avaliacao": {"estagio_registrado": 3, "sem_creatinina": False},
        "idade": "14 anos",
        "sexo": "fêmea",
        "estagio_predito": 3,
        "probabilidades": {"1": 0.05, "2": 0.25, "3": 0.62, "4": 0.08},
        "fatores": [
            {"variavel": "Creatinina", "valor": "310 µmol/L", "direcao": "a favor",
             "contribuicao": 1.2},
            {"variavel": "Idade", "valor": "14 anos", "direcao": "a favor",
             "contribuicao": 0.4},
            {"variavel": "UPC", "valor": "ausente", "direcao": "contra",
             "contribuicao": -0.3},
            {"variavel": "Sexo", "valor": "fêmea", "direcao": "contra",
             "contribuicao": -0.1},
            {"variavel": "Bócio", "valor": "não", "direcao": "a favor",
             "contribuicao": 0.05},
        ],
        "exames_disponiveis": ["Bioquímica renal", "Urinálise"],
        "exames_faltantes": ["UPC", "Pressão arterial"],
        "recomendacao_exames": [
            {"exame": "UPC", "ganho_bits": 0.31, "reduz_duvida": True},
            {"exame": "Pressão arterial", "ganho_bits": 0.02, "reduz_duvida": True},
        ],
        "comorbidades": [
            {"nome": "doença periodontal", "probabilidade": 0.55},
            {"nome": "hipertireoidismo", "probabilidade": 0.22},
            {"nome": "cistite/DTUIF", "probabilidade": 0.08},
            {"nome": "artrite", "probabilidade": 0.04},
        ],
    }
