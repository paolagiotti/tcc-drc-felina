#!/usr/bin/env bash
# Versao 2 da traducao da camada interpretativa: execucao oficial.
#
# Uso:  bash rodada_camada_v2.sh
#
# Le as fichas e as respostas da v1 da rodada final mais recente
# (entregas/rodada_final_*), gera o texto da v2 para os 123 gatos (usa a API;
# cerca de US$ 0,10) e copia tudo para entregas/camada_v2_AAAAMMDD_HHMMSS/.
# Os numeros da v2 saem dessa pasta; os da v1 e dos tres escopos continuam na
# rodada final.
set -euo pipefail

# Interpretador: python3.11 no PATH, ou o caminho dado na variavel PYTHON.
PY="${PYTHON:-python3.11}"
DESTINO="entregas/camada_v2_$(date +%Y%m%d_%H%M%S)"

echo "== conferindo pasta, arquivos e bibliotecas =="
"$PY" verificar.py || { echo; echo "Interrompido: resolva os problemas acima."; exit 1; }

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY nao definida no ambiente. Nada foi gerado."
    exit 1
fi

echo "== camada interpretativa, versao 2 (usa a API; custa) =="
"$PY" camada_v2.py

echo "== copiando para $DESTINO =="
mkdir -p "$DESTINO"  # dentro de entregas/
cp -r saida/relatorios/camada_v2/. "$DESTINO/"
cp camada_v2.py camada_interpretativa.py config.py requirements.txt MANIFESTO.txt \
   verificar.py rodada_camada_v2.sh "$DESTINO/"

echo
echo "Versao 2 concluida em $DESTINO"
echo "Os numeros da v2 devem ser tirados desta pasta."
