#!/usr/bin/env bash
# Segunda iteracao da traducao da camada interpretativa: execucao oficial.
# Le as fichas e as respostas da v1 da execucao final mais recente
# (entregas/rodada_final_*), gera o texto da v2 para os 123 gatos (usa a API;
# cerca de US$ 0,10) e copia tudo para entregas/camada_v2_<data e hora>/.
#
# Uso:  bash rodada_camada_v2.sh
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
echo "Segunda iteracao concluida em $DESTINO"
