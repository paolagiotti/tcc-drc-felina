#!/usr/bin/env bash
# Execucao final: roda a cadeia inteira numa execucao unica e limpa e copia as
# saidas, os scripts e o manifesto para entregas/rodada_final_<data e hora>/.
# Os numeros do texto saem dessa pasta, e de nenhuma outra execucao.
#
# Uso:  bash rodada_final.sh
set -euo pipefail

# Interpretador: python3.11 no PATH, ou o caminho dado na variavel PYTHON.
PY="${PYTHON:-python3.11}"
DESTINO="entregas/rodada_final_$(date +%Y%m%d_%H%M%S)"

echo "== conferindo pasta, arquivos e bibliotecas =="
"$PY" verificar.py || { echo; echo "Rodada interrompida: resolva os problemas acima."; exit 1; }

# (a conferencia de versoes e feita pelo verificar.py, secao 5; ele interrompe
#  a rodada se algum arquivo estiver fora do manifesto ou faltando)

echo "== limpando saidas regeraveis =="
# (preserva as respostas pagas em saida/relatorios/camada)
rm -rf saida/cache saida/base saida/relatorios/escopo1* saida/relatorios/escopo2 \
       saida/relatorios/escopo3 saida/figuras/tcc saida/relatorios/camada/prompts \
       saida/relatorios/camada/fichas.jsonl
mkdir -p saida/relatorios

echo "== auditoria das colunas de origem =="
# Aponta colunas com sentinela, valor fora de faixa ou qualificador temporal nao
# aproveitado. Nao decide nada; registra o que exige conferencia humana.
"$PY" auditar_colunas.py

echo "== etapa 1: triagem =="
"$PY" triagem.py

# Estadiamento: cada tratamento de ausentes em quatro modos (quatro estagios ou
# binario; coorte completa ou so os gatos com creatinina).

echo "== estadiamento, tratamento IMPUTADO: quatro modos =="
"$PY" experimento_iris.py --ausentes imputar
"$PY" experimento_iris.py --ausentes imputar --alvo binario
"$PY" experimento_iris.py --ausentes imputar --restrito
"$PY" experimento_iris.py --ausentes imputar --restrito --alvo binario

echo "== estadiamento, tratamento NATIVO: quatro modos =="
"$PY" experimento_iris.py --ausentes nativo
"$PY" experimento_iris.py --ausentes nativo --alvo binario
"$PY" experimento_iris.py --ausentes nativo --restrito
"$PY" experimento_iris.py --ausentes nativo --restrito --alvo binario

echo "== estadiamento, validacao aninhada =="
# Estima o desempenho de todo o processo de selecao: e o numero reportado.
"$PY" experimento_iris.py --ausentes nativo --aninhada --sufixo _final_aninhada

echo "== estadiamento, tratamento INDICADOR: so o alvo de quatro estagios =="
"$PY" experimento_iris.py --ausentes indicador

echo "== interpretabilidade (SHAP) =="
"$PY" shap_estadiamento.py

echo "== recomendacao: comorbidades e exames =="
"$PY" escopo2_comorbidades.py
"$PY" escopo3_recomendador.py

echo "== camada interpretativa: fichas dos pacientes =="
"$PY" fichas_pacientes.py

if ls base_conhecimento/*.pdf >/dev/null 2>&1; then
    echo "== camada interpretativa: indexacao da base =="
    "$PY" indexar_base.py
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        echo "== camada interpretativa: geracao e avaliacao (usa a API; custa) =="
        "$PY" camada_interpretativa.py
    else
        echo "   OPENAI_API_KEY nao definida: geracao pulada."
    fi
else
    echo "   base_conhecimento/ sem PDFs: indexacao e geracao puladas."
fi

echo "== figuras do capitulo de resultados =="
"$PY" figuras.py

echo "== copiando para $DESTINO =="
mkdir -p "$DESTINO"  # dentro de entregas/
cp -r saida/relatorios "$DESTINO/"
cp -r saida/dados "$DESTINO/"
cp -r saida/figuras "$DESTINO/" 2>/dev/null || true
cp config.py requirements.txt MANIFESTO.txt rodada_final.sh verificar.py \
   dados/dicionario_variaveis.csv "$DESTINO/"
if [ -f saida/base/meta.json ]; then     # o que a indexacao extraiu e descartou
    cp saida/base/meta.json "$DESTINO/base_conhecimento_meta.json"
fi
cp modelagem.py avaliacao.py triagem.py experimento_iris.py \
   shap_estadiamento.py auditar_colunas.py \
   figuras.py escopo2_comorbidades.py escopo3_recomendador.py \
   fichas_pacientes.py indexar_base.py camada_interpretativa.py inventario.py "$DESTINO/"
"$PY" -c "import hashlib;print(hashlib.sha256(open('Survival_data.csv','rb').read()).hexdigest())" \
    > "$DESTINO/sha256_survival_data.txt"

echo
echo "Execucao final concluida em $DESTINO"
echo "Os numeros do texto saem desta pasta."
