#!/usr/bin/env bash
# Rodada final: gera, numa execucao unica e limpa, todos os numeros do escopo 1.
#
# Uso:  bash rodada_final.sh
#
# Apaga o cache, roda a triagem e os quatro modos do experimento, e copia tudo
# para uma pasta com data e hora. Os numeros do TCC devem sair daqui, e de
# nenhuma outra execucao.
set -euo pipefail

# Interpretador: python3.11 no PATH, ou o caminho dado na variavel PYTHON.
PY="${PYTHON:-python3.11}"
DESTINO="entregas/rodada_final_$(date +%Y%m%d_%H%M%S)"

echo "== conferindo pasta, arquivos e bibliotecas =="
"$PY" verificar.py || { echo; echo "Rodada interrompida: resolva os problemas acima."; exit 1; }

# (a conferencia de versoes e feita pelo verificar.py, secao 5; ele interrompe
#  a rodada se algum arquivo estiver fora do manifesto ou faltando)

echo "== limpando saidas regeraveis =="
# limpa todas as saidas regeraveis (preserva as respostas pagas da camada)
bash limpar.sh --executar >/dev/null
# (as 13 execucoes abaixo gravam em pastas distintas, por alvo, restricao e tratamento)

echo "== auditoria das colunas de origem =="
# Aponta colunas com sentinela, valor fora de faixa ou qualificador temporal nao
# aproveitado. Nao decide nada; registra o que exige conferencia humana.
"$PY" auditar_colunas.py

echo "== etapa 1: triagem =="
"$PY" triagem.py

# Duas configuracoes vao para o texto, e por isso os dois tratamentos de
# ausentes precisam ser gerados:
#   - principal: gradient boosting com tratamento nativo
#   - referencia interpretavel: regressao logistica com imputacao
# O tratamento nativo nao admite SMOTE (exige valores completos para
# interpolar) nem regressao logistica (nao aceita ausentes).

echo "== escopo 1, tratamento IMPUTADO: quatro modos =="
"$PY" experimento_iris.py --ausentes imputar
"$PY" experimento_iris.py --ausentes imputar --alvo binario
"$PY" experimento_iris.py --ausentes imputar --restrito
"$PY" experimento_iris.py --ausentes imputar --restrito --alvo binario

echo "== escopo 1, tratamento NATIVO: quatro modos =="
"$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente --bracos peso nenhum
"$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente --bracos peso nenhum --alvo binario
"$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente --bracos peso nenhum --restrito
"$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente --bracos peso nenhum --restrito --alvo binario

echo "== escopo 1, validacao aninhada (estimativa honesta) =="
# Estima o desempenho de TODO o processo de selecao, nao apenas da configuracao
# vencedora. E o numero a reportar como resultado do trabalho.
"$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente \
    --bracos peso nenhum --aninhada --criterio-selecao rps --sufixo _final_aninhada

if [ "${PARCIMONIA_ANINHADA:-0}" = "1" ]; then
    echo "== escopo 1, validacao aninhada dos conjuntos parcimoniosos =="
    # Teste de parcimonia (conjuntos de Bradley et al., 2019): analise
    # complementar, declarada como tal no texto. So roda com
    # PARCIMONIA_ANINHADA=1 bash rodada_final.sh
    "$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente \
        --bracos peso nenhum --aninhada --criterio-selecao rps \
        --conjunto-final PARCIMONIOSO_BRADLEY --sufixo _aninhada_bradley
    "$PY" experimento_iris.py --ausentes nativo --modelos floresta gradiente \
        --bracos peso nenhum --aninhada --criterio-selecao rps \
        --conjunto-final PARCIMONIOSO_MAIS --sufixo _aninhada_mais
fi

echo "== escopo 1, tratamento INDICADOR: so o alvo principal =="
# Terceiro braco do experimento de tratamento de ausentes. Roda apenas o alvo
# de quatro estagios, que e o do sistema; serve a tabela comparativa dos tres
# tratamentos, nao a selecao de configuracao.
"$PY" experimento_iris.py --ausentes indicador

echo "== interpretabilidade (SHAP) e sobrevida =="
"$PY" shap_estadiamento.py
"$PY" sobrevida_discordancias.py --pasta escopo1_nativo

echo "== escopo 2: ranqueamento de comorbidades =="
"$PY" escopo2_comorbidades.py

echo "== escopo 3: recomendacao de exames =="
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
   shap_estadiamento.py sobrevida_discordancias.py auditar_colunas.py \
   figuras.py escopo2_comorbidades.py escopo3_recomendador.py \
   fichas_pacientes.py indexar_base.py camada_interpretativa.py "$DESTINO/"
"$PY" -c "import hashlib;print(hashlib.sha256(open('Survival_data.csv','rb').read()).hexdigest())" \
    > "$DESTINO/sha256_survival_data.txt"

echo
echo "Rodada final concluida em $DESTINO"
echo "Todos os numeros do TCC devem ser tirados desta pasta."
