#!/usr/bin/env bash
# Limpeza do dia a dia: apaga saidas regeraveis.
#
#   bash limpar.sh                    mostra o que apagaria
#   bash limpar.sh --executar         apaga as saidas regeraveis
#   bash limpar.sh --tudo --executar  inclui as respostas do modelo de
#                                     linguagem (CUSTARAM DINHEIRO) e o indice
#                                     da base de conhecimento
#
# NUNCA toca: codigo, dados de origem, dados/dicionario_variaveis.csv,
# base_conhecimento/ (os PDFs) e entregas/ (as rodadas finais).
set -uo pipefail

EXECUTAR=0; TUDO=0
for a in "$@"; do
    case "$a" in
        --executar) EXECUTAR=1 ;;
        --tudo)     TUDO=1 ;;
        *) echo "opcao desconhecida: $a"; exit 1 ;;
    esac
done

alvos=()
add() { [ -e "$1" ] && alvos+=("$1"); }

# regeraveis, sem custo
add saida/cache
for d in saida/relatorios/escopo1*; do [ -d "$d" ] && alvos+=("$d"); done
add saida/relatorios/escopo2
add saida/relatorios/escopo3
add saida/figuras/tcc
add saida/relatorios/camada/prompts
add saida/relatorios/camada/fichas.jsonl

# com custo: so com --tudo
caros=(saida/relatorios/camada/respostas.jsonl
       saida/relatorios/camada/resumo_avaliacao.json
       saida/relatorios/camada/revisao_qualitativa.md
       saida/base)
if [ "$TUDO" = "1" ]; then
    for c in "${caros[@]}"; do add "$c"; done
fi

if [ ${#alvos[@]} -eq 0 ]; then
    echo "Nada a limpar."
else
    echo "Seriam removidos:"
    for a in "${alvos[@]}"; do printf "  %-50s %6s\n" "$a" "$(du -sh "$a" 2>/dev/null | cut -f1)"; done
fi

echo
echo "PRESERVADOS: codigo, dados de origem, dados/dicionario_variaveis.csv,"
echo "             base_conhecimento/ (PDFs), entregas/ (rodadas finais)"
if [ "$TUDO" = "0" ]; then
    presentes=()
    for c in "${caros[@]}"; do [ -e "$c" ] && presentes+=("$c"); done
    if [ ${#presentes[@]} -gt 0 ]; then
        echo "             e, por terem custo, sem --tudo:"
        for c in "${presentes[@]}"; do echo "               $c"; done
    fi
fi

if [ "$EXECUTAR" = "1" ] && [ ${#alvos[@]} -gt 0 ]; then
    for a in "${alvos[@]}"; do rm -rf "$a"; done
    echo
    echo "${#alvos[@]} itens removidos."
elif [ ${#alvos[@]} -gt 0 ]; then
    echo
    echo "SIMULACAO. Para apagar:  bash limpar.sh --executar"
fi
