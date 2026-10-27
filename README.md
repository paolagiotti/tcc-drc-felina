# Apoio à decisão clínica para a doença renal crônica felina

Código do Trabalho de Conclusão de Curso *Sistema computacional de apoio à
decisão clínica para interpretação laboratorial e estadiamento da doença renal
crônica em felinos* (Paola Giotti Cardoso, Engenharia de Software, Unisinos,
2026; orientação da Profa. Janaina Conceição Sutil Lemos).

É uma prova de conceito. O sistema recebe os dados de rotina de um gato com
suspeita de doença renal crônica (DRC), inclusive com exames faltando, e devolve
ao médico veterinário um apoio em três partes, com pesos diferentes:

1. **Componente de estadiamento**, o núcleo do trabalho: classifica o paciente
   nos estágios I a IV da diretriz IRIS por aprendizado supervisionado sobre 18
   variáveis de rotina, com a ausência de exames tratada como informação, e
   explica cada predição por valores de Shapley (SHAP) calculados fora da amostra.
   É o componente avaliado em profundidade.
2. **Componente de recomendação**, avaliado de forma exploratória, com duas
   saídas: o ranqueamento das comorbidades concomitantes (quatro classificadores,
   um por comorbidade) e a sugestão dos exames complementares que mais reduziriam
   a dúvida do modelo sobre o estágio (redução esperada de entropia).
3. **Camada interpretativa**: um modelo de linguagem traduz as saídas em um texto
   curto para o veterinário e as contextualiza com trechos de nove documentos de
   referência (geração aumentada por recuperação, RAG), com citação do documento
   e da página. O modelo de linguagem só redige: não calcula nem altera nenhuma
   predição. A camada foi conferida por verificação automática, sem avaliação
   por veterinários.

Os dados são os 625 gatos do conjunto `Survival_data` de Conroy e O'Neill
(2019), do programa VetCompass (Reino Unido): 123 têm estágio IRIS registrado e
formam a coorte do estadiamento; 618 formam a coorte das comorbidades. O
conjunto de dados **não está neste repositório** (ver [Dados](#dados)).

## Como o código está organizado

Um script por etapa, na raiz do projeto, todos lendo as decisões de um único
arquivo de configuração. A ordem abaixo é a ordem de execução.

| Arquivo | O que faz | Depende de |
|---|---|---|
| `config.py` | Todas as decisões parametrizadas: colunas, faixas, critérios, conjuntos de variáveis, configuração do sistema, exames, comorbidades e parâmetros da camada | |
| `inventario.py` | Perfila as 121 colunas do arquivo de origem e gera o rascunho do dicionário de variáveis (revisado à mão em `dados/dicionario_variaveis.csv`) | arquivo de origem |
| `auditar_colunas.py` | Aponta sentinelas textuais, valores fora de faixa e qualificadores temporais não aproveitados; não decide nada | arquivo de origem, dicionário |
| `triagem.py` | Aplica os critérios de inclusão e gera as duas coortes, os mapas de ausência e os conjuntos de variáveis | arquivo de origem, dicionário |
| `modelagem.py` | Pipeline sem vazamento (imputação, codificação e balanceamento dentro de cada partição), estimadores e regra IRIS | biblioteca |
| `avaliacao.py` | Partições compartilhadas, métricas (acurácia balanceada, kappa quadrático, RPS, Brier contra pisos) e comparação pareada | biblioteca |
| `experimento_iris.py` | **Estadiamento**: pisos, modelos, tratamentos de ausentes, ablações, seleção por RPS em validação aninhada | `triagem.py` |
| `shap_estadiamento.py` | Valores de Shapley fora da amostra, globais e por gato | `triagem.py` |
| `sobrevida_discordancias.py` | Análise exploratória de sobrevida das discordâncias entre o modelo e a regra IRIS | `experimento_iris.py` (tratamento nativo) |
| `escopo2_comorbidades.py` | **Recomendação, saída 1**: ranqueamento de comorbidades (formulações independente e em cadeia) | `triagem.py` |
| `escopo3_recomendador.py` | **Recomendação, saída 2**: exames por redução esperada de entropia, com simulação retrospectiva | `triagem.py` |
| `fichas_pacientes.py` | Ficha estruturada de cada gato, entrada da camada interpretativa | os três anteriores |
| `indexar_base.py` | Extrai o texto dos PDFs, divide em trechos e janelas e grava o índice vetorial (FAISS) | PDFs em `base_conhecimento/` |
| `camada_interpretativa.py` | Tradução, contextualização (RAG), comparação com o modelo sem recuperação e verificação automática | fichas, índice, chave da API |
| `camada_v2.py` | Segunda iteração da tradução: o código prepara um resumo clínico da ficha, o modelo redige, e a verificação é aplicada às duas versões | fichas e respostas da execução final, chave da API |
| `figuras.py`, `figuras_extra.py` | Figuras do capítulo de resultados, geradas das saídas da execução final | experimentos |

Ferramentas:

| Arquivo | Uso |
|---|---|
| `verificar.py` | Conferência da pasta e do ambiente: manifesto, arquivos sobrando ou faltando, SHA-256 do arquivo de origem, versões das bibliotecas; `--api` testa o acesso ao modelo de linguagem |
| `rodada_final.sh` | Execução oficial: triagem, estadiamento, recomendação, fichas, camada e figuras, gravados em `entregas/rodada_final_<data>/`. Os números do texto saem dessa pasta |
| `rodada_camada_v2.sh` | Execução oficial da segunda iteração da tradução, gravada em `entregas/camada_v2_<data>/` |
| `limpar.sh` | Apaga as saídas regeráveis de `saida/`; preserva as respostas pagas, os dados e as execuções finais |
| `requirements.txt` | Versões fixadas das bibliotecas |
| `MANIFESTO.txt` | SHA-256 de cada arquivo do projeto, conferido por `verificar.py` |
| `tests/` | Testes automatizados (ver [Testes](#testes)) |

Pastas criadas na execução: `saida/` (trabalho), `entregas/` (execuções finais)
e `base_conhecimento/` (PDFs). Nenhuma delas vai para o repositório.

## Dados

O arquivo de origem é o `Survival_data.csv` do conjunto *Chronic kidney disease
in cats attending primary-care practice in the UK: a VetCompass study* (Conroy,
M.; O'Neill, D. G., 2019; DOI 10.34840/n76c-mp81), distribuído sob a licença
CC BY-NC-ND 4.0. A licença autoriza o uso não comercial com atribuição e não
autoriza redistribuir versões modificadas; por isso o repositório não contém o
arquivo nem versões processadas dele (coortes, predições por gato, fichas). Para
reproduzir:

1. Obtenha o `Survival_data.csv` no repositório do Royal Veterinary College e
   coloque-o na raiz do projeto (ou aponte o caminho na variável de ambiente
   `DRC_BRUTO`). O `verificar.py` confere o SHA-256 do arquivo contra o usado na
   execução final.
2. `dados/dicionario_variaveis.csv` é o dicionário de variáveis revisado à mão
   (papel de cada coluna: entrada, alvo, desfecho, terapia, derivada ou
   excluir). É o que o código lê. A coluna de valores de exemplo, gerada pelo
   inventário, foi removida da versão publicada por conter valores do conjunto
   de dados.
3. Para a camada interpretativa, coloque em `base_conhecimento/` os nove
   documentos citados na metodologia, com o nome de arquivo igual à chave de
   citação do texto (por exemplo, `sparkes2016.pdf`). As três monografias têm a
   página inicial do texto técnico configurada em `indexar_base.INICIO_TEXTO`.
   Os PDFs são material de terceiros e também ficam fora do repositório.

## Instalação

Python 3.11. As versões das bibliotecas estão fixadas: a mesma semente com
versões diferentes de XGBoost e scikit-learn produz números diferentes.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O `sentence-transformers` baixa o PyTorch e só é necessário para a camada
interpretativa; o mesmo vale para `pymupdf`, `faiss-cpu` e `openai`. A geração
dos textos usa a API da OpenAI, com a chave na variável de ambiente
`OPENAI_API_KEY` (nunca em arquivo do projeto). O modelo é fixado em
`config.py` (`MODELO_LLM` e `VERSAO_LLM`): a cada chamada o código confere a
versão servida e interrompe a execução se ela for outra.

## Execução

Tudo roda de dentro da pasta do projeto. Antes de qualquer coisa:

```bash
python verificar.py
```

Execução completa, de onde saem os números do texto (cerca de 40 minutos; a
camada interpretativa só entra se houver PDFs em `base_conhecimento/` e a
chave da API definida):

```bash
bash rodada_final.sh
```

Por componente, para desenvolvimento:

| Etapa | Comando | Custo |
|---|---|---|
| Triagem (só quando mudam dados ou critérios) | `python triagem.py` | nenhum |
| Estadiamento, validação aninhada | `python experimento_iris.py --ausentes nativo --modelos floresta gradiente --bracos peso nenhum --aninhada` | nenhum |
| Comorbidades | `python escopo2_comorbidades.py` | nenhum |
| Exames | `python escopo3_recomendador.py` | nenhum |
| Fichas dos pacientes | `python fichas_pacientes.py` | nenhum |
| Índice da base de conhecimento | `python indexar_base.py` | nenhum |
| Camada: só os prompts, sem chamar a API | `python camada_interpretativa.py --simular` | nenhum |
| Camada: cinco gatos | `python camada_interpretativa.py --limite 5` | cerca de 1 centavo de dólar |
| Camada: os 123 gatos | `python camada_interpretativa.py` | cerca de US$ 0,30 |
| Tradução, segunda iteração: simulação | `python camada_v2.py --simular` | nenhum |
| Tradução, segunda iteração: piloto de cinco gatos | `python camada_v2.py --piloto` | cerca de 1 centavo de dólar |
| Tradução, segunda iteração: execução oficial | `bash rodada_camada_v2.sh` | cerca de US$ 0,10 |

Depois de `--simular`, os arquivos de `saida/relatorios/camada/prompts/` são
exatamente o que o modelo de linguagem receberia, e
`saida/relatorios/camada/recuperacao.json` mostra os trechos que cada consulta
trouxe. A segunda iteração lê as fichas e as respostas da execução final mais
recente em `entregas/`, nunca de `saida/`.

Os scripts de execução usam `python3.11`; outro interpretador pode ser indicado
na variável `PYTHON`.

## Testes

Os testes cobrem as regras e os mecanismos que não dependem dos dados nem da
API: a coerência da configuração e do dicionário, a regra IRIS e o adaptador do
XGBoost para rótulos de 1 a 4, as métricas (RPS, Brier, kappa, métricas de
ordenação), os critérios e as derivações da triagem sobre quadros sintéticos, a
redução esperada de entropia com um modelo de brinquedo, as consultas e a
recuperação alternada da camada, a verificação automática dos textos nas duas
versões e a preparação do texto da base de conhecimento.

```bash
pip install pytest
pytest
```

Nenhum teste executa a cadeia completa; isso é papel de `rodada_final.sh`.

## Reprodutibilidade

- Versões fixadas em `requirements.txt`; semente fixa (`config.SEMENTE`).
- SHA-256 do arquivo de origem conferido antes de cada execução.
- `MANIFESTO.txt` com o SHA-256 de cada arquivo do projeto, conferido antes de
  cada execução; `python verificar.py --atualizar-manifesto` registra uma edição
  intencional.
- Execução final única: as saídas, o manifesto e os scripts são copiados para
  `entregas/rodada_final_<data>/`, e os números do texto saem dessa pasta.
- Versão do modelo de linguagem declarada e conferida em cada chamada.

Os scripts foram revisados depois da execução final (comentários, docstrings e
mensagens); a lógica não mudou. O manifesto deste repositório registra a versão
publicada.

## Licença

O código está sob a licença MIT (arquivo `LICENSE`). O conjunto de dados
(CC BY-NC-ND 4.0) e os documentos da base de conhecimento pertencem aos seus
autores e não são redistribuídos aqui.

