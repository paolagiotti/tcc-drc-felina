"""
Conferencia do ambiente e da pasta do projeto, antes de qualquer execucao.

Responde, sem depender de memoria, a quatro perguntas:

  1. Cada arquivo do projeto e o da versao registrada no MANIFESTO.txt?
     Compara a impressao digital (SHA-256) de cada arquivo com a do manifesto.
     Arquivo diferente do manifesto foi editado depois dele.
  2. Ha arquivos sobrando ou faltando na pasta?
  3. O arquivo de origem e o mesmo usado na execucao final (SHA-256)?
  4. As bibliotecas estao instaladas nas versoes fixadas em requirements.txt?

Uso:
    python verificar.py
    python verificar.py --atualizar-manifesto
        registra o estado atual dos arquivos como nova referencia. Use so depois
        de uma edicao intencional (por exemplo, os custos reais dos exames em
        config.py), para que ela deixe de aparecer como divergencia.
    python verificar.py --api
        acrescenta um teste real da API da OpenAI com o modelo configurado e
        os mesmos parametros da camada interpretativa. Custa uma fracao de
        centavo; nao rodar sem necessidade.

Os scripts de execucao (rodada_final.sh e rodada_camada_v2.sh) chamam esta
conferencia e param se ela apontar problema.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
problemas = 0


def ok(m): print(f"  ok   {m}")
def aviso(m): print(f"  !    {m}")
def erro(m, c=""):
    global problemas
    problemas += 1
    print(f"  ERRO {m}")
    if c:
        print(f"       -> {c}")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


if "--atualizar-manifesto" in sys.argv:
    man = AQUI / "MANIFESTO.txt"
    if not man.exists():
        sys.exit("MANIFESTO.txt nao encontrado: nada a atualizar.")
    linhas = man.read_text(encoding="utf-8").splitlines()
    nomes = [l.split(maxsplit=1)[1].strip() for l in linhas
             if l.strip() and not l.startswith("#")]
    from datetime import date
    novo = [f"# manifesto atualizado em {date.today():%d/%m/%Y}",
            "# Impressao digital (SHA-256) de cada arquivo do projeto. Conferir com:",
            "#   python verificar.py"]
    for n in nomes:
        p = AQUI / n
        if p.exists():
            novo.append(f"{sha(p)}  {n}")
    man.write_text("\n".join(novo) + "\n", encoding="utf-8")
    print(f"MANIFESTO.txt atualizado com {len(novo) - 3} arquivos.")
    sys.exit(0)

# ---------------------------------------------------------------------------
print("=== 1. arquivos do projeto contra o MANIFESTO.txt ===")
man_arq = AQUI / "MANIFESTO.txt"
manifesto = {}
if not man_arq.exists():
    erro("MANIFESTO.txt nao encontrado", "ele acompanha os scripts no repositorio")
else:
    texto_man = man_arq.read_text(encoding="utf-8").splitlines()
    for linha in texto_man:
        if linha.strip() and not linha.startswith("#"):
            h, nome = linha.split(maxsplit=1)
            manifesto[nome.strip()] = h
    data = next((l for l in texto_man if l.startswith("# manifesto")), "# manifesto ?")
    print(f"  ({data[2:]}, {len(manifesto)} arquivos)")

desatualizados, faltando = [], []
for nome, h in sorted(manifesto.items()):
    p = AQUI / nome
    if not p.exists():
        faltando.append(nome)
    elif sha(p) != h:
        desatualizados.append(nome)
for n in faltando:
    erro(f"FALTANDO: {n}", "restaure o arquivo a partir do repositorio")
for n in desatualizados:
    if n.endswith(".csv"):
        aviso(f"DIFERE do manifesto: {n}; pode ser edicao sua, confira antes de substituir")
    else:
        erro(f"DIFERE do manifesto: {n}",
             "restaure a versao do repositorio, ou registre a edicao com --atualizar-manifesto")
if manifesto and not (faltando or desatualizados):
    ok("todos os arquivos batem com o manifesto")

# ---------------------------------------------------------------------------
print("\n=== 2. arquivos na pasta que nao estao no manifesto ===")
DADOS_ORIGEM = {
    "Survival_data.csv": "dado de origem; USADO pelo codigo",
    "risk_factor_data.csv": "dado de origem; NAO usado por nenhum script",
    "Prevalence_study_data.csv": "dado de origem; NAO usado por nenhum script",
}
# __pycache__ e recriado pelo proprio Python sempre que um script importa outro
# (inclusive por este verificador, com --api); nao e sobra da pasta.
IGNORAR = {"MANIFESTO.txt", "README.md", "LICENSE", "pytest.ini", ".DS_Store", ".env",
           "gitignore.txt", "__pycache__", ".pytest_cache"}
PASTAS = {"saida", "entregas", "base_conhecimento", "dados", "docs", "tests",
          ".git", ".github", ".venv", "venv"}
sobras = False
for p in sorted(AQUI.iterdir()):
    if p.name in IGNORAR or p.name in manifesto:
        continue
    if p.is_dir():
        if p.name in PASTAS:
            continue
        aviso(f"pasta nao reconhecida: {p.name}/")
    elif p.name in DADOS_ORIGEM:
        print(f"  --   {p.name}: {DADOS_ORIGEM[p.name]}")
        continue
    elif p.name == "dicionario_variaveis.csv":
        aviso("dicionario_variaveis.csv na RAIZ: o codigo usa dados/. Ver secao 4.")
    elif p.suffix == ".zip":
        aviso(f"descartavel: {p.name}")
    else:
        aviso(f"nao reconhecido: {p.name}; nao faz parte do projeto")
    sobras = True
if not sobras:
    ok("nenhum arquivo sobrando")
if (AQUI / "shap.py").exists():
    erro("shap.py na pasta: esconde a biblioteca SHAP", "apague o arquivo")

# ---------------------------------------------------------------------------
print("\n=== 3. dado de origem ===")
SHA_SURVIVAL = "54605ed7360991257a9371397dcda3070972265038ccabc56e2d5d8cbc89cc0f"
sv = AQUI / "Survival_data.csv"
if not sv.exists():
    erro("Survival_data.csv nao encontrado", "coloque o arquivo na pasta do projeto")
elif sha(sv) != SHA_SURVIVAL:
    erro("Survival_data.csv difere do arquivo usado na execucao final",
         "os resultados do trabalho nao seriam reproduziveis com este arquivo")
else:
    ok("Survival_data.csv identico ao da execucao final (SHA-256 confere)")

# ---------------------------------------------------------------------------
print("\n=== 4. dicionario de variaveis ===")
d_dados = AQUI / "dados" / "dicionario_variaveis.csv"
d_raiz = AQUI / "dicionario_variaveis.csv"
if not d_dados.exists():
    erro("dados/dicionario_variaveis.csv nao encontrado", "o codigo le desta pasta")
else:
    ok("dados/dicionario_variaveis.csv presente (e o que o codigo usa)")
    if d_raiz.exists():
        try:
            import pandas as pd
            a, b = pd.read_csv(d_raiz), pd.read_csv(d_dados)
            k = a.columns[0]
            m = a.merge(b, on=k, suffixes=("_raiz", "_dados"))
            col = "papel" if "papel" in a.columns else a.columns[1]
            dif = m[m[f"{col}_raiz"].astype(str) != m[f"{col}_dados"].astype(str)]
            so_raiz = set(a[k]) - set(b[k])
            so_dados = set(b[k]) - set(a[k])
            if dif.empty and not so_raiz and not so_dados:
                aviso("copias sem diferenca na coluna 'papel': diferenca inofensiva "
                      "para os resultados. A da raiz pode ser apagada.")
            else:
                erro(f"copias diferem em 'papel' em {len(dif)} variavel(is) "
                     f"(e {len(so_raiz) + len(so_dados)} so numa delas); isso muda "
                     f"os resultados", "decida qual versao vale e apague a outra")
                for _, r in dif.iterrows():
                    print(f"         {str(r[k])[:40]:<40} raiz={str(r[f'{col}_raiz']):<18} "
                          f"dados={r[f'{col}_dados']}")
                for v in sorted(so_raiz):
                    print(f"         so na raiz : {v}")
                for v in sorted(so_dados):
                    print(f"         so em dados: {v}")
        except Exception as e:  # noqa: BLE001
            aviso(f"nao foi possivel comparar as copias: {e}")

# ---------------------------------------------------------------------------
print("\n=== 5. bibliotecas ===")
from importlib.metadata import PackageNotFoundError, version  # noqa: E402

req = AQUI / "requirements.txt"
opcional = False
if req.exists():
    for linha in req.read_text(encoding="utf-8").splitlines():
        if "camada interpretativa" in linha.lower():
            opcional = True
        linha = linha.split("#")[0].strip()
        if "==" not in linha:
            continue
        pac, esp = linha.split("==")
        try:
            inst = version(pac)
        except PackageNotFoundError:
            inst = None
        desc = f"esperada {esp}, " + (f"instalada {inst}" if inst else "NAO instalada")
        if inst == esp:
            ok(f"{pac} {inst}")
        elif opcional:
            aviso(f"{pac}: {desc} (so necessaria para a camada interpretativa)")
        else:
            erro(f"{pac}: {desc}", f"{sys.executable} -m pip install -r requirements.txt")

# ---------------------------------------------------------------------------
print("\n=== 6. camada interpretativa ===")
base = AQUI / "base_conhecimento"
pdfs = list(base.glob("*.pdf")) if base.exists() else []
(ok if pdfs else aviso)(f"base_conhecimento/: {len(pdfs)} PDF(s)")
tem_chave = bool(os.environ.get("OPENAI_API_KEY"))
(ok if tem_chave else aviso)("OPENAI_API_KEY " + ("definida" if tem_chave else
                             "NAO definida (so necessaria para gerar os textos)"))

# ---------------------------------------------------------------------------
if "--api" in sys.argv:
    print("\n=== 7. API da OpenAI (chamada real, custo de fracao de centavo) ===")
    import re
    try:
        sys.path.insert(0, str(AQUI))
        import config as cfg
        from camada_interpretativa import testar_acesso
    except Exception as e:  # noqa: BLE001
        erro(f"nao foi possivel carregar a camada: {e}",
             "resolva antes os problemas das secoes anteriores")
    else:
        modelo = cfg.MODELO_LLM
        print(f"  modelo configurado: {modelo} | versao declarada: "
              f"{getattr(cfg, 'VERSAO_LLM', modelo)}")
        passou, msg = testar_acesso()
        if passou:
            ok(msg)
        else:
            erro(msg)
            # diagnostico: o nome curto funciona? entao falta so liberar a versao
            curto = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", modelo)
            if curto != modelo and os.environ.get("OPENAI_API_KEY"):
                p2, m2 = testar_acesso(modelo=curto)
                if p2:
                    aviso(f"o nome curto '{curto}' funciona: a conta tem acesso. "
                          f"Falta so liberar '{modelo}' nos modelos permitidos do projeto.")

print()
if problemas:
    print(f"{problemas} problema(s) a resolver antes de rodar.")
    sys.exit(1)
print("Tudo certo.")
