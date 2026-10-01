"""Extrator compartilhado: o estado de cada commit vem somente do primeiro pai.

Merges históricos contam como um evento (diff contra o primeiro pai); merges não são
aceitos como alvo. Branches irmãs e descendentes não participam das entradas.
Não usa buggy, datas de correções futuras ou código executável dos repositórios.
"""
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from immutables import Map
from esquema import FEATURES

FORMATO = "%x00%x1e%H%x00%P%x00%ae%x00%ce%x00%at%x00%ct%x00%ai%x00%B%x00"
DIA = 86400
FIX = re.compile(r"\b(fix|fixes|fixed|bug|bugfix|npe|regression)\b", re.I)
PALAVRAS = {
    "p_fix": FIX, "p_refactor": re.compile(r"\b(refactor\w*|cleanup|rename\w*)\b", re.I),
    "p_test": re.compile(r"\b(test|tests|testing)\b", re.I),
    "p_doc": re.compile(r"\b(doc|docs|javadoc|typo|readme)\b", re.I),
    "p_revert": re.compile(r"\brevert\w*\b", re.I),
    "p_minor": re.compile(r"\b(minor|trivial|nit)\b", re.I),
    "p_pressa": re.compile(r"\b(wip|temp|temporary|hack|workaround|quick|hotfix)\b", re.I),
    "p_add": re.compile(r"\b(add|adds|added|adding|new|support|implement\w*)\b", re.I),
}


class ExtracaoIncompativel(ValueError):
    pass


def git(repo, argumentos, **kwargs):
    ambiente = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1"}
    return subprocess.run(["git", "-c", "core.quotePath=false", "-c", "diff.renames=true",
                           f"--git-dir={Path(repo)}", *argumentos], env=ambiente,
                          check=True, capture_output=True, **kwargs)


def eh_teste(path):
    nome = path.rsplit("/", 1)[-1]
    partes = path.lower().split("/")
    return (bool({"test", "tests", "testsrc", "testutils"} & set(partes[:-1]))
            or nome.lower().startswith("test") or nome.endswith(("Test.java", "Tests.java", "IT.java", "TestCase.java")))


def interpretar(raw):
    campos = raw.split(b"\0", 8)
    if len(campos) != 9:
        raise ExtracaoIncompativel("Registro Git incompleto ou mensagem com separador incompatível.")
    sha, pais, autor, committer, at, ct, data, mensagem = [p.decode("utf-8", "replace") for p in campos[:8]]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ExtracaoIncompativel("SHA inválido no histórico.")
    fuso = data[-5:]
    deslocamento = (1 if fuso[0] == "+" else -1) * (int(fuso[1:3])*3600 + int(fuso[3:5])*60)
    tokens = campos[8].split(b"\0")
    arquivos, submodulo, i = [], False, 0
    while i < len(tokens):
        token = tokens[i].lstrip(b"\n"); i += 1
        if not token:
            continue
        if token.startswith(b":"):
            # --raw informa modos, inclusive submódulos, sem ler o código do arquivo.
            partes = token.split()
            submodulo |= b"160000" in partes[:2]
            # O primeiro modo possui prefixo ':'; verificar sem ele também.
            submodulo |= partes[0] == b":160000"
            i += 2 if partes[-1].startswith((b"R", b"C")) else 1
            continue
        partes = token.split(b"\t", 2)
        if len(partes) != 3:
            raise ExtracaoIncompativel("Numstat inválido; extração interrompida.")
        a, d, caminho = partes
        if caminho:
            antigo = novo = caminho.decode("utf-8", "surrogateescape")
        else:
            if i + 1 >= len(tokens):
                raise ExtracaoIncompativel("Renomeação truncada.")
            antigo, novo = (t.decode("utf-8", "surrogateescape") for t in tokens[i:i+2]); i += 2
        binario = a == b"-" or d == b"-"
        arquivos.append((antigo, novo, 0 if binario else int(a), 0 if binario else int(d), binario))
    return {"sha": sha, "pais": pais.split(), "autor": autor.lower(), "committer": committer.lower(),
            "at": int(at), "ct": int(ct), "fuso": deslocamento, "mensagem": mensagem.strip(),
            "arquivos": arquivos, "submodulo": submodulo}


def ler_historico(repo, alvos):
    """União das cadeias de primeiro pai, em ordem topológica crescente."""
    if (Path(repo) / "shallow").exists():
        raise ExtracaoIncompativel("Histórico raso/incompleto. Obtenha o histórico completo para prever.")
    alvos = list(dict.fromkeys(alvos))
    if not alvos or any(not re.fullmatch(r"[0-9a-f]{40}", s) for s in alvos):
        raise ExtracaoIncompativel("Informe SHAs completos e válidos.")
    # stdin em arquivo evita bloqueio de pipes com listas grandes de SHAs.
    with tempfile.TemporaryFile() as entrada, tempfile.TemporaryFile() as saida:
        entrada.write(("\n".join(alvos) + "\n").encode()); entrada.seek(0)
        comando = ["git", "-c", "core.quotePath=false", "-c", "diff.renameLimit=100000",
                   f"--git-dir={repo}", "log", "--stdin", "--first-parent", "--reverse", "--topo-order",
                   "--raw", "--numstat", "-z", "-M50%", "--root", "--diff-merges=first-parent",
                   "--no-ext-diff", "--no-textconv", "--no-show-signature", f"--format={FORMATO}"]
        ambiente = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_NO_REPLACE_OBJECTS": "1"}
        processo = subprocess.run(comando, stdin=entrada, stdout=saida, stderr=subprocess.PIPE,
                                  env=ambiente, timeout=1800)
        if processo.returncode:
            raise ExtracaoIncompativel("Não foi possível ler todo o histórico: " + processo.stderr.decode("utf-8", "replace")[:400])
        saida.seek(0)
        resto = b""
        while bloco := saida.read(4*1024*1024):
            partes = (resto + bloco).split(b"\0\x1e")
            resto = partes.pop()
            for parte in partes:
                if parte.strip(b"\0\n"):
                    yield interpretar(parte)
        if resto.strip(b"\0\n"):
            yield interpretar(resto)


def estaticas(c):
    arq = c["arquivos"]
    java = [a for a in arq if a[1].endswith(".java")]
    testes = [a for a in arq if eh_teste(a[1])]
    prod = [a for a in java if not eh_teste(a[1])]
    la, ld = sum(a[2] for a in arq), sum(a[3] for a in arq)
    hora = ((c["at"] + c["fuso"]) % DIA)/3600
    dia = int(((c["at"]+c["fuso"])//DIA + 3) % 7)
    msg = c["mensagem"]
    return {"la": la, "ld": ld, "nf": len(arq),
        "q_hora": hora, "q_dia_semana": dia, "q_fim_semana": int(dia>=5),
        "q_madrugada": int(hora < 6 or hora >=22), "q_fuso_h": c["fuso"]/3600,
        "q_atraso_commit_h": max(0, (c["ct"]-c["at"])/3600),
        "o_n_java": len(java), "o_n_teste": len(testes), "o_n_producao": len(prod),
        "o_n_outros": len(arq)-len(java), "o_tem_teste": int(bool(testes)),
        "o_frac_la_teste": sum(a[2] for a in testes)/la if la else 0,
        "o_la_producao": sum(a[2] for a in prod), "o_ld_producao": sum(a[3] for a in prod),
        "o_n_extensoes": len({Path(a[1]).suffix.lower() for a in arq}),
        "o_n_pastas": len({a[1].rsplit("/",1)[0] if "/" in a[1] else "" for a in arq}),
        "o_n_binarios": sum(a[4] for a in arq), "o_n_renomeados": sum(a[0]!=a[1] for a in arq),
        "o_max_la_arquivo": max((a[2] for a in arq),default=0), "o_frac_remocao": ld/(la+ld) if la+ld else 0,
        "m_caracteres": len(msg), "m_palavras": len(msg.split()), "m_linhas": msg.count("\n")+1,
        "m_assunto_caracteres": len(msg.split("\n",1)[0]),
        "w_autor_diferente_committer": int(c["autor"]!=c["committer"])}


def media(itens, padrao=0):
    return sum(itens)/len(itens) if itens else padrao


def pistas(c, estado):
    # Relógio monotônico ancestral: trata relógios Git que recuam sem usar descendentes.
    agora = max(c["ct"], estado.get("relogio", c["ct"]))
    autor = c["autor"]
    a = estado.get(("autor", autor), (0, agora, agora, (), 0))
    antigos = [x[0] for x in c["arquivos"]]
    conhecidos = [estado[("arquivo", p)] for p in antigos if ("arquivo", p) in estado]
    todos = [estado.get(("arquivo", p), (0, agora, agora, Map(), 0)) for p in antigos]
    pastas = {p.rsplit("/",1)[0] if "/" in p else "" for p in [x[1] for x in c["arquivos"]]}
    n = len(antigos)
    h = {"w_autor_commits": a[0], "w_autor_dias_desde_primeiro": (agora-a[1])/DIA if a[0] else -1,
         "w_autor_dias_desde_ultimo": (agora-a[2])/DIA if a[0] else -1,
         "w_autor_commits_7d": sum(t >= agora-7*DIA for t in a[3]),
         "w_autor_nos_arquivos": sum(f[3].get(autor,0) for f in todos),
         "w_autor_frac_arquivos_conhecidos": sum(autor in f[3] for f in todos)/n if n else 0,
         "w_autor_nas_pastas": sum(estado.get(("pasta",autor,p),0) for p in pastas),
         "w_autor_frac_correcoes": a[4]/a[0] if a[0] else 0,
         "l_n_arquivos_novos": n-len(conhecidos), "l_frac_arquivos_novos": 1-len(conhecidos)/n if n else 0,
         "l_arq_mudancas_max": max((f[0] for f in conhecidos),default=0),
         "l_arq_mudancas_media": media([f[0] for f in conhecidos]),
         "l_arq_idade_media_d": media([(agora-f[1])/DIA for f in conhecidos],-1),
         "l_arq_dias_desde_ultima_min": min(((agora-f[2])/DIA for f in conhecidos),default=-1),
         "l_arq_dias_desde_ultima_media": media([(agora-f[2])/DIA for f in conhecidos],-1),
         "l_arq_autores_max": max((len(f[3]) for f in conhecidos),default=0),
         "l_arq_autores_media": media([len(f[3]) for f in conhecidos]),
         "l_arq_correcoes_max": max((f[4] for f in conhecidos),default=0),
         "l_arq_correcoes_media": media([f[4] for f in conhecidos]),
         "l_arq_frac_correcoes": media([f[4]/f[0] for f in conhecidos])}
    h.update(estaticas(c))
    return {f: h[f] for f in FEATURES}


def avancar(c, estado):
    agora = max(c["ct"], estado.get("relogio", c["ct"]))
    autor = c["autor"]
    a = estado.get(("autor",autor),(0,agora,agora,(),0))
    fix = int(bool(FIX.search(c["mensagem"])))
    with estado.mutate() as novo:
        novo["relogio"] = agora
        novo[("autor",autor)] = (a[0]+1, a[1], agora, tuple(t for t in a[3] if t>=agora-7*DIA)+(agora,),a[4]+fix)
        for antes, depois, *_ in c["arquivos"]:
            f = estado.get(("arquivo",antes),(0,agora,agora,Map(),0))
            novo[("arquivo",depois)] = (f[0]+1,f[1],agora,f[3].set(autor,f[3].get(autor,0)+1),f[4]+fix)
            if antes!=depois and ("arquivo",antes) in novo:
                del novo[("arquivo",antes)]
        for pasta in {x[1].rsplit("/",1)[0] if "/" in x[1] else "" for x in c["arquivos"]}:
            chave=("pasta",autor,pasta); novo[chave]=estado.get(chave,0)+1
        return novo.finish()


def processar_commits(commits, alvos):
    """Mapas persistentes compartilham memória, mas nunca misturam branches."""
    alvos = set(alvos)
    filhos = Counter(c["pais"][0] for c in commits if c["pais"])
    estados, linhas = {}, []
    for c in commits:
        pai = c["pais"][0] if c["pais"] else None
        if pai and pai not in estados:
            raise ExtracaoIncompativel("Pai ausente do histórico; não é permitido preencher histórico com zero.")
        estado = estados[pai] if pai else Map()
        if c["sha"] in alvos:
            motivos = []
            if len(c["pais"]) != 1: motivos.append("merge_ou_raiz")
            if not c["arquivos"]: motivos.append("sem_arquivos")
            if any(a[4] for a in c["arquivos"]): motivos.append("binario")
            if c["submodulo"]: motivos.append("submodulo")
            registro = {"commit_id":c["sha"],"author_date_git":c["at"],"committer_date_git":c["ct"],
                        "motivo": ";".join(motivos), **pistas(c,estado)}
            registro.update({nome: int(bool(p.search(c["mensagem"]))) for nome,p in PALAVRAS.items()})
            linhas.append(registro)
        if filhos[c["sha"]]:
            estados[c["sha"]] = avancar(c,estado)
        if pai:
            filhos[pai]-=1
            if not filhos[pai]: del estados[pai]
    if {r["commit_id"] for r in linhas} != alvos:
        raise ExtracaoIncompativel("Nem todos os commits solicitados foram encontrados.")
    return linhas


def extrair(repo, alvos):
    return processar_commits(list(ler_historico(repo,alvos)),alvos)


def extrair_um(repo, sha):
    linha = extrair(repo,[sha])[0]
    if linha["motivo"]:
        raise ExtracaoIncompativel("Commit incompatível: " + linha["motivo"])
    return {f:linha[f] for f in FEATURES}
