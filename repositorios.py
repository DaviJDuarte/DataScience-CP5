"""Histórico público em cache. Apenas Git, sem checkout, hooks ou execução de código."""
import os
import subprocess
from pathlib import Path
from dados import pasta_repos
from esquema import ROOT, FEATURES
from extracao import git, extrair_um, ExtracaoIncompativel
from github_api import consultar, baixar_commit, medir, repositorio, ErroGitHub
from novas_pistas import dispersao_commit
from extrair_dispersao import ler_alvos


def caminho_repo(repo):
    # Validar também chamadas diretas desta função, além da interface.
    repo=repositorio("https://github.com/"+repo)
    owner,nome=repo.split("/")
    local=pasta_repos()/f"{nome}.git"
    if owner.lower()=="apache" and local.exists(): return local
    return pasta_repos()/f"cache_{owner}__{nome}.git"


def obter_repo(repo, baixar=False):
    path=caminho_repo(repo)
    if path.exists():
        try:
            resposta=git(path,["rev-parse","--is-bare-repository"],timeout=15)
            if resposta.stdout.strip()!=b'true':
                raise ExtracaoIncompativel("O cache precisa ser um repositório Git bare.")
        except (OSError,subprocess.SubprocessError) as exc:
            raise ExtracaoIncompativel("Cache Git inválido ou incompleto: "+str(path)) from exc
        return path
    if not baixar:
        raise ExtracaoIncompativel("O histórico deste repositório ainda não está em cache. Use Baixar histórico público.")
    path.parent.mkdir(parents=True,exist_ok=True)
    comando=["git","-c","protocol.file.allow=never","-c","protocol.ext.allow=never",
             "clone","--bare","--no-recurse-submodules",f"https://github.com/{repo}.git",str(path)]
    try:
        subprocess.run(comando,check=True,capture_output=True,timeout=1800,
                       env={**os.environ,"GIT_TERMINAL_PROMPT":"0"})
    except (OSError,subprocess.SubprocessError) as exc:
        raise ExtracaoIncompativel("Não foi possível baixar o histórico completo. Verifique rede, espaço em disco e Git instalado.") from exc
    return path


def metricas_github(repo,sha,token=None,dispersao=False):
    meta,_=consultar(repo,token)
    if meta.get("private",True): raise ErroGitHub("Somente repositórios públicos são aceitos.")
    commit,arquivos=baixar_commit(repo,sha,token)
    esperadas=medir(commit,arquivos)
    path=obter_repo(repo)
    sha=commit["sha"]
    try:
        git(path,["cat-file","-e",sha+"^{commit}"],timeout=15)
    except subprocess.CalledProcessError:
        # Atualiza só quando o SHA selecionado ainda não está disponível; nenhum push.
        try:
            git(path,["fetch","--no-tags",f"https://github.com/{repo}.git",sha],timeout=1200)
        except subprocess.SubprocessError as exc:
            raise ExtracaoIncompativel("Não foi possível obter o commit e seus ancestrais completos.") from exc
    metricas=extrair_um(path,sha)
    if any(metricas[f]!=esperadas[f] for f in ["la","ld","nf"]):
        raise ExtracaoIncompativel("Contagens de Git e GitHub divergentes. Previsão recusada.")
    if dispersao:
        alvos = list(ler_alvos(path, [sha]))
        if len(alvos) != 1 or alvos[0]['sha'] != sha:
            raise ExtracaoIncompativel("Diff do alvo indisponível para calcular dispersão.")
        metricas.update(dispersao_commit(alvos[0]))
    return metricas,commit["html_url"]
