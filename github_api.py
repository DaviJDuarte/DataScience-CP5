"""Consulta apenas metadados públicos; não clona nem executa repositórios."""
import os
import re
from urllib.parse import urlparse
import requests


class ErroGitHub(ValueError):
    pass


class ExtracaoIncompativel(ErroGitHub):
    """O commit foi obtido, mas não segue a convenção de contagem verificada no treino."""


def repositorio(url):
    parsed = urlparse(url.strip())
    partes = parsed.path.strip("/").removesuffix(".git").split("/")
    if (parsed.scheme != "https" or parsed.netloc.lower() != "github.com"
            or parsed.query or parsed.fragment or len(partes) != 2
            or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in partes)
            or any(p in {".", ".."} for p in partes)):
        raise ErroGitHub("Use uma URL como https://github.com/apache/commons-lang.")
    return "/".join(partes)


def consultar(caminho, token=None, params=None):
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    token = token or os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(f"https://api.github.com/repos/{caminho}", headers=headers,
                            params=params, timeout=30)
    except requests.RequestException as exc:
        raise ErroGitHub("Falha de conexão com o GitHub. Use a demonstração local.") from exc
    if resp.status_code in {403, 429}:
        raise ErroGitHub("Limite da API ou acesso negado. Aguarde ou configure GITHUB_TOKEN.")
    if resp.status_code == 401:
        raise ErroGitHub("Token inválido. Corrija-o ou consulte sem token.")
    if resp.status_code in {404, 422}:
        raise ErroGitHub("Repositório ou commit indisponível. Confira a URL e a referência.")
    if resp.status_code == 409:
        raise ErroGitHub("Repositório vazio ou sem histórico disponível.")
    if not resp.ok:
        raise ErroGitHub(f"GitHub indisponível (HTTP {resp.status_code}). Use o modo local.")
    try:
        return resp.json(), "next" in resp.links
    except ValueError as exc:
        raise ErroGitHub("Resposta inválida do GitHub.") from exc


def listar_commits(repo, pagina=1, token=None):
    metadados, _ = consultar(repo, token)
    if metadados.get("private", True):
        raise ErroGitHub("Esta demonstração aceita somente repositórios públicos.")
    # Sem parâmetro sha, a API usa a branch padrão do repositório.
    dados, proxima = consultar(f"{repo}/commits", token, {"per_page": 30, "page": pagina})
    if not isinstance(dados, list) or not dados:
        raise ErroGitHub("Nenhum commit nesta página ou repositório vazio.")
    return dados, proxima


def baixar_commit(repo, sha, token=None):
    """Commit e lista completa de arquivos, percorrendo a paginação."""
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha):
        raise ErroGitHub("SHA inválido.")
    arquivos, primeiro = [], None
    for pagina in range(1, 31):
        dados, proxima = consultar(f"{repo}/commits/{sha}", token,
                                   {"per_page": 100, "page": pagina})
        if primeiro is None:
            primeiro = dados
            if len(dados.get("parents", [])) != 1:
                raise ExtracaoIncompativel("Merges e commits iniciais não são compatíveis nesta versão.")
        if dados.get("sha") != primeiro.get("sha"):
            raise ExtracaoIncompativel("Paginação retornou commits diferentes; previsão recusada.")
        arquivos.extend(dados.get("files", []))
        if not proxima:
            break
    if proxima or len(arquivos) >= 3000:
        raise ExtracaoIncompativel("Limite de arquivos da API atingido; lista possivelmente incompleta.")
    return primeiro, arquivos


def medir(commit, arquivos):
    """la, ld e nf na convenção conferida com o treino; recusa o que não foi verificado."""
    if not arquivos or len({f["filename"] for f in arquivos}) != len(arquivos):
        raise ExtracaoIncompativel("Lista de arquivos vazia ou inconsistente.")
    for f in arquivos:
        # Arquivo apenas movido: conta em nf e não tem linhas nem patch.
        if (f.get("status") == "renamed" and "patch" not in f
                and f.get("additions") == 0 and f.get("deletions") == 0):
            continue
        if (f.get("status") not in {"added", "removed", "modified", "renamed"}
                or "patch" not in f or "Subproject commit" in f.get("patch", "")):
            raise ExtracaoIncompativel("Cópias, binários, submódulos ou diffs ausentes não são compatíveis nesta versão.")
        # Um patch truncado não oferece evidência suficiente de compatibilidade.
        linhas = f["patch"].splitlines()
        adicionadas = sum(s.startswith("+") for s in linhas)
        removidas = sum(s.startswith("-") for s in linhas)
        if adicionadas != f.get("additions") or removidas != f.get("deletions"):
            raise ExtracaoIncompativel("Diff incompleto ou contagens inconsistentes; previsão recusada.")
    la = sum(f["additions"] for f in arquivos)
    ld = sum(f["deletions"] for f in arquivos)
    stats = commit.get("stats", {})
    if la != stats.get("additions") or ld != stats.get("deletions"):
        raise ExtracaoIncompativel("A soma dos arquivos diverge do total; previsão recusada.")
    return {"la": la, "ld": ld, "nf": len(arquivos)}


def extrair_commit(repo, sha, token=None):
    commit, arquivos = baixar_commit(repo, sha, token)
    return medir(commit, arquivos), commit["html_url"]
