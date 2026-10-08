"""Extrai oito pistas adicionais do diff dos mesmos commits elegíveis.

Usa CP5_REPOS_DIR para os clones já disponíveis. Cache incluído na entrega.
"""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from dados import carregar_dividir, hash_extrator, pasta_repos, salvar_json
from esquema import ROOT
from extracao import FORMATO, interpretar, estaticas, ExtracaoIncompativel
from novas_pistas import dispersao_commit, ESTATICAS

CACHE = ROOT / 'artefatos/pistas_dispersao.parquet'
MANIFESTO = ROOT / 'resultados/extracao_dispersao.json'


def assinatura_codigo():
    return hashlib.sha256((ROOT / 'novas_pistas.py').read_bytes() + Path(__file__).read_bytes()).hexdigest()


def ler_alvos(repo, shas):
    """Lê apenas os alvos, sem repetir a extração de todo o histórico ancestral."""
    if not shas or any(len(s) != 40 or any(c not in '0123456789abcdef' for c in s) for s in shas):
        raise ValueError('Esperados SHAs Git completos.')
    with tempfile.TemporaryFile() as entrada, tempfile.TemporaryFile() as saida:
        entrada.write(('\n'.join(shas) + '\n').encode()); entrada.seek(0)
        comando = ['git', '-c', 'core.quotePath=false', '-c', 'diff.renameLimit=100000',
                   f'--git-dir={repo}', 'log', '--stdin', '--no-walk=unsorted', '--raw', '--numstat',
                   '-z', '-M50%', '--root', '--diff-merges=first-parent', '--no-ext-diff',
                   '--no-textconv', '--no-show-signature', f'--format={FORMATO}']
        p = subprocess.run(comando, stdin=entrada, stdout=saida, stderr=subprocess.PIPE,
                           env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1'}, timeout=1800)
        if p.returncode:
            raise ExtracaoIncompativel(p.stderr.decode('utf-8', 'replace')[:400])
        saida.seek(0); resto = b''
        while bloco := saida.read(4 * 1024 * 1024):
            partes = (resto + bloco).split(b'\0\x1e'); resto = partes.pop()
            for parte in partes:
                if parte.strip(b'\0\n'):
                    yield interpretar(parte)
        if resto.strip(b'\0\n'):
            yield interpretar(resto)


def carregar_pistas(frame):
    meta = json.loads(MANIFESTO.read_text(encoding='utf-8'))
    if meta['codigo_sha256'] != assinatura_codigo() or meta['extrator_original_sha256'] != hash_extrator():
        raise ValueError('O código das novas pistas mudou. Refaça a extração em outra pasta.')
    if meta['cache_sha256'] != hashlib.sha256(CACHE.read_bytes()).hexdigest():
        raise ValueError('Cache das novas pistas diferente do manifesto.')
    pistas = pd.read_parquet(CACHE).set_index('commit_id')
    if pistas.index.duplicated().any() or not frame.commit_id.isin(pistas.index).all():
        raise ValueError('Cobertura incompleta ou duplicada das novas pistas.')
    saida = frame.copy()
    saida[list(ESTATICAS)] = pistas.loc[frame.commit_id, list(ESTATICAS)].to_numpy()
    if not np.isfinite(saida[list(ESTATICAS)].to_numpy()).all():
        raise ValueError('Novas pistas não finitas.')
    return saida


def executar():
    treino, teste, _, info = carregar_dividir()
    elegiveis = pd.concat([treino, teste], ignore_index=True)
    if CACHE.exists() and MANIFESTO.exists():
        carregar_pistas(elegiveis)
        print('Cache de dispersão conferido.', flush=True)
        return
    # O roteamento vem da extração original, inclusive o repositório compartilhado Hadoop.
    base = pd.read_parquet(ROOT / 'artefatos/pistas_completas.parquet')
    base = base[base.commit_id.isin(elegiveis.commit_id)].set_index('commit_id')
    registros = []
    for repo, grupo in base.groupby('repositorio_git', sort=True):
        pasta = pasta_repos() / (repo.split('/')[-1] + '.git')
        vistos = set()
        for c in ler_alvos(pasta, grupo.index.tolist()):
            sha = c['sha']
            if sha not in grupo.index or sha in vistos:
                raise ValueError('Git retornou alvo inesperado ou repetido.')
            vistos.add(sha)
            antigo = estaticas(c)
            for nome, valor in antigo.items():
                if not np.isclose(valor, grupo.loc[sha, nome], rtol=0, atol=1e-10):
                    raise ValueError(f'Paridade da extração original divergente: {sha} / {nome}')
            registros.append({'commit_id': sha, **dispersao_commit(c)})
        if vistos != set(grupo.index):
            raise ValueError(f'Extração incompleta: {repo}')
        print(f'{repo}: {len(vistos)} commits; métricas originais conferidas.', flush=True)
    tabela = pd.DataFrame(registros).sort_values('commit_id').reset_index(drop=True)
    assert len(tabela) == len(elegiveis) and not tabela.commit_id.duplicated().any()
    tabela.to_parquet(CACHE, index=False)
    salvar_json(MANIFESTO, {
        'codigo_sha256': assinatura_codigo(), 'extrator_original_sha256': hash_extrator(),
        'cache_sha256': hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        'csv_sha256': info['csv_sha256'], 'colunas': list(ESTATICAS), 'linhas': len(tabela),
        'paridade_estaticas_originais': True, 'usa_rotulos': False,
        'definicao': 'Diff textual contra o primeiro pai; entropia de adições + remoções por arquivo.'})
    print(f'Extração concluída: {len(tabela)} commits, oito pistas novas.', flush=True)


if __name__ == '__main__':
    executar()
