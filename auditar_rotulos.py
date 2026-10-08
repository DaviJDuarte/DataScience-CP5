"""Auditoria de proveniência ApacheJIT: reproduz regras, não inventa novos rótulos.

python auditar_rotulos.py --pacote CAMINHO --repos CAMINHO
O pacote oficial já baixado e os clones Git são lidos, nunca executados ou alterados.
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from dados import CSV_SHA256, carregar_dividir, pasta_repos, presentes, salvar_json
from esquema import ROOT
from extracao import git

PASTA = ROOT / 'resultados/auditoria_rotulos'
MD5_PACOTE = '528bf0ee04b15976be6bf15f8efddc65'
HADOOP = {'apache/hadoop', 'apache/hadoop-hdfs', 'apache/hadoop-mapreduce'}
FONTES = {
    'pacote': 'https://zenodo.org/records/5907847',
    'artigo': 'https://arxiv.org/html/2203.00101',
    'coleta': 'https://github.com/hosseinkshvrz/apachejit/blob/master/src/gitminer.py',
    'ligacao': 'https://github.com/hosseinkshvrz/apachejit/blob/master/notebooks/data_linking_filtering.ipynb',
    'construcao': 'https://github.com/hosseinkshvrz/apachejit/blob/master/notebooks/dataset_construction.ipynb',
    'issue_upstream': 'https://github.com/hosseinkshvrz/apachejit/issues/1',
}


def familia(projetos):
    """Agrupamento de relatório; não tenta adivinhar o componente de cada commit."""
    return projetos.where(~projetos.isin(HADOOP), 'apache/hadoop (familia)')


def conferir_chave(issue_key, mensagem):
    """Variações de pontuação são válidas; #123 sozinho não confirma PROJETO-123."""
    prefixo, numero = issue_key.rsplit('-', 1)
    inicio, fim = r'(?<![A-Za-z0-9])', r'(?![A-Za-z0-9])'
    literal = bool(re.search(inicio + re.escape(issue_key) + fim, mensagem, re.I))
    flexivel = bool(re.search(inicio + re.escape(prefixo) + r'[- _:=]+' + numero + fim, mensagem, re.I))
    outras = sorted(set(re.findall(inicio + re.escape(prefixo) + r'-[0-9]+' + fim, mensagem, re.I)) - {issue_key})
    numero_hash = bool(re.search('#' + numero + r'(?![0-9])', mensagem))
    return {'chave_literal_na_mensagem': literal, 'chave_flexivel_na_mensagem': flexivel,
            'outras_chaves_do_projeto': '; '.join(outras),
            'numero_em_referencia_hash': numero_hash,
            'possivel_colisao_issue_pr': not flexivel and bool(outras) and numero_hash}


def ler_fonte(pacote):
    if hashlib.md5(pacote.read_bytes()).hexdigest() != MD5_PACOTE:
        raise ValueError('Pacote diferente do ApacheJIT v2 verificado.')
    with zipfile.ZipFile(pacote) as z:
        def ler(nome): return pd.read_csv(z.open('apachejit/data/' + nome))
        raw = z.read('apachejit/dataset/apachejit_total.csv')
        if hashlib.sha256(raw).hexdigest() != CSV_SHA256:
            raise ValueError('CSV do pacote diferente da base utilizada no projeto.')
        total = pd.read_csv(z.open('apachejit/dataset/apachejit_total.csv'))
        issues = pd.concat([ler(f'{ano}.csv')[['Issue key', 'Created', 'Resolution', 'Issue Type', 'Status']]
                            for ano in range(2010, 2020)], ignore_index=True)
        issues['project'] = issues['Issue key'].str.split('-').str[0]
        issues = issues[~issues.project.isin(['MESOS', 'ROCKETMQ'])].copy()
        projetos = sorted(issues.project.unique())
        found = pd.concat([ler(p + '.csv').assign(project=p) for p in projetos], ignore_index=True)
        links = pd.concat([ler('commit_links_' + p + '.csv') for p in projetos], ignore_index=True)
        chaves = set(ler('keys_apachejava_ast.csv').commit_id)
        limpos = ler('clean_filtered.csv').merge(ler('keys_clean_ast.csv'), on='commit_id', validate='one_to_one')
        arquivos = ['apachejit/notebooks/data_linking_filtering.ipynb',
                    'apachejit/notebooks/dataset_construction.ipynb', 'apachejit/src/gitminer.py',
                    'apachejit/src/collector.py', 'apachejit/src/gumtree.py']
        hashes = {a: hashlib.sha256(z.read(a)).hexdigest() for a in arquivos}
    return total, issues, found, links, chaves, limpos, hashes


def reconstruir(issues, found, links, chaves):
    """Reimplementação das células 16–44; pandas não executa o notebook da fonte.

    A fonte compara Created sem fuso com epochs Git convertidos sem fuso. Reproduzimos
    essa convenção somente para conferir os rótulos; o fuso do export Jira é desconhecido.
    Os cortes do nosso treino são comparados depois com fix_date Git em UTC.
    """
    relatorios = found.merge(issues[['Issue key', 'Created', 'Resolution', 'Status']],
                            left_on='issue_key', right_on='Issue key', validate='many_to_one')
    relatorios['Created'] = pd.to_datetime(relatorios.Created, format='%d/%b/%y %H:%M')
    # Preserva a ordem da fonte ao escolher a primeira associação para cada par.
    ligados = links.merge(relatorios[['commit_id', 'issue_key', 'Created', 'Resolution', 'Status']],
                           left_on='fix_hash', right_on='commit_id', how='inner')
    ligados = ligados.drop_duplicates(['fix_hash', 'bug_hash']).copy()
    for coluna in ['bug_date', 'fix_date']:
        ligados[coluna] = pd.to_datetime(ligados[coluna], unit='s')
    temporal = ligados[(ligados.bug_date < ligados.Created) & (ligados.bug_date < ligados.fix_date)].copy()
    fixcount = temporal.groupby('fix_hash').size()
    max_fix = int(fixcount.mean() + fixcount.std(ddof=1))
    apos_fix = temporal[temporal.fix_hash.isin(fixcount[fixcount <= max_fix].index)].copy()
    bugcount = apos_fix.groupby('bug_hash').size()
    max_bug = int(bugcount.mean() + bugcount.std(ddof=1))
    apos_bug = apos_fix[apos_fix.bug_hash.isin(bugcount[bugcount <= max_bug].index)].copy()
    aceitos = apos_bug[apos_bug.bug_hash.isin(chaves)].copy()
    # A evidência completa por issue evita perder uma referência Fixed quando duas
    # issues estão associadas ao mesmo fixing commit. O conjunto de pares não muda.
    evidencias = aceitos.drop(columns=['commit_id', 'issue_key', 'Created', 'Resolution', 'Status']).merge(
        relatorios[['commit_id', 'issue_key', 'Created', 'Resolution', 'Status']],
        left_on='fix_hash', right_on='commit_id', how='left')
    evidencias['relato_fixed'] = evidencias.Resolution.eq('Fixed')
    etapas = {'szz_bruto': links, 'ligado_issue': ligados, 'apos_temporal': temporal,
              'apos_fixcount': apos_fix, 'apos_bugcount': apos_bug, 'apos_ast': aceitos}
    return etapas, evidencias, relatorios, {'max_fixcount': max_fix, 'max_bugcount': max_bug}


def criar_proveniencia(total, etapas, evidencias, found, chaves, limpos):
    p = total[['commit_id', 'project', 'buggy', 'fix', 'author_date', 'year']].copy()
    p['project_familia'] = familia(p.project)
    for nome, etapa in etapas.items():
        p[nome] = p.commit_id.isin(etapa.bug_hash)
    fixes_sem_links = set(found.commit_id) - set(etapas['szz_bruto'].fix_hash)
    p['origem_limpo_residual'] = p.commit_id.isin(limpos.commit_id)
    p['origem_fix_sem_bug_rastreado'] = p.commit_id.isin(fixes_sem_links & chaves)
    p['buggy_reconstruido'] = p.apos_ast
    p['fix_reconstruido'] = p.commit_id.isin(found.commit_id) & ~p.origem_limpo_residual
    p['rotulo_divergente'] = p.buggy.ne(p.buggy_reconstruido)
    p['negativo_candidato_szz_descartado'] = ~p.buggy & p.szz_bruto
    p['motivo_descarte'] = np.select([
        p.negativo_candidato_szz_descartado & ~p.apos_temporal,
        p.negativo_candidato_szz_descartado & p.apos_temporal & ~p.apos_fixcount,
        p.negativo_candidato_szz_descartado & p.apos_fixcount & ~p.apos_bugcount],
        ['ordem_temporal', 'fixcount_acima_limite', 'bugcount_acima_limite'], default='')
    grupos = evidencias.groupby('bug_hash')
    primeira = grupos.fix_date.min()
    p['primeira_correcao_vinculada'] = pd.to_datetime(p.commit_id.map(primeira), utc=True)
    p['n_pares_aceitos'] = p.commit_id.map(etapas['apos_ast'].groupby('bug_hash').size()).fillna(0).astype(int)
    p['algum_relato_fixed'] = p.commit_id.map(grupos.relato_fixed.any()).eq(True)
    p['positivo_sem_relato_fixed'] = p.buggy & ~p.algum_relato_fixed
    return p


def auditar_mensagens(relatorios, repos):
    """Confere todas as ligações issue→fix em objetos Git, sem baixar ou executar código."""
    resultados = []
    mapa = {'AMQ': 'activemq', 'HDFS': 'hadoop-hdfs', 'MAPREDUCE': 'hadoop-mapreduce'}
    for projeto, grupo in relatorios.groupby('project', sort=True):
        nome = mapa.get(projeto, projeto.lower())
        shas = set(grupo.commit_id)
        roteamento = {}
        for repo in [repos / f'{nome}.git'] + ([repos / 'hadoop.git'] if projeto in ['HDFS', 'MAPREDUCE'] else []):
            disponiveis = presentes(repo, sorted(shas - set(roteamento)))
            roteamento.update({s: repo for s in disponiveis})
        registros = {}
        for repo in sorted(set(roteamento.values())):
            alvos = sorted(s for s, r in roteamento.items() if r == repo)
            raw = git(repo, ['cat-file', '--batch'], input=('\n'.join(alvos) + '\n').encode(), timeout=180).stdout
            pos = 0
            for sha in alvos:
                fim = raw.index(b'\n', pos)
                identificado, tipo, tamanho = raw[pos:fim].split()
                if identificado.decode() != sha or tipo != b'commit':
                    raise ValueError('Objeto Git inesperado na auditoria de mensagens.')
                inicio = fim + 1; fim = inicio + int(tamanho)
                cabecalho, mensagem = raw[inicio:fim].split(b'\n\n', 1)
                linha = next(l for l in cabecalho.splitlines() if l.startswith(b'committer '))
                timestamp = int(linha.rsplit(b' ', 2)[1])
                registros[sha] = (timestamp, mensagem.decode('utf-8', 'replace'), repo.name)
                pos = fim + 1
        for r in grupo.itertuples():
            registro = registros.get(r.commit_id)
            if registro is None:
                resultados.append({'issue_key': r.issue_key, 'fix_hash': r.commit_id, 'objeto_disponivel': False})
                continue
            timestamp, mensagem, repo = registro
            data_git = pd.Timestamp(timestamp, unit='s')
            resultados.append({'issue_key': r.issue_key, 'fix_hash': r.commit_id, 'objeto_disponivel': True,
                'repositorio_git': repo, 'fix_date_git': data_git.tz_localize('UTC'),
                **conferir_chave(r.issue_key, mensagem),
                'correcao_antes_relato_mais_1d': bool(data_git < r.Created - pd.Timedelta(days=1)),
                'assunto': mensagem.splitlines()[0] if mensagem.splitlines() else ''})
        print(f'{projeto}: {len(grupo)} associações; {len(registros)}/{len(shas)} objetos Git lidos.', flush=True)
    return pd.DataFrame(resultados)


def anexar_alertas(p, evidencias, mensagens):
    """Sinais para revisão, sem remover pares ou substituir a marca publicada."""
    evidencias = evidencias.merge(mensagens.drop(columns='objeto_disponivel'),
                                 on=['fix_hash', 'issue_key'], how='left', validate='many_to_one')
    evidencias['evidencia_sem_alerta'] = (evidencias.relato_fixed
        & evidencias.chave_flexivel_na_mensagem.eq(True)
        & evidencias.correcao_antes_relato_mais_1d.eq(False))
    por_bug = evidencias.groupby('bug_hash')
    p = p.copy()
    for coluna, valores in {
        'algum_vinculo_sem_chave': por_bug.chave_flexivel_na_mensagem.agg(lambda s: s.eq(False).any()),
        'alguma_colisao_issue_pr': por_bug.possivel_colisao_issue_pr.any(),
        'alguma_correcao_antes_relato': por_bug.correcao_antes_relato_mais_1d.any(),
        'alguma_evidencia_sem_alerta': por_bug.evidencia_sem_alerta.any(),
    }.items():
        p[coluna] = p.commit_id.map(valores).eq(True)
    p['positivo_so_vinculos_sem_chave'] = p.buggy & ~p.commit_id.map(por_bug.chave_flexivel_na_mensagem.any()).eq(True)
    p['positivo_sem_evidencia_sem_alerta'] = p.buggy & ~p.alguma_evidencia_sem_alerta
    p['positivo_para_revisao'] = p.buggy & (p.positivo_sem_relato_fixed | p.algum_vinculo_sem_chave
                                                       | p.alguma_correcao_antes_relato)
    return p, evidencias


def resumo_grupos(frame, coluna):
    return frame.groupby(coluna, sort=True).agg(commits=('buggy', 'size'), positivos=('buggy', 'sum'),
        taxa_buggy=('buggy', 'mean')).reset_index()


def erros_familia(oof):
    grupos = oof.assign(project_familia=familia(oof.project)).groupby('project_familia')
    linhas = []
    for nome, d in grupos:
        tn, fp, fn, tp = confusion_matrix(d.buggy, d.previsao, labels=[0, 1]).ravel()
        linhas.append({'project_familia': nome, 'commits': len(d), 'taxa_buggy': d.buggy.mean(),
            'precision': precision_score(d.buggy, d.previsao, zero_division=0),
            'recall': recall_score(d.buggy, d.previsao, zero_division=0),
            'f1': f1_score(d.buggy, d.previsao, zero_division=0),
            'verdadeiros_negativos': int(tn), 'falsos_positivos': int(fp),
            'falsos_negativos': int(fn), 'verdadeiros_positivos': int(tp)})
    return pd.DataFrame(linhas)


def executar(pacote, repos):
    PASTA.mkdir(parents=True, exist_ok=True)
    modelo_path = ROOT / 'artefatos/melhorias/pipeline.joblib'
    hash_modelo = hashlib.sha256(modelo_path.read_bytes()).hexdigest()
    total, issues, found, links, chaves, limpos, hashes = ler_fonte(pacote)
    etapas, evidencias, relatorios, limites = reconstruir(issues, found, links, chaves)
    p = criar_proveniencia(total, etapas, evidencias, found, chaves, limpos)
    if total.commit_id.duplicated().any() or p.rotulo_divergente.any():
        raise ValueError('Divergência de rótulos: investigar antes de publicar a auditoria.')
    if p.fix.ne(p.fix_reconstruido).any():
        raise ValueError('A proveniência da coluna fix não reproduziu a fonte.')
    if not (p.buggy | p.origem_limpo_residual | p.origem_fix_sem_bug_rastreado).all():
        raise ValueError('Há commits sem proveniência.')
    evidencias['fix_date'] = pd.to_datetime(evidencias.fix_date, utc=True)
    evidencias['bug_date'] = pd.to_datetime(evidencias.bug_date, utc=True)
    mensagens = auditar_mensagens(relatorios, repos)
    # Disponibilidade temporal se baseia em timestamps Git, não no fuso desconhecido do Jira.
    conferencia = evidencias.merge(mensagens[['fix_hash', 'issue_key', 'fix_date_git']], on=['fix_hash', 'issue_key'], how='left')
    if conferencia.fix_date_git.isna().any() or conferencia.fix_date.ne(conferencia.fix_date_git).any():
        raise ValueError('Datas de correção do pacote divergem do Git ou faltam objetos.')
    p, evidencias = anexar_alertas(p, evidencias, mensagens)
    primeira = p.set_index('commit_id').primeira_correcao_vinculada
    treino, teste, folds, info = carregar_dividir()
    linhas = []
    for nome, trecho in [('treino_final', treino), *[(f'fold_{k}', treino.iloc[it]) for k, (it, _) in enumerate(folds, 1)]]:
        datas = trecho.commit_id.map(primeira)
        qtd = int((trecho.buggy.eq(1) & datas.gt(trecho.data.max())).sum())
        linhas.append({'trecho': nome, 'fim_treino': str(trecho.data.max()), 'commits_treino': len(trecho),
            'positivos_treino': int(trecho.buggy.sum()), 'primeira_correcao_posterior': qtd,
            'fracao_dos_positivos': qtd / trecho.buggy.sum()})
    temporal = pd.DataFrame(linhas)
    parte = pd.concat([treino[['commit_id']].assign(particao='treino'),
        teste[['commit_id', 'data']].assign(particao=lambda d: np.where(d.data.dt.year < 2019, 'avaliacao_ate_2018', '2019'))]).drop(columns='data')
    p = p.merge(parte, on='commit_id', how='left', validate='one_to_one').fillna({'particao': 'excluido_extracao'})
    por_ano = p.groupby(['particao', 'year']).agg(commits=('buggy', 'size'), positivos=('buggy', 'sum'),
        negativos_candidatos_descartados=('negativo_candidato_szz_descartado', 'sum'),
        positivos_sem_fixed=('positivo_sem_relato_fixed', 'sum')).reset_index()
    fonte_grupos = resumo_grupos(p, 'project')
    familias = resumo_grupos(p, 'project_familia')
    oof_path = ROOT / 'resultados/melhorias/previsoes_validacao.parquet'
    oof = pd.read_parquet(oof_path)
    erros = erros_familia(oof)
    assert erros.commits.sum() == len(oof)
    assert erros.falsos_positivos.sum() == ((oof.buggy == 0) & (oof.previsao == 1)).sum()
    etapas_resumo = pd.DataFrame([{'etapa': nome, 'pares': len(d), 'bugs_distintos': d.bug_hash.nunique(),
        'negativos_publicados_nesta_etapa': int((~p.buggy & p.commit_id.isin(d.bug_hash)).sum())} for nome, d in etapas.items()])
    revisar = evidencias[evidencias.bug_hash.isin(p.loc[p.positivo_para_revisao, 'commit_id'])].copy()
    # Mantém também as evidências alternativas: ver um vínculo suspeito não basta para inverter y.
    revisar = revisar.merge(p[['commit_id', 'particao', 'positivo_sem_relato_fixed',
        'positivo_so_vinculos_sem_chave', 'positivo_sem_evidencia_sem_alerta']],
        left_on='bug_hash', right_on='commit_id', suffixes=('', '_bug'), validate='many_to_one')
    revisar['Resolution'] = revisar.Resolution.fillna('Ausente no export')
    revisar['bug_url'] = 'https://github.com/' + revisar.project.map(lambda x: {'AMQ': 'apache/activemq',
        'HDFS': 'apache/hadoop', 'MAPREDUCE': 'apache/hadoop'}.get(x, 'apache/' + x.lower())) + '/commit/' + revisar.bug_hash
    revisar['issue_url'] = 'https://issues.apache.org/jira/browse/' + revisar.issue_key
    revisar['fix_url'] = revisar.bug_url.str.rsplit('/', n=1).str[0] + '/' + revisar.fix_hash
    vinculos = mensagens.merge(relatorios[['issue_key', 'commit_id', 'Resolution', 'Status', 'Created']],
        left_on=['issue_key', 'fix_hash'], right_on=['issue_key', 'commit_id'], validate='one_to_one')
    contagens = evidencias.groupby(['fix_hash', 'issue_key']).bug_hash.nunique()
    vinculos['positivos_associados'] = pd.MultiIndex.from_frame(vinculos[['fix_hash', 'issue_key']]).map(contagens).fillna(0).astype(int)
    vinculos = vinculos[vinculos.chave_flexivel_na_mensagem.eq(False)
                        | vinculos.correcao_antes_relato_mais_1d.eq(True) | vinculos.Resolution.ne('Fixed')].copy()
    negativos = p[p.negativo_candidato_szz_descartado]
    resumo = {
        'linhas': len(p), 'positivos': int(p.buggy.sum()), 'negativos': int((~p.buggy).sum()),
        'duplicatas_sha': int(p.commit_id.duplicated().sum()), 'rotulos_divergentes': int(p.rotulo_divergente.sum()),
        'fix_divergentes': int(p.fix.ne(p.fix_reconstruido).sum()),
        'sem_proveniencia': int((~(p.buggy | p.origem_limpo_residual | p.origem_fix_sem_bug_rastreado)).sum()),
        'negativos_candidatos_szz_descartados': len(negativos), 'motivos_descarte': negativos.motivo_descarte.value_counts().to_dict(),
        'positivos_sem_relato_fixed': int(p.positivo_sem_relato_fixed.sum()),
        'positivos_com_vinculo_sem_chave': int(p.algum_vinculo_sem_chave.sum()),
        'positivos_com_colisao_issue_pr': int(p.alguma_colisao_issue_pr.sum()),
        'positivos_so_vinculos_sem_chave': int(p.positivo_so_vinculos_sem_chave.sum()),
        'positivos_com_correcao_antes_relato': int(p.alguma_correcao_antes_relato.sum()),
        'positivos_para_revisao': int(p.positivo_para_revisao.sum()),
        'positivos_sem_evidencia_sem_alerta': int(p.positivo_sem_evidencia_sem_alerta.sum()),
        'reports_sem_resolution_fixed': int(issues.Resolution.ne('Fixed').sum()),
        'reports_associados_sem_fixed': int(relatorios.Resolution.ne('Fixed').sum()),
        'limites': limites, 'disponibilidade_temporal': linhas,
        'familia_hadoop': familias[familias.project_familia.eq('apache/hadoop (familia)')].iloc[0].to_dict(),
        'vinculos_git': {'associacoes': len(mensagens), 'objetos_indisponiveis': int((~mensagens.objeto_disponivel).sum()),
            'sem_chave_literal': int(mensagens.chave_literal_na_mensagem.eq(False).sum()),
            'sem_chave_flexivel': int(mensagens.chave_flexivel_na_mensagem.eq(False).sum()),
            'possivel_colisao_issue_pr': int(mensagens.possivel_colisao_issue_pr.eq(True).sum()),
            'fix_antes_report_mais_1d': int(mensagens.correcao_antes_relato_mais_1d.eq(True).sum()),
            'datas_fix_aceitas_conferidas': True},
        'rotulos_alterados': 0, 'modelo_alterado': False,
        'conclusao': 'Agrupamento por família corrigido nos relatórios; nenhuma troca automática de rótulo é justificada.',
        'limites_auditoria': ['Reprodução do SZZ verifica consistência interna, não verdade do defeito.',
            'O fuso de Created do export Jira não é declarado; mantida a comparação sem fuso do código original.',
            'Primeira correção vinculada é evidência disponível no pacote, não garantia da primeira descoberta real.',
            'Negativos residuais não têm certificação de ausência de defeito.',
            'Campos de proveniência, datas futuras e flags de revisão não podem entrar como preditores.'],
        'fontes': FONTES,
    }
    p.to_parquet(PASTA / 'proveniencia.parquet', index=False)
    evidencias.to_parquet(PASTA / 'evidencias_positivos.parquet', index=False)
    mensagens.to_parquet(PASTA / 'vinculos_git.parquet', index=False)
    fila = p[p.positivo_para_revisao].copy()
    fila['prioridade_revisao'] = np.select([fila.positivo_so_vinculos_sem_chave,
        fila.positivo_sem_relato_fixed, fila.positivo_sem_evidencia_sem_alerta], [1, 2, 3], default=4)
    fila.sort_values(['prioridade_revisao', 'project', 'commit_id']).to_csv(PASTA / 'fila_commits.csv', index=False)
    revisar.to_csv(PASTA / 'positivos_para_revisao.csv', index=False)
    vinculos.to_csv(PASTA / 'vinculos_para_revisao.csv', index=False)
    negativos.to_csv(PASTA / 'negativos_candidatos_descartados.csv', index=False)
    for nome, tabela in [('etapas', etapas_resumo), ('grupos_fonte', fonte_grupos), ('familias', familias),
                          ('disponibilidade_temporal', temporal), ('por_periodo', por_ano), ('erros_por_familia', erros)]:
        tabela.to_csv(PASTA / (nome + '.csv'), index=False)
    salvar_json(PASTA / 'resumo.json', resumo)
    # .gitattributes usa LF: o hash precisa sobreviver a um checkout em outro sistema.
    for arquivo in PASTA.iterdir():
        if arquivo.suffix in {'.csv', '.json'}:
            arquivo.write_bytes(arquivo.read_bytes().replace(b'\r\n', b'\n'))
    assert hashlib.sha256(modelo_path.read_bytes()).hexdigest() == hash_modelo
    assert hashlib.sha256((ROOT / 'data/apachejit_total.csv').read_bytes()).hexdigest() == CSV_SHA256
    salvar_json(PASTA / 'manifesto.json', {'codigo_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'pacote_md5': MD5_PACOTE, 'csv_sha256': CSV_SHA256, 'fontes_no_pacote_sha256': hashes,
        'pipeline_sha256_antes_e_depois': hash_modelo,
        'previsoes_validacao_sha256': hashlib.sha256(oof_path.read_bytes()).hexdigest(),
        'arquivos_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in PASTA.iterdir() if f.is_file() and f.name != 'manifesto.json'}})
    manifesto = PASTA / 'manifesto.json'
    manifesto.write_bytes(manifesto.read_bytes().replace(b'\r\n', b'\n'))
    print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)


def carregar_auditoria():
    meta = json.loads((PASTA / 'manifesto.json').read_text(encoding='utf-8'))
    if meta['codigo_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Código da auditoria alterado; refaça a auditoria.')
    for caminho, chave in [('data/apachejit_total.csv', 'csv_sha256'),
                           ('artefatos/melhorias/pipeline.joblib', 'pipeline_sha256_antes_e_depois'),
                           ('resultados/melhorias/previsoes_validacao.parquet', 'previsoes_validacao_sha256')]:
        if hashlib.sha256((ROOT / caminho).read_bytes()).hexdigest() != meta[chave]:
            raise ValueError('A base, o modelo ou as previsões mudaram desde a auditoria: ' + caminho)
    for nome, digest in meta['arquivos_sha256'].items():
        if hashlib.sha256((PASTA / nome).read_bytes()).hexdigest() != digest:
            raise ValueError('Resultado da auditoria diferente do manifesto: ' + nome)
    return json.loads((PASTA / 'resumo.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pacote', type=Path, default=ROOT.parent / 'risco_commits/data/apachejit_dataset_replication.zip')
    parser.add_argument('--repos', type=Path, default=pasta_repos())
    args = parser.parse_args()
    executar(args.pacote, args.repos)
