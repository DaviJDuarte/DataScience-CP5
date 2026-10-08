"""Congela a escolha da validação, avalia uma vez e salva o modelo para o app.

O teste conhecido é exploratório. Seus resultados não realimentam a busca.
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from dados import carregar_dividir, salvar_json
from esquema import ROOT, SEED
from modelo import FAIXAS, carregar_modelo, prever
from revisar_balanceamento import separar_periodos, medir
from novas_pistas import colunas_entrada
from extrair_dispersao import carregar_pistas
from buscar_melhorias import PASTA, ajustar_ampliado, assinatura_experimento

ARTEFATOS = ROOT / 'artefatos/melhorias'


def intervalo_pareado(frame, nova, anterior, repeticoes=1000):
    """Bootstrap de semanas completas: incerteza descritiva, sem corrigir seleção."""
    semana = frame.data.dt.strftime('%G-%V')
    quadros = []
    for pred in (nova, anterior):
        y = frame.buggy.to_numpy()
        quadros.append(pd.DataFrame({'semana': semana, 'tp': (y == 1) & (pred == 1),
            'fp': (y == 0) & (pred == 1), 'fn': (y == 1) & (pred == 0)})
            .groupby('semana')[['tp', 'fp', 'fn']].sum().to_numpy())
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(quadros[0]), size=(repeticoes, len(quadros[0])))
    pontos = []
    for q in quadros:
        tp, fp, fn = q[indices].sum(axis=1).T
        pontos.append(np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(len(tp), dtype=float), where=(2*tp+fp+fn) > 0))
    intervalo = np.quantile(pontos[0] - pontos[1], [0.025, 0.975])
    return {'delta_f1_ic95': intervalo.tolist(), 'repeticoes': repeticoes, 'blocos': len(quadros[0]),
            'unidade': 'semana de integração, amostragem pareada',
            'limite': 'Descritivo do período conhecido; não corrige adaptação ao holdout nem prova generalização.'}


def avaliar(pipe, original, anterior, frame, periodo):
    nova, base, previa = [prever(p, frame) for p in (pipe, original, anterior)]
    tabela = frame[['commit_id', 'project', 'data', *colunas_entrada(pipe.variante_pistas_), 'buggy']].copy()
    tabela[['score', 'previsao']] = nova[['score', 'previsao']]
    tabela[['score_original', 'previsao_original']] = base[['score', 'previsao']]
    tabela[['score_anterior', 'previsao_anterior']] = previa[['score', 'previsao']]
    tabela['caso'] = np.select([
        (tabela.buggy == 1) & (tabela.previsao == 1),
        (tabela.buggy == 0) & (tabela.previsao == 0),
        (tabela.buggy == 0) & (tabela.previsao == 1)],
        ['Verdadeiro positivo', 'Verdadeiro negativo', 'Falso positivo'], default='Falso negativo')
    resumo = {'periodo': periodo, 'n_teste': len(frame), 'taxa_buggy': float(frame.buggy.mean()),
              'inicio': str(frame.data.min()), 'fim': str(frame.data.max()),
              'metricas': medir(frame.buggy, nova.score, pipe.limiar_decisao_),
              'original_mesmos_commits': medir(frame.buggy, base.score, 0.5),
              'anterior_mesmos_commits': medir(frame.buggy, previa.score, anterior.limiar_decisao_),
              'referencia_majoritaria': medir(frame.buggy, np.zeros(len(frame))),
              'matriz_confusao': confusion_matrix(frame.buggy, nova.previsao, labels=[0, 1]).tolist(),
              'matriz_original': confusion_matrix(frame.buggy, base.previsao, labels=[0, 1]).tolist(),
              'matriz_anterior': confusion_matrix(frame.buggy, previa.previsao, labels=[0, 1]).tolist(),
              'incerteza_vs_anterior': intervalo_pareado(frame, nova.previsao.to_numpy(), previa.previsao.to_numpy()),
              'avaliacoes_nesta_versao': 1,
              'ressalva': 'Holdout conhecido: reavaliação exploratória; não é teste novo/intocado.'}
    return resumo, tabela


def auditar_projetos(oof):
    linhas = []
    for projeto, g in oof.groupby('project'):
        linhas.append({'project': projeto, 'commits': len(g), 'taxa_buggy': g.buggy.mean(),
            'precision': precision_score(g.buggy, g.previsao, zero_division=0),
            'recall': recall_score(g.buggy, g.previsao, zero_division=0),
            'f1': f1_score(g.buggy, g.previsao, zero_division=0),
            'falsos_positivos': int(((g.buggy == 0) & (g.previsao == 1)).sum())})
    return pd.DataFrame(linhas)


def executar():
    escolha = json.loads((PASTA / 'selecao_validacao.json').read_text(encoding='utf-8'))
    treino, teste, _, info = carregar_dividir()
    assinatura, _ = assinatura_experimento(info)
    if escolha['assinatura'] != assinatura:
        raise ValueError('Busca ou dados alterados desde a escolha.')
    if (ARTEFATOS / 'metadados.json').exists():
        meta = json.loads((ARTEFATOS / 'metadados.json').read_text(encoding='utf-8'))
        if meta['assinatura_melhorias'] != assinatura:
            raise ValueError('Modelo salvo pertence a outra busca.')
        carregar_modelo()
        print('Modelo já concluído; avaliação preservada.', flush=True)
        return
    original, meta_original = carregar_modelo(atual=False)
    # Referência imediatamente anterior, explicitamente independente da seleção ativa do app.
    pasta_anterior = ROOT / 'artefatos/revisao_balanceamento'
    meta_anterior = json.loads((pasta_anterior / 'metadados.json').read_text(encoding='utf-8'))
    if hashlib.sha256((pasta_anterior / 'pipeline.joblib').read_bytes()).hexdigest() != meta_anterior['pipeline_sha256']:
        raise ValueError('Referência anterior diferente do manifesto.')
    anterior = joblib.load(pasta_anterior / 'pipeline.joblib')
    treino, teste = carregar_pistas(treino), carregar_pistas(teste)
    principal, separado = separar_periodos(teste)
    config = escolha['config']
    pipe, curva, auditoria = ajustar_ampliado(treino, config)
    final = {**config, 'estrategia': 'Busca ampliada de pistas e regularização',
             'limiar': pipe.limiar_decisao_, 'ajuste_final': auditoria,
             'f1_validacao': escolha['f1_validacao'], 'desvio_f1': escolha['desvio_f1'],
             'regra': escolha['regra'], 'assinatura_melhorias': assinatura}
    # Escolha e limiar persistidos ANTES da primeira avaliação final deste candidato.
    salvar_json(PASTA / 'selecao.json', final)
    curva.to_csv(PASTA / 'curva_limiar_final.csv', index=False)
    resumo, previsoes = avaliar(pipe, original, anterior, principal, 'Avaliação principal: 06/10/2017 a 31/12/2018 (UTC).')
    sensibilidade, previsoes_2019 = avaliar(pipe, original, anterior, separado, 'Sensibilidade: integração em 2019 ou depois.')
    salvar_json(PASTA / 'teste_final.json', resumo)
    salvar_json(PASTA / 'sensibilidade_2019.json', sensibilidade)
    previsoes.to_parquet(PASTA / 'previsoes_teste.parquet', index=False)
    previsoes_2019.to_parquet(PASTA / 'previsoes_2019.parquet', index=False)
    pd.concat([treino[['commit_id', 'data']].assign(conjunto='treino'),
               principal[['commit_id', 'data']].assign(conjunto='teste_ate_2018'),
               separado[['commit_id', 'data']].assign(conjunto='sensibilidade_2019')]).to_csv(PASTA / 'particoes.csv', index=False)
    oof = pd.read_parquet(PASTA / 'candidatos' / (config['id'] + '.parquet'))
    oof.to_parquet(PASTA / 'previsoes_validacao.parquet', index=False)
    auditar_projetos(oof).to_csv(PASTA / 'erros_por_projeto.csv', index=False)
    exemplos = previsoes.groupby('caso', sort=True).head(1)
    restantes = previsoes.drop(exemplos.index)
    perto = restantes.assign(distancia=(restantes.score - pipe.limiar_decisao_).abs()).sort_values('distancia').head(1)
    claro = restantes.drop(perto.index).sort_values('score').head(1)
    exemplos = pd.concat([exemplos, perto, claro]).drop(columns='distancia', errors='ignore')
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    exemplos.to_csv(ARTEFATOS / 'exemplos.csv', index=False)
    nomes, minimos = zip(*FAIXAS)
    grupo = pd.cut(oof.score, [*minimos, np.inf], right=False, labels=list(nomes))
    faixas = oof.groupby(grupo, observed=False).agg(commits=('buggy', 'size'), taxa_buggy=('buggy', 'mean'))
    faixas = faixas.rename_axis('faixa').reset_index()
    faixas.insert(1, 'score_minimo', minimos); faixas['taxa_geral'] = oof.buggy.mean()
    faixas.to_csv(ARTEFATOS / 'faixas.csv', index=False)
    joblib.dump(pipe, ARTEFATOS / 'pipeline.joblib', compress=3)
    meta = {**meta_original, **final, 'variaveis': colunas_entrada(config['pistas']),
            'variaveis_modelo': list(pipe.named_steps['modelo'].feature_names_in_),
            'novas_pistas_sha256': hashlib.sha256((ROOT / 'novas_pistas.py').read_bytes()).hexdigest(),
            'extracao_dispersao_sha256': hashlib.sha256((ROOT / 'extrair_dispersao.py').read_bytes()).hexdigest(),
            'resultado_avaliacao': 'resultados/melhorias/teste_final.json',
            'divisao': {**info, 'n_teste': len(principal), 'n_sensibilidade_2019': len(separado)},
            'pipeline_sha256': hashlib.sha256((ARTEFATOS / 'pipeline.joblib').read_bytes()).hexdigest()}
    # O gap da configuração original não descreve este estimador.
    meta.pop('gap', None)
    # Ativação apenas quando todos os artefatos já existem.
    salvar_json(ARTEFATOS / 'metadados.json', meta)
    recarregado, _ = carregar_modelo()
    p = prever(recarregado, exemplos)
    np.testing.assert_allclose(p.score, exemplos.score, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(p.previsao, exemplos.previsao)
    print(json.dumps({'configuracao': final, 'principal': resumo, '2019': sensibilidade}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    executar()
