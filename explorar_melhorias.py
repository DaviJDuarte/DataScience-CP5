"""Piloto de desenvolvimento: regularização e janelas temporais, sem avaliar o holdout.

Mantém as 48 entradas e o mesmo ajuste temporal interno do limiar. Não troca o modelo
do aplicativo. Os candidatos são definidos antes da execução; não são novos testes independentes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from dados import carregar_dividir, salvar_json
from esquema import ROOT
from modelo import prever
from revisar_balanceamento import ajustar, medir

PASTA = ROOT / 'resultados/exploracao_regularizacao'
CANDIDATOS = [
    {'nome': 'referencia', 'parametros': {}, 'fracao_recente': 1.0},
    {'nome': 'folhas_15_l2_10', 'parametros': {
        'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 15, 'min_child_samples': 100,
        'reg_alpha': 0.1, 'reg_lambda': 10.0, 'colsample_bytree': 0.8, 'subsample': 0.8, 'subsample_freq': 1},
     'fracao_recente': 1.0},
    {'nome': 'folhas_15_l2_30', 'parametros': {
        'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 15, 'min_child_samples': 200,
        'reg_alpha': 1.0, 'reg_lambda': 30.0, 'colsample_bytree': 0.8, 'subsample': 0.8, 'subsample_freq': 1},
     'fracao_recente': 1.0},
    {'nome': 'folhas_7_l2_50', 'parametros': {
        'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 7, 'min_child_samples': 100,
        'reg_alpha': 1.0, 'reg_lambda': 50.0, 'colsample_bytree': 0.9, 'subsample': 0.9, 'subsample_freq': 1},
     'fracao_recente': 1.0},
    {'nome': 'folhas_31_l2_5', 'parametros': {
        'n_estimators': 700, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 150,
        'reg_alpha': 0.1, 'reg_lambda': 5.0, 'colsample_bytree': 0.9, 'subsample': 0.8, 'subsample_freq': 1},
     'fracao_recente': 1.0},
    {'nome': 'folhas_27_l2_10', 'parametros': {
        'n_estimators': 500, 'learning_rate': 0.04, 'num_leaves': 27, 'min_child_samples': 120,
        'reg_alpha': 0.5, 'reg_lambda': 10.0, 'colsample_bytree': 0.8, 'subsample': 0.8, 'subsample_freq': 1},
     'fracao_recente': 1.0},
    {'nome': 'janela_recente_80pct', 'parametros': {}, 'fracao_recente': 0.8},
    {'nome': 'janela_recente_60pct', 'parametros': {}, 'fracao_recente': 0.6},
]


def executar():
    # O período final não é utilizado em nenhuma métrica ou escolha deste piloto.
    treino, _, folds, divisao = carregar_dividir()
    config = json.loads((ROOT/'resultados/selecao.json').read_text(encoding='utf-8'))
    PASTA.mkdir(exist_ok=True)
    manifesto = {'candidatos': CANDIDATOS, 'modelo_base': config,
                 'codigo_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'csv_sha256': divisao['csv_sha256'], 'extrator_sha256': divisao['extrator_sha256'],
                 'selecao_limiar': '80%/20% temporal interno, mesma regra da comparação anterior',
                 'holdout_avaliado': False,
                 'ressalva': 'Comparação exploratória em folds já conhecidos; não estima ganho independente.'}
    salvar_json(PASTA/'manifesto.json', manifesto)
    linhas, previsoes = [], []
    for candidato in CANDIDATOS:
        params = {**config['parametros'], **candidato['parametros']}
        for k, (it, iv) in enumerate(folds, 1):
            passado, futuro = treino.iloc[it], treino.iloc[iv]
            datas = np.sort(passado.committer_date_git.unique())
            inicio = int(len(datas)*(1-candidato['fracao_recente']))
            passado = passado[passado.committer_date_git >= datas[inicio]]
            assert passado.data.max() < futuro.data.min()
            pipe, _, auditoria = ajustar(passado, params, None, True)
            p = prever(pipe, futuro)
            m = medir(futuro.buggy, p.score, pipe.limiar_decisao_)
            linhas.append({'candidato': candidato['nome'], 'fold': k, 'n_treino': len(passado),
                           'n_validacao': len(futuro), 'limiar': pipe.limiar_decisao_, **m,
                           'treino_inicio': str(passado.data.min()), 'treino_fim': str(passado.data.max()),
                           'validacao_inicio': str(futuro.data.min()), **auditoria})
            previsoes.append(futuro[['commit_id','project','buggy']].assign(
                candidato=candidato['nome'], fold=k, score=p.score, previsao=p.previsao))
            print(f"{candidato['nome']} | fold {k}: F1={m['f1']:.4f}, AP={m['average_precision']:.4f}", flush=True)
        salvar_json(PASTA/'folds.json', linhas)
    detalhe = pd.DataFrame(linhas)
    resumo = detalhe.groupby('candidato', sort=False).agg(
        f1=('f1','mean'), desvio_f1=('f1',lambda x: x.std(ddof=0)),
        precision=('precision','mean'), recall=('recall','mean'),
        average_precision=('average_precision','mean'))
    resumo['delta_f1'] = resumo.f1 - resumo.loc['referencia','f1']
    resumo.sort_values('f1',ascending=False).to_csv(PASTA/'comparacao.csv')
    oof = pd.concat(previsoes, ignore_index=True)
    oof.to_csv(PASTA/'previsoes_validacao.csv.gz', index=False)
    auditoria_projetos=[]
    for nome, g in oof[oof.candidato.eq('referencia')].groupby('project'):
        auditoria_projetos.append({'project':nome,'commits':len(g),'taxa_buggy':g.buggy.mean(),
            'f1':f1_score(g.buggy,g.previsao,zero_division=0),
            'precision':precision_score(g.buggy,g.previsao,zero_division=0),
            'recall':recall_score(g.buggy,g.previsao,zero_division=0),
            'falsos_positivos':int(((g.buggy==0)&(g.previsao==1)).sum())})
    pd.DataFrame(auditoria_projetos).to_csv(PASTA/'erros_por_projeto.csv',index=False)
    print(resumo.sort_values('f1',ascending=False).round(4).to_string(), flush=True)


if __name__ == '__main__':
    executar()
