"""Busca ampliada exclusivamente no período de treino. Retomável por configuração.

python buscar_melhorias.py
O período final só é aberto posteriormente por finalizar_melhorias.py.
"""
import hashlib
import importlib.metadata
import json
from time import perf_counter

import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from dados import carregar_dividir, salvar_json
from esquema import ROOT, SEED
from treinamento import construir
from revisar_balanceamento import dividir_limiar, escolher_limiar, medir
from novas_pistas import EngenhariaPistas, VARIANTES, colunas_entrada
from extrair_dispersao import carregar_pistas, CACHE

PASTA = ROOT / 'resultados/melhorias'
N_TRIALS = {'LightGBM': 24, 'XGBoost': 12}
PILOTO = {
    'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 15, 'min_child_samples': 200,
    'reg_alpha': 1.0, 'reg_lambda': 30.0, 'colsample_bytree': 0.8,
    'subsample': 0.8, 'subsample_freq': 1,
}


def construir_ampliado(config):
    pipe = construir(config['modelo'], config['parametros'])
    if config['pistas'] != 'originais':
        pipe = Pipeline([('pistas', EngenhariaPistas(config['pistas'])), *pipe.steps])
    return pipe


def ajustar_ampliado(treino, config):
    """O limiar vê só o trecho interno; pesos são calculados por fit."""
    colunas = colunas_entrada(config['pistas'])
    anterior, interno = dividir_limiar(treino)
    contagens, pesos = [], []
    def fit(frame):
        pipe = construir_ampliado(config)
        kwargs = {}
        contagens.append({str(k): int(v) for k, v in frame.buggy.value_counts().sort_index().items()})
        if config['pesos'] == 'balanced':
            w = compute_sample_weight('balanced', frame.buggy)
            kwargs['modelo__sample_weight'] = w
            pesos.append({str(c): float(w[frame.buggy.to_numpy() == c][0]) for c in (0, 1)})
        else:
            pesos.append({'0': 1.0, '1': 1.0})
        pipe.fit(frame[colunas], frame.buggy, **kwargs)
        return pipe
    auxiliar = fit(anterior)
    limiar, curva = escolher_limiar(interno.buggy, auxiliar.predict_proba(interno[colunas])[:, 1])
    pipe = fit(treino)
    pipe.limiar_decisao_ = limiar
    pipe.variante_pistas_ = config['pistas']
    auditoria = {
        'n_ajuste_interno': len(anterior), 'n_validacao_interna': len(interno), 'n_treino': len(treino),
        'fim_ajuste_interno': str(anterior.data.max()), 'inicio_validacao_interna': str(interno.data.min()),
        'fim_treino': str(treino.data.max()), 'contagens_por_fit': contagens, 'pesos_por_fit': pesos,
    }
    return pipe, curva, auditoria


def controles():
    original = json.loads((ROOT / 'resultados/selecao.json').read_text(encoding='utf-8'))['parametros']
    return [{'id': f'{nome}_{pistas}', 'tipo': 'controle', 'modelo': 'LightGBM',
             'pistas': pistas, 'pesos': None, 'parametros': params}
            for nome, params in [('anterior', original), ('regularizado', PILOTO)]
            for pistas in VARIANTES]


def avaliar_config(config, treino, folds):
    json_path = PASTA / 'candidatos' / (config['id'] + '.json')
    oof_path = json_path.with_suffix('.parquet')
    if json_path.exists() and oof_path.exists():
        salvo = json.loads(json_path.read_text(encoding='utf-8'))
        if salvo['config'] != config:
            raise ValueError('Configuração diferente do cache.')
        return salvo
    inicio = perf_counter()
    linhas, previsoes = [], []
    for k, (it, iv) in enumerate(folds, 1):
        passado, futuro = treino.iloc[it], treino.iloc[iv]
        assert passado.data.max() < futuro.data.min()
        pipe, _, auditoria = ajustar_ampliado(passado, config)
        score = pipe.predict_proba(futuro[colunas_entrada(config['pistas'])])[:, 1]
        m = medir(futuro.buggy, score, pipe.limiar_decisao_)
        linhas.append({'fold': k, 'limiar': pipe.limiar_decisao_, **m, **auditoria,
                       'n_validacao': len(futuro), 'inicio_validacao': str(futuro.data.min()),
                       'fim_validacao': str(futuro.data.max())})
        previsoes.append(futuro[['commit_id', 'project', 'data', 'buggy']].assign(
            candidato=config['id'], fold=k, score=score,
            previsao=(score >= pipe.limiar_decisao_).astype(int), limiar=pipe.limiar_decisao_))
    d = pd.DataFrame(linhas)
    resultado = {'config': config, 'folds': linhas, 'tempo_s': perf_counter() - inicio,
                 'f1_validacao': float(d.f1.mean()), 'desvio_f1': float(d.f1.std(ddof=0)),
                 **{m: float(d[m].mean()) for m in ['precision', 'recall', 'average_precision', 'roc_auc']}}
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(previsoes, ignore_index=True).to_parquet(oof_path, index=False)
    salvar_json(json_path, resultado)
    print(f"{config['id']} / {config['pistas']} / pesos={config['pesos']}: "
          f"F1={resultado['f1_validacao']:.4f}, AP={resultado['average_precision']:.4f} "
          f"({resultado['tempo_s']:.0f}s)", flush=True)
    return resultado


def sugerir_ampliado(trial, modelo):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 100.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
    }
    if modelo == 'LightGBM':
        params.update(num_leaves=trial.suggest_categorical('num_leaves', [7, 15, 31, 63]),
                      min_child_samples=trial.suggest_int('min_child_samples', 50, 500, step=50),
                      min_split_gain=trial.suggest_float('min_split_gain', 0.0, 0.5), subsample_freq=1)
    else:
        params.update(max_depth=trial.suggest_int('max_depth', 2, 7),
                      min_child_weight=trial.suggest_float('min_child_weight', 1.0, 100.0, log=True),
                      gamma=trial.suggest_float('gamma', 0.0, 5.0))
    return {'id': f'{modelo.lower()}_{trial.number:02d}', 'tipo': 'optuna', 'modelo': modelo,
            'pistas': trial.suggest_categorical('pistas', list(VARIANTES)),
            'pesos': trial.suggest_categorical('pesos', [None, 'balanced']), 'parametros': params}


def assinatura_experimento(info):
    arquivos = ['buscar_melhorias.py', 'novas_pistas.py', 'extrair_dispersao.py',
                'PROTOCOLO_MELHORIAS.md', 'dados.py', 'extracao.py', 'esquema.py',
                'treinamento.py', 'revisar_balanceamento.py', 'resultados/selecao.json']
    conteudo = {'arquivos': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in arquivos},
                'csv_sha256': info['csv_sha256'], 'extrator_sha256': info['extrator_sha256'],
                'novas_pistas_sha256': hashlib.sha256(CACHE.read_bytes()).hexdigest(),
                'n_trials': N_TRIALS, 'controles': controles(),
                'versoes': {p: importlib.metadata.version(p) for p in
                            ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'optuna']}}
    assinatura = hashlib.sha256(json.dumps(conteudo, sort_keys=True).encode()).hexdigest()
    return assinatura, conteudo


def executar():
    treino, _, folds, info = carregar_dividir()
    treino = carregar_pistas(treino)
    assinatura, manifesto = assinatura_experimento(info)
    path = PASTA / 'manifesto.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['assinatura'] != assinatura:
            raise ValueError('Busca alterada. Não reutilize resultados em outro protocolo/código.')
    else:
        salvar_json(path, {'assinatura': assinatura, **manifesto, 'holdout_usado_na_selecao': False})
    for config in controles():
        avaliar_config(config, treino, folds)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    armazenamento = 'sqlite:///' + (ROOT / 'data/busca_melhorias.db').as_posix()
    for modelo, n_trials in N_TRIALS.items():
        estudo = optuna.create_study(direction='maximize', study_name=f'melhorias_{modelo}_{assinatura[:10]}',
            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
            storage=armazenamento, load_if_exists=True)
        def objetivo(trial):
            config = sugerir_ampliado(trial, modelo)
            resultado = avaliar_config(config, treino, folds)
            trial.set_user_attr('config', config)
            return resultado['f1_validacao']
        if any(t.state not in (optuna.trial.TrialState.COMPLETE,) for t in estudo.trials):
            raise ValueError('Trial interrompido; audite o banco antes de retomar.')
        estudo.optimize(objetivo, n_trials=max(0, n_trials - len(estudo.trials)))
        estudo.trials_dataframe().to_csv(PASTA / f'trials_{modelo.lower()}.csv', index=False)
    resultados = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((PASTA / 'candidatos').glob('*.json'))]
    tabela = pd.DataFrame([{**r['config'], **{k: v for k, v in r.items() if k not in ['config', 'folds']},
                           'n_pistas': len(colunas_entrada(r['config']['pistas'])) +
                               (16 if r['config']['pistas'] != 'originais' else 0),
                           'n_arvores': r['config']['parametros']['n_estimators']} for r in resultados])
    tabela = tabela.sort_values(['f1_validacao', 'desvio_f1', 'n_pistas', 'n_arvores'],
                                ascending=[False, True, True, True], kind='stable')
    tabela.to_csv(PASTA / 'comparacao.csv', index=False)
    selecionado = next(r for r in resultados if r['config']['id'] == tabela.iloc[0]['id'])
    salvar_json(PASTA / 'selecao_validacao.json', {'assinatura': assinatura, **selecionado,
                'regra': 'Maior F1 médio; menor desvio; menos pistas; menos árvores.',
                'holdout_usado_na_selecao': False, 'configuracoes_concluidas': len(resultados)})
    print(tabela[['id', 'pistas', 'pesos', 'f1_validacao', 'desvio_f1', 'precision', 'recall',
                   'average_precision']].head(12).round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    executar()
