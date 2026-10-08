"""Pistas adicionais calculáveis no momento do commit, sem rótulos ou identidade.

Mantém o extrator original intacto para permitir comparações reproduzíveis.
As razões são calculadas dentro do pipeline, igualmente no treino e no app.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from esquema import FEATURES
from extracao import eh_teste, ExtracaoIncompativel

ESTATICAS = {
    'n_entropia': 'Dispersão das linhas alteradas entre arquivos (bits)',
    'n_entropia_normalizada': 'Dispersão dividida pelo máximo para o número de arquivos',
    'n_fracao_churn_max': 'Fração das linhas alteradas concentrada no maior arquivo',
    'n_coef_variacao_churn': 'Desvio das alterações por arquivo dividido pela média',
    'n_subsistemas': 'Pastas de primeiro nível distintas (raiz conta como uma)',
    'n_profundidade_media': 'Número médio de pastas no caminho dos arquivos',
    'n_fracao_ld_teste': 'Fração das remoções em arquivos de teste',
    'n_ld_max_arquivo': 'Maior remoção em um arquivo',
}
DERIVADAS = {
    'd_churn': 'Total de linhas adicionadas e removidas',
    'd_churn_por_arquivo': 'Linhas alteradas por arquivo',
    'd_fracao_java': 'Fração de arquivos Java',
    'd_fracao_testes': 'Fração de arquivos de teste',
    'd_fracao_producao': 'Fração de arquivos Java de produção',
    'd_fracao_churn_producao': 'Fração das linhas alteradas em Java de produção',
    'd_churn_por_arquivo_producao': 'Linhas alteradas por arquivo Java de produção',
    'd_concentracao_adicoes': 'Fração das adições no arquivo com mais adições',
    'd_arquivos_por_pasta': 'Arquivos alterados por pasta',
    'd_experiencia_autor_por_arquivo': 'Mudanças anteriores do autor por arquivo afetado',
    'd_mudancas_por_autor': 'Mudanças médias dos arquivos por autor anterior (+1)',
    'd_atividade_autor_por_dia': 'Eventos anteriores do autor por dia de experiência (+1)',
    'd_fracao_atividade_recente': 'Eventos na última semana / experiência do autor (+1)',
    'd_churn_por_experiencia': 'Linhas alteradas / experiência anterior do autor (+1)',
    'd_mudancas_por_dia': 'Mudanças anteriores dos arquivos por dia de histórico (+1)',
    'd_palavras_por_linha': 'Palavras por linha da mensagem',
}
VARIANTES = ('originais', 'razoes', 'dispersao')


def colunas_entrada(variante):
    if variante not in VARIANTES:
        raise ValueError('Conjunto de pistas desconhecido.')
    return FEATURES + (list(ESTATICAS) if variante == 'dispersao' else [])


def dispersao_commit(commit):
    """Somente o diff do alvo contra seu pai; nenhuma informação posterior."""
    arquivos = commit['arquivos']
    if (not arquivos or len(commit['pais']) > 1 or commit['submodulo']
            or any(a[4] for a in arquivos)):
        raise ExtracaoIncompativel('Dispersão exige commit simples, textual e sem submódulos.')
    churn = np.array([a[2] + a[3] for a in arquivos], dtype=float)
    total, nf = churn.sum(), len(arquivos)
    proporcoes = churn[churn > 0] / total if total else np.array([])
    entropia = float(-(proporcoes * np.log2(proporcoes)).sum())
    removidas = sum(a[3] for a in arquivos)
    return {
        'n_entropia': entropia,
        'n_entropia_normalizada': entropia / np.log2(nf) if nf > 1 else 0.0,
        'n_fracao_churn_max': float(churn.max() / total) if total else 0.0,
        'n_coef_variacao_churn': float(churn.std() / churn.mean()) if total else 0.0,
        'n_subsistemas': len({a[1].split('/')[0] if '/' in a[1] else '' for a in arquivos}),
        'n_profundidade_media': float(np.mean([a[1].count('/') for a in arquivos])),
        'n_fracao_ld_teste': sum(a[3] for a in arquivos if eh_teste(a[1])) / removidas if removidas else 0.0,
        'n_ld_max_arquivo': max(a[3] for a in arquivos),
    }


class EngenhariaPistas(TransformerMixin, BaseEstimator):
    """Transformação determinística: não estima estatísticas nem recebe rótulos."""
    def __init__(self, variante='razoes'):
        self.variante = variante

    def fit(self, X, y=None):
        esperadas = colunas_entrada(self.variante)
        if not isinstance(X, pd.DataFrame) or list(X.columns) != esperadas:
            raise ValueError('Entradas ou ordem diferentes do esquema da engenharia de pistas.')
        self.feature_names_in_ = np.array(esperadas, dtype=object)
        self.n_features_in_ = len(esperadas)
        return self

    def transform(self, X):
        check_is_fitted(self, 'feature_names_in_')
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Entradas diferentes das usadas no treinamento.')
        x = X.astype(float).copy()
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError('Pistas ausentes ou não finitas.')
        if self.variante == 'originais':
            return x
        churn = x.la + x.ld
        producao = x.o_la_producao + x.o_ld_producao
        # Denominador zero significa ausência; a razão correspondente fica em zero.
        def razao(a, b):
            return a.div(b.replace(0, np.nan)).fillna(0)
        x['d_churn'] = churn
        x['d_churn_por_arquivo'] = razao(churn, x.nf)
        x['d_fracao_java'] = razao(x.o_n_java, x.nf)
        x['d_fracao_testes'] = razao(x.o_n_teste, x.nf)
        x['d_fracao_producao'] = razao(x.o_n_producao, x.nf)
        x['d_fracao_churn_producao'] = razao(producao, churn)
        x['d_churn_por_arquivo_producao'] = razao(producao, x.o_n_producao)
        x['d_concentracao_adicoes'] = razao(x.o_max_la_arquivo, x.la)
        x['d_arquivos_por_pasta'] = razao(x.nf, x.o_n_pastas)
        x['d_experiencia_autor_por_arquivo'] = razao(x.w_autor_nos_arquivos, x.nf)
        x['d_mudancas_por_autor'] = x.l_arq_mudancas_media / (1 + x.l_arq_autores_media)
        x['d_atividade_autor_por_dia'] = x.w_autor_commits / (1 + x.w_autor_dias_desde_primeiro.clip(lower=0))
        x['d_fracao_atividade_recente'] = x.w_autor_commits_7d / (1 + x.w_autor_commits)
        x['d_churn_por_experiencia'] = churn / (1 + x.w_autor_commits)
        x['d_mudancas_por_dia'] = x.l_arq_mudancas_media / (1 + x.l_arq_idade_media_d.clip(lower=0))
        x['d_palavras_por_linha'] = razao(x.m_palavras, x.m_linhas)
        return x

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return np.array(list(self.feature_names_in_) + (list(DERIVADAS) if self.variante != 'originais' else []), dtype=object)
