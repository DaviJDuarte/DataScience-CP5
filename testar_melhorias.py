"""Casos de borda das novas pistas e isolamento temporal da busca, sem rede."""
import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from esquema import ROOT, FEATURES, NEGATIVAS
from novas_pistas import EngenhariaPistas, dispersao_commit, colunas_entrada, ESTATICAS
from extrair_dispersao import ler_alvos
from extracao import ExtracaoIncompativel
from buscar_melhorias import ajustar_ampliado
from dados import carregar_dividir
from modelo import carregar_modelo, prever, entradas


class NovasPistas(unittest.TestCase):
    def test_entropia_uniforme_concentrada_e_diff_sem_linhas(self):
        c = {'pais': ['a'*40], 'submodulo': False, 'arquivos': [
            ('src/A.java', 'src/A.java', 2, 0, False), ('test/A.java', 'test/A.java', 0, 2, False)]}
        r = dispersao_commit(c)
        self.assertEqual(r['n_entropia'], 1)
        self.assertEqual(r['n_entropia_normalizada'], 1)
        self.assertEqual(r['n_fracao_churn_max'], 0.5)
        self.assertEqual(r['n_fracao_ld_teste'], 1)
        self.assertEqual(r['n_subsistemas'], 2)
        c['arquivos'][1] = ('test/A.java', 'test/B.java', 0, 0, False)
        r = dispersao_commit(c)
        self.assertEqual(r['n_entropia'], 0)
        self.assertEqual(r['n_fracao_churn_max'], 1)
        c['arquivos'] = [('A.java', 'B.java', 0, 0, False)]
        r = dispersao_commit(c)
        self.assertTrue(np.isfinite(list(r.values())).all())
        self.assertEqual(r['n_entropia_normalizada'], 0)

    def test_diff_incompativel_recusado(self):
        for c in [
            {'pais': ['a'*40], 'submodulo': False, 'arquivos': []},
            {'pais': ['a'*40, 'b'*40], 'submodulo': False, 'arquivos': [('a', 'a', 1, 1, False)]},
            {'pais': ['a'*40], 'submodulo': True, 'arquivos': [('a', 'a', 1, 1, False)]},
            {'pais': ['a'*40], 'submodulo': False, 'arquivos': [('a', 'a', 0, 0, True)]},
        ]:
            with self.assertRaises(ExtracaoIncompativel):
                dispersao_commit(c)

    def test_razoes_sem_historico_zero_e_indices(self):
        x = pd.DataFrame([{f: (-1 if f in NEGATIVAS - {'q_fuso_h'} else 0) for f in FEATURES}], index=[57])
        x['nf'], x['la'], x['o_n_pastas'] = 2, 10, 1
        t = EngenhariaPistas().fit(x)
        r = t.transform(x)
        self.assertEqual(r.index.tolist(), [57])
        self.assertEqual(r.loc[57, 'd_churn_por_arquivo'], 5)
        self.assertEqual(r.loc[57, 'd_churn_por_experiencia'], 10)
        self.assertEqual(r.loc[57, 'd_fracao_churn_producao'], 0)
        self.assertTrue(np.isfinite(r.to_numpy()).all())
        self.assertEqual(r.columns.tolist(), t.get_feature_names_out().tolist())
        with self.assertRaisesRegex(ValueError, 'Entradas'):
            t.transform(x[x.columns[::-1]])
        with self.assertRaisesRegex(ValueError, 'Entradas'):
            EngenhariaPistas().fit(x.assign(buggy=1))
        y = x.assign(**{f: 0 for f in ESTATICAS})
        self.assertEqual(EngenhariaPistas('dispersao').fit_transform(y).shape[1], 72)
        self.assertEqual(len(colunas_entrada('dispersao')), 56)

    def test_parser_git_alvo_unico_renomeacao_e_futuro(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def run(*args):
                return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                    env={**os.environ, 'GIT_AUTHOR_NAME': 'Teste', 'GIT_AUTHOR_EMAIL': 't@example.invalid',
                         'GIT_COMMITTER_NAME': 'Teste', 'GIT_COMMITTER_EMAIL': 't@example.invalid'}).stdout.decode().strip()
            run('init', '-b', 'main')
            arq = repo / 'nome com espaço.java'
            arq.write_text('class A {}\n', encoding='utf-8')
            run('add', '.'); run('commit', '-m', 'inicio')
            run('mv', arq.name, 'ação.java'); run('commit', '-m', 'renomear')
            alvo = run('rev-parse', 'HEAD')
            antes = list(ler_alvos(repo / '.git', [alvo]))
            (repo / 'ação.java').write_text('class A { int n; }\n', encoding='utf-8')
            run('add', '.'); run('commit', '-m', 'alteração futura')
            depois = list(ler_alvos(repo / '.git', [alvo]))
            self.assertEqual(antes, depois)
            self.assertEqual(len(antes), 1)
            self.assertEqual(antes[0]['sha'], alvo)
            self.assertEqual(antes[0]['arquivos'][0][:2], ('nome com espaço.java', 'ação.java'))
            self.assertEqual(dispersao_commit(antes[0])['n_entropia'], 0)

    def test_limiar_e_pesos_nao_veem_validacao_externa(self):
        frame = pd.DataFrame([{f: 1 for f in FEATURES} for _ in range(20)])
        frame['buggy'] = [0, 0, 0, 1] * 5
        frame['committer_date_git'] = np.arange(20)
        frame['data'] = pd.date_range('2010-01-01', periods=20, tz='UTC')
        fits, predicoes = [], []
        class Simulado:
            def fit(self, x, y, **kwargs):
                fits.append((x.index.tolist(), y.to_numpy(), kwargs['modelo__sample_weight']))
            def predict_proba(self, x):
                predicoes.append(x.index.tolist())
                return np.tile([0.6, 0.4], (len(x), 1))
        config = {'modelo': 'XGBoost', 'parametros': {}, 'pistas': 'originais', 'pesos': 'balanced'}
        with patch('buscar_melhorias.construir_ampliado', side_effect=lambda _: Simulado()):
            pipe, _, a = ajustar_ampliado(frame, config)
        self.assertEqual(fits[0][0], list(range(16)))
        self.assertEqual(predicoes, [list(range(16, 20))])
        self.assertEqual(fits[1][0], list(range(20)))
        for _, y, w in fits:
            self.assertAlmostEqual(w[y == 0].sum(), w[y == 1].sum())
        self.assertLess(a['fim_ajuste_interno'], a['inicio_validacao_interna'])
        self.assertEqual(pipe.limiar_decisao_, 0.4)


class ExperimentoConcluido(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.treino, cls.teste, cls.folds, _ = carregar_dividir()
        cls.pasta = ROOT / 'resultados/melhorias'

    def test_candidatos_mesmos_commits_e_temporalidade(self):
        arquivos = list((self.pasta / 'candidatos').glob('*.json'))
        self.assertEqual(len(arquivos), 42)
        ids = set(self.treino.iloc[np.concatenate([v for _, v in self.folds])].commit_id)
        for path in arquivos:
            r = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(r['folds']), 3)
            for f in r['folds']:
                self.assertLess(f['fim_ajuste_interno'], f['inicio_validacao_interna'])
                self.assertLess(f['fim_treino'], f['inicio_validacao'])
                for contagem, peso in zip(f['contagens_por_fit'], f['pesos_por_fit']):
                    if r['config']['pesos'] == 'balanced':
                        self.assertAlmostEqual(contagem['0'] * peso['0'], contagem['1'] * peso['1'])
            oof = pd.read_parquet(path.with_suffix('.parquet'))
            self.assertTrue(oof.commit_id.is_unique)
            self.assertEqual(set(oof.commit_id), ids)
        anterior = json.loads((self.pasta / 'candidatos/anterior_originais.json').read_text(encoding='utf-8'))
        self.assertAlmostEqual(anterior['f1_validacao'], 0.6060961531312352, places=12)

    def test_selecao_corresponde_a_regra_e_holdout_pareado(self):
        escolha = json.loads((self.pasta / 'selecao_validacao.json').read_text(encoding='utf-8'))
        comparacao = pd.read_csv(self.pasta / 'comparacao.csv')
        self.assertEqual(escolha['config']['id'], comparacao.iloc[0]['id'])
        self.assertAlmostEqual(escolha['f1_validacao'], comparacao.f1_validacao.max(), places=12)
        self.assertFalse(escolha['holdout_usado_na_selecao'])
        pred = pd.read_parquet(self.pasta / 'previsoes_teste.parquet')
        esperado = self.teste[self.teste.data < pd.Timestamp('2019-01-01', tz='UTC')]
        self.assertEqual(set(pred.commit_id), set(esperado.commit_id))
        self.assertFalse(set(pred.commit_id) & set(self.treino.commit_id))
        pipe, _ = carregar_modelo()
        nova = prever(pipe, pred)
        np.testing.assert_allclose(nova.score, pred.score, rtol=0, atol=1e-12)
        anterior, _ = carregar_modelo(pasta=ROOT / 'artefatos/revisao_balanceamento')
        antiga = prever(anterior, pred)
        np.testing.assert_allclose(antiga.score, pred.score_anterior, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(antiga.previsao, pred.previsao_anterior)

    def test_entrada_adicional_obrigatoria_quando_modelo_exige(self):
        x = pd.DataFrame([{f: 0 for f in FEATURES}])
        x['nf'] = 1
        with self.assertRaisesRegex(ValueError, 'incompletos'):
            entradas(x, colunas_entrada('dispersao'))
        x = x.assign(**{f: 0 for f in ESTATICAS})
        x['n_subsistemas'] = 1
        x['n_entropia_normalizada'] = 2
        with self.assertRaisesRegex(ValueError, 'normalizada'):
            entradas(x, colunas_entrada('dispersao'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
