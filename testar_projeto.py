"""Verificações de protocolo, paridade do app e falhas de integração."""
import copy
import json
import unittest
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
import requests
from streamlit.testing.v1 import AppTest
from dados import carregar_dividir, salvar_json
from modelo import ROOT, FEATURES, carregar_modelo, prever, pasta_artefatos
from github_api import repositorio, extrair_commit, consultar, ErroGitHub
from novas_pistas import colunas_entrada


class Integridade(unittest.TestCase):
    def test_particoes_e_cobertura(self):
        with patch('dados.extrair_base',side_effect=AssertionError('O pacote deve funcionar sem clones Git')):
            with patch('requests.get',side_effect=AssertionError('O CSV do ZIP deve permitir execução offline')):
                treino,teste,folds,info=carregar_dividir()
        self.assertEqual(info['n_excluidos']+len(treino)+len(teste),106674)
        self.assertFalse(set(treino.commit_id)&set(teste.commit_id))
        self.assertLess(treino.data.max(),teste.data.min())
        for tr,va in folds:
            self.assertLess(treino.iloc[tr].data.max(),treino.iloc[va].data.min())
            self.assertFalse(set(tr)&set(va))
        for frame in [treino,teste]:
            for f in ['la','ld','nf']: self.assertTrue(frame[f].eq(frame[f+'_csv']).all())

    def test_pipeline_e_exemplos(self):
        pipe,meta=carregar_modelo()
        exemplos=pd.read_csv(pasta_artefatos()/'exemplos.csv')
        self.assertGreaterEqual(len(exemplos),5)
        self.assertEqual(set(exemplos.caso),{'Verdadeiro positivo','Verdadeiro negativo','Falso positivo','Falso negativo'})
        pred=prever(pipe,exemplos)
        np.testing.assert_array_equal(pred.previsao,exemplos.previsao)
        np.testing.assert_allclose(pred.score,exemplos.score,rtol=0,atol=1e-12)
        self.assertEqual(meta['variaveis'],colunas_entrada(meta.get('pistas', 'originais')))
        self.assertEqual(pipe.named_steps['imputacao'].n_features_in_, len(meta.get('variaveis_modelo', FEATURES)))

    def test_buscas_concluidas(self):
        tabela=pd.read_csv(ROOT/'resultados/comparacao.csv')
        self.assertEqual(len(tabela),9)
        for nome in ['random_forest','xgboost','lightgbm']:
            trials=pd.read_csv(ROOT/f'resultados/optuna_{nome}_trials.csv')
            self.assertEqual(trials.state.eq('COMPLETE').sum(),20)
            grid=pd.read_csv(ROOT/f'resultados/grid_{nome}_candidatos.csv')
            self.assertEqual(len(grid),8)
        final=json.loads((ROOT/'resultados/teste_final.json').read_text(encoding='utf-8'))
        self.assertEqual(final['avaliacoes_nesta_versao'],1)
        self.assertIn('não é teste novo',final['ressalva'])

    def test_app_offline_e_retorno_apos_falha_api(self):
        exemplos=pd.read_csv(pasta_artefatos()/'exemplos.csv')
        with patch('requests.get',side_effect=requests.ConnectionError('offline')) as rede:
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
            self.assertEqual(len(app.exception),0)
            self.assertFalse(rede.called)
            for i,linha in exemplos.iterrows():
                app.selectbox[0].select(i).run()
                self.assertEqual(app.metric[0].value,f'{linha.score:.6f}')
                self.assertTrue(any('Paridade confirmada' in x.value for x in app.success))
            app.checkbox[0].check().run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.metric[0].value,f'{exemplos.iloc[-1].score:.6f}')
            app.radio[0].set_value('GitHub').run()
            app.text_input[0].set_value('https://github.com/apache/camel')
            app.button[0].click().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('conexão' in x.value for x in app.error))
            app.radio[0].set_value('Demonstração local').run()
            app.checkbox[0].uncheck().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('Paridade confirmada' in x.value for x in app.success))


class GitHub(unittest.TestCase):
    def setUp(self):
        self.commit={'sha':'a'*40,'parents':[{'sha':'b'*40}],
          'html_url':'https://github.com/apache/camel/commit/'+'a'*40,
          'stats':{'additions':1,'deletions':1},
          'files':[{'filename':'a.java','status':'modified','additions':1,'deletions':1,'patch':'@@ -1 +1 @@\n-antigo\n+novo'}]}

    def test_urls(self):
        self.assertEqual(repositorio('https://github.com/apache/camel.git'),'apache/camel')
        for url in ['http://github.com/a/b','https://github.com.evil/a/b','https://github.com/a/b/tree/main',
                    'https://usuario@github.com/a/b','https://github.com/a/..','file:///a/b']:
            with self.assertRaises(ErroGitHub): repositorio(url)

    def test_paginacao(self):
        a,b=copy.deepcopy(self.commit),copy.deepcopy(self.commit)
        for c in [a,b]: c['stats']={'additions':2,'deletions':2}
        b['files'][0]['filename']='b.java'
        with patch('github_api.consultar',side_effect=[(a,True),(b,False)]): valores,_=extrair_commit('apache/camel','a'*40)
        self.assertEqual(valores,{'la':2,'ld':2,'nf':2})

    def test_recusas(self):
        casos=[]
        c=copy.deepcopy(self.commit); c['parents']*=2; casos.append(c)
        c=copy.deepcopy(self.commit); del c['files'][0]['patch']; casos.append(c)
        c=copy.deepcopy(self.commit); c['stats']['additions']=10; casos.append(c)
        c=copy.deepcopy(self.commit); c['files'][0]['patch']='@@ -1 +1 @@\n-antigo'; casos.append(c)
        c=copy.deepcopy(self.commit); c['files']*=3000; casos.append(c)
        for c in casos:
            with patch('github_api.consultar',return_value=(c,False)):
                with self.assertRaises(ErroGitHub): extrair_commit('apache/camel','a'*40)

    def test_erros_api(self):
        for status,trecho in [(403,'Limite'),(429,'Limite'),(404,'indisponível'),(409,'vazio'),(401,'Token'),(500,'indisponível')]:
            with patch('requests.get',return_value=Mock(status_code=status,ok=False)):
                with self.assertRaisesRegex(ErroGitHub,trecho): consultar('apache/camel')


if __name__=='__main__':
    resultado=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    salvar_json(ROOT/'resultados/verificacoes.json',{'testes':resultado.testsRun,'falhas':len(resultado.failures),
        'erros':len(resultado.errors),'paridade_interface':resultado.wasSuccessful(),'modo_local_sem_internet':resultado.wasSuccessful()})
    raise SystemExit(0 if resultado.wasSuccessful() else 1)
