"""Testes de separação temporal, pesos por fit e paridade da revisão."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dados import carregar_dividir
from modelo import ROOT, FEATURES, carregar_modelo, pasta_artefatos, prever
from revisar_balanceamento import (RESULTADOS, FIM_PRINCIPAL, ajustar, dividir_limiar,
                                   escolher_limiar, separar_periodos)


class Balanceamento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.treino, cls.teste, cls.folds, _ = carregar_dividir()

    def test_periodos_mesmo_treino_sem_2019_no_principal(self):
        principal, posterior = separar_periodos(self.teste)
        self.assertTrue((principal.data < FIM_PRINCIPAL).all())
        self.assertTrue((posterior.data >= FIM_PRINCIPAL).all())
        self.assertEqual(len(principal) + len(posterior), len(self.teste))
        self.assertFalse(set(principal.commit_id) & set(posterior.commit_id))
        salvas = pd.read_csv(RESULTADOS / 'particoes.csv')
        self.assertEqual(set(salvas[salvas.conjunto == 'treino'].commit_id), set(self.treino.commit_id))
        previsoes = pd.read_csv(RESULTADOS / 'previsoes_teste.csv')
        self.assertEqual(set(previsoes.commit_id), set(principal.commit_id))

    def test_limiar_so_usa_passado_interno(self):
        auditoria = pd.read_json(RESULTADOS / 'folds.json')
        for it, iv in self.folds:
            anterior, interno = dividir_limiar(self.treino.iloc[it])
            self.assertLess(anterior.data.max(), interno.data.min())
            self.assertLess(interno.data.max(), self.treino.iloc[iv].data.min())
            self.assertFalse(set(anterior.committer_date_git) & set(interno.committer_date_git))
        for r in auditoria[auditoria.ajustar_limiar].itertuples():
            self.assertLess(pd.Timestamp(r.fim_ajuste_interno), pd.Timestamp(r.inicio_validacao_interna))
            self.assertLess(pd.Timestamp(r.fim_validacao_interna), pd.Timestamp(r.inicio_validacao))

    def test_pesos_recalculados_em_cada_fit(self):
        chamadas = []
        class Estimador:
            def fit(self, x, y):
                chamadas.append((x.index.tolist(), y.copy()))
                return self
            def predict_proba(self, x):
                return np.tile([0.6, 0.4], (len(x), 1))
        trecho = self.treino.iloc[self.folds[0][0]]
        with patch('revisar_balanceamento.construir', side_effect=lambda *a: Estimador()) as fabrica:
            _, _, registro = ajustar(trecho, {}, 'balanced', True)
        self.assertEqual(len(chamadas), 2)
        anterior, _ = dividir_limiar(trecho)
        self.assertEqual(chamadas[0][0], anterior.index.tolist())
        self.assertEqual(chamadas[1][0], trecho.index.tolist())
        for call in fabrica.call_args_list:
            self.assertEqual(call.args[1]['class_weight'], 'balanced')
        for classe, n in trecho.buggy.value_counts().items():
            self.assertAlmostEqual(registro['pesos_fit'][classe], len(trecho)/(2*n))

    def test_otimizacao_limiar_e_empates(self):
        limiar, _ = escolher_limiar([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(limiar, 0.3)
        limiar, _ = escolher_limiar([0, 1], [0.05, 0.95])
        self.assertEqual(limiar, 0.5)

    def test_inferencia_respeita_limiar_salvo_e_indices(self):
        exemplos = pd.read_csv(pasta_artefatos() / 'exemplos.csv').iloc[:2].copy()
        exemplos.index = [51, 78]
        class ModeloSimulado:
            classes_ = np.array([0, 1])
            limiar_decisao_ = 0.3
            def predict_proba(self, x):
                return np.tile([0.6, 0.4], (len(x), 1))
        p = prever(ModeloSimulado(), exemplos)
        self.assertEqual(p.index.tolist(), [51, 78])
        self.assertEqual(p.previsao.tolist(), [1, 1])

    def test_metadados_e_pipeline_exigem_mesmo_limiar(self):
        pasta = pasta_artefatos()
        meta = json.loads((pasta / 'metadados.json').read_text(encoding='utf-8'))
        meta['limiar'] = 0.99
        with tempfile.TemporaryDirectory() as diretorio:
            destino = Path(diretorio)
            (destino / 'metadados.json').write_text(json.dumps(meta), encoding='utf-8')
            (destino / 'pipeline.joblib').write_bytes((pasta / 'pipeline.joblib').read_bytes())
            with patch('modelo.pasta_artefatos', return_value=destino):
                with self.assertRaisesRegex(ValueError, 'Limiar'):
                    carregar_modelo()

    def test_original_preservado_e_comparacao_pareada(self):
        pipe, _ = carregar_modelo(atual=False)
        self.assertEqual(getattr(pipe, 'limiar_decisao_', 0.5), 0.5)
        previsoes = pd.read_csv(RESULTADOS / 'previsoes_teste.csv')
        anterior = prever(pipe, previsoes[FEATURES])
        np.testing.assert_allclose(anterior.score, previsoes.score_original, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(anterior.previsao, previsoes.previsao_original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
