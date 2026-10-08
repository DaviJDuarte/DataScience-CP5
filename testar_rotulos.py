"""Proveniência, limites das flags e integridade dos resultados, sem rede nem clones."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import auditar_rotulos as audit
from dados import CSV_SHA256
from esquema import ROOT, FEATURES
from novas_pistas import colunas_entrada


class RegrasDaAuditoria(unittest.TestCase):
    def test_chave_nao_confunde_numero_de_pr_e_aceita_pontuacao(self):
        for texto in ['HBASE-3755: ajuste', 'hbase=3755 ajuste', 'HBASE:3755 ajuste', 'HBASE 3755 ajuste']:
            self.assertTrue(audit.conferir_chave('HBASE-3755', texto)['chave_flexivel_na_mensagem'])
        for texto in ['HBASE-37550', 'XHBASE-3755', 'HBASE-3755x', 'Closes apache/hbase#3755']:
            self.assertFalse(audit.conferir_chave('HBASE-3755', texto)['chave_flexivel_na_mensagem'])
        flags = audit.conferir_chave('HIVE-2107', 'HIVE-24886: ajuste\nCloses apache/hive#2107')
        self.assertTrue(flags['possivel_colisao_issue_pr'])
        self.assertEqual(flags['outras_chaves_do_projeto'], 'HIVE-24886')
        self.assertFalse(audit.conferir_chave('HIVE-2107', 'HIVE-2107 HIVE-24886 #2107')['possivel_colisao_issue_pr'])

    def test_familia_agrupa_sem_descartar_ou_mudar_original(self):
        s = pd.Series(['apache/hadoop', 'apache/hadoop-hdfs', 'apache/hadoop-mapreduce', 'apache/hive'], index=[8, 3, 9, 1])
        antes = s.copy()
        agrupado = audit.familia(s)
        pd.testing.assert_series_equal(s, antes)
        self.assertEqual(agrupado.index.tolist(), s.index.tolist())
        self.assertEqual(agrupado.nunique(), 2)
        self.assertEqual(agrupado.iloc[:3].unique().tolist(), ['apache/hadoop (familia)'])

    def test_reconstrucao_datas_estritas_deduplicacao_e_ast(self):
        issues = pd.DataFrame({'Issue key': ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6'],
            'Created': ['02/Jan/70 00:00'] * 6,
            'Resolution': ['Open'] + ['Fixed'] * 5, 'Status': ['Open'] + ['Closed'] * 5})
        found = pd.DataFrame({'issue_key': issues['Issue key'], 'commit_id': ['f1', 'f2', 'f3', 'f4', 'f5', 'f1']})
        links = pd.DataFrame({'fix_hash': ['f1', 'f1', 'f2', 'f3', 'f4', 'f5'],
            'bug_hash': ['b1', 'b1', 'b1', 'b2', 'b3', 'b4'],
            'bug_date': [0, 0, 0, 86400, 0, 0], 'fix_date': [172800, 172800, 172800, 172800, 0, 172800]})
        etapas, ev, _, limites = audit.reconstruir(issues, found, links, {'b1'})
        self.assertEqual(len(etapas['ligado_issue']), 5)
        self.assertEqual(set(etapas['apos_temporal'].bug_hash), {'b1', 'b4'})
        self.assertEqual(set(etapas['apos_ast'].bug_hash), {'b1'})
        self.assertEqual(limites, {'max_fixcount': 1, 'max_bugcount': 2})
        self.assertEqual(set(ev.loc[ev.fix_hash.eq('f1'), 'issue_key']), {'A-1', 'A-6'})
        self.assertTrue(ev.relato_fixed.any())

    def test_alerta_preserva_rotulo_e_evidencia_alternativa(self):
        p = pd.DataFrame({'commit_id': ['a', 'b', 'c'], 'buggy': [True, True, False],
                          'positivo_sem_relato_fixed': [False, False, False]})
        ev = pd.DataFrame({'bug_hash': ['a', 'a', 'b'], 'fix_hash': ['f1', 'f2', 'f1'],
                           'issue_key': ['A-1', 'A-2', 'A-1'], 'relato_fixed': [True] * 3})
        msg = pd.DataFrame({'fix_hash': ['f1', 'f2'], 'issue_key': ['A-1', 'A-2'],
            'objeto_disponivel': [True, True], 'chave_flexivel_na_mensagem': [False, True],
            'possivel_colisao_issue_pr': [True, False], 'correcao_antes_relato_mais_1d': [False, False]})
        novo, todas = audit.anexar_alertas(p, ev, msg)
        pd.testing.assert_series_equal(novo.buggy, p.buggy)
        self.assertEqual(len(todas), 3)
        self.assertEqual(novo.positivo_para_revisao.tolist(), [True, True, False])
        self.assertEqual(novo.positivo_so_vinculos_sem_chave.tolist(), [False, True, False])
        self.assertTrue(novo.loc[0, 'alguma_evidencia_sem_alerta'])


class AuditoriaConcluida(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resumo = audit.carregar_auditoria()
        cls.p = pd.read_parquet(audit.PASTA / 'proveniencia.parquet')
        cls.ev = pd.read_parquet(audit.PASTA / 'evidencias_positivos.parquet')

    def test_reproducao_e_integridade_da_base_e_modelo(self):
        original = pd.read_csv(ROOT / 'data/apachejit_total.csv').set_index('commit_id')
        p = self.p.set_index('commit_id').loc[original.index]
        self.assertEqual(hashlib.sha256((ROOT / 'data/apachejit_total.csv').read_bytes()).hexdigest(), CSV_SHA256)
        for coluna in ['buggy', 'fix']:
            pd.testing.assert_series_equal(p[coluna], original[coluna])
            self.assertTrue(p[coluna].eq(p[coluna + '_reconstruido']).all())
        self.assertTrue((p.buggy & p.fix).any())  # Um commit pode corrigir e introduzir bugs.
        self.assertEqual(set(self.ev.bug_hash), set(p.index[p.buggy]))
        self.assertEqual(self.resumo['rotulos_alterados'], 0)
        self.assertFalse(self.resumo['modelo_alterado'])
        for arquivo in audit.PASTA.iterdir():
            if arquivo.suffix in {'.csv', '.json'}:
                self.assertNotIn(b'\r\n', arquivo.read_bytes(), arquivo.name)

    def test_agrupamento_preserva_todos_os_erros_oof(self):
        oof = pd.read_parquet(ROOT / 'resultados/melhorias/previsoes_validacao.parquet')
        por_familia = pd.read_csv(audit.PASTA / 'erros_por_familia.csv')
        self.assertEqual(por_familia.commits.sum(), len(oof))
        for coluna, real, pred in [('falsos_positivos', 0, 1), ('falsos_negativos', 1, 0),
                                  ('verdadeiros_positivos', 1, 1), ('verdadeiros_negativos', 0, 0)]:
            self.assertEqual(por_familia[coluna].sum(), ((oof.buggy == real) & (oof.previsao == pred)).sum())
        familia_hadoop = por_familia[por_familia.project_familia.eq('apache/hadoop (familia)')].iloc[0]
        self.assertGreater(familia_hadoop.taxa_buggy, 0)
        self.assertEqual(self.resumo['familia_hadoop']['commits'], 16192)

    def test_datas_futuras_conferidas_e_fora_das_entradas(self):
        p = self.p[self.p.particao.eq('treino')]
        registro = self.resumo['disponibilidade_temporal'][0]
        esperado = (p.buggy & p.primeira_correcao_vinculada.gt(pd.Timestamp(registro['fim_treino']))).sum()
        self.assertEqual(esperado, registro['primeira_correcao_posterior'])
        self.assertEqual(esperado, 3099)
        self.assertEqual(registro['positivos_treino'], p.buggy.sum())
        self.assertTrue(self.ev.fix_date.eq(self.ev.fix_date_git).all())
        self.assertFalse(set(self.p.columns) & set(FEATURES))
        self.assertFalse(set(self.p.columns) & set(colunas_entrada('dispersao')))

    def test_manifesto_rejeita_resultado_modificado(self):
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp)
            for arq in audit.PASTA.iterdir():
                if arq.is_file():
                    (destino / arq.name).write_bytes(arq.read_bytes())
            (destino / 'resumo.json').write_text('{}', encoding='utf-8')
            with patch.object(audit, 'PASTA', destino), self.assertRaisesRegex(ValueError, 'Resultado da auditoria'):
                audit.carregar_auditoria()


if __name__ == '__main__':
    unittest.main(verbosity=2)
