"""Testes da causalidade do histórico e do parser Git, sem rede."""
import tempfile
import unittest
import subprocess
import os
from pathlib import Path
from extracao import processar_commits, extrair, interpretar, eh_teste, ExtracaoIncompativel
from esquema import FEATURES
from modelo import entradas


def commit(letra,pai=None,msg="alteração",nome="src/App.java",antes=None):
    return {"sha":letra*40,"pais":[pai*40] if pai else [],"autor":"autor@exemplo.invalid",
            "committer":"autor@exemplo.invalid","at":100000,"ct":100000,"fuso":0,"mensagem":msg,
            "arquivos":[(antes or nome,nome,1,0,False)],"submodulo":False}


class Historico(unittest.TestCase):
    def test_branch_irma_futuro_e_proprio_commit_nao_entram(self):
        commits=[commit("a"),commit("b","a","fix bug"),commit("c","a","fix atual"),commit("d","c","fix futuro")]
        r={x['commit_id']:x for x in processar_commits(commits,["b"*40,"c"*40,"d"*40])}
        self.assertEqual(r["c"*40]["l_arq_mudancas_max"],1)
        self.assertEqual(r["c"*40]["l_arq_correcoes_max"],0)
        self.assertEqual(r["c"*40]["w_autor_commits"],1)
        self.assertEqual(r["d"*40]["l_arq_correcoes_max"],1)
        self.assertEqual(set(FEATURES)-set(r['c'*40]),set())

    def test_mensagem_futura_nao_altera_passado(self):
        base=[commit("a"),commit("b","a"),commit("c","b")]
        antes=processar_commits(base,['b'*40])[0]
        base[-1]['mensagem']='fix bug regression hotfix'
        depois=processar_commits(base,['b'*40])[0]
        self.assertEqual(antes,depois)

    def test_renomeacao_preserva_historia(self):
        base=[commit('a',msg='fix bug'),commit('b','a',nome='src/Novo.java',antes='src/App.java'),
              commit('c','b',nome='src/Novo.java')]
        r=processar_commits(base,['c'*40])[0]
        self.assertEqual(r['l_arq_mudancas_max'],2)
        self.assertEqual(r['l_arq_correcoes_max'],1)
        self.assertEqual(r['w_autor_nos_arquivos'],2)

    def test_merge_contra_primeiro_pai(self):
        a,b,c=commit('a'),commit('b','a'),commit('c','a')
        m=commit('d','b'); m['pais'].append('c'*40)
        alvo=commit('e','d')
        r=processar_commits([a,b,c,m,alvo],['d'*40,'e'*40])
        self.assertIn('merge_ou_raiz',r[0]['motivo'])
        self.assertEqual(r[1]['l_arq_mudancas_max'],3)

    def test_pai_faltante_rejeitado(self):
        with self.assertRaises(ExtracaoIncompativel): processar_commits([commit('b','a')],['b'*40])

    def test_classificacao_arquivos(self):
        self.assertFalse(eh_teste('src/Credit.java'))
        self.assertTrue(eh_teste('src/ConnectionIT.java'))
        self.assertTrue(eh_teste('src/test/resources/input.json'))
        self.assertTrue(eh_teste('TestExample.java'))

    def test_inferencia_nao_aceita_historico_ausente(self):
        with self.assertRaises(ValueError): entradas({'la':1,'ld':0,'nf':1})

    def test_git_real_parser_e_ancestralidade(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)/'repo'; repo.mkdir()
            def run(*args):
                return subprocess.run(['git','-C',str(repo),*args],check=True,capture_output=True,
                    env={**os.environ,'GIT_AUTHOR_NAME':'Teste','GIT_AUTHOR_EMAIL':'teste@example.invalid',
                         'GIT_COMMITTER_NAME':'Teste','GIT_COMMITTER_EMAIL':'teste@example.invalid',
                         'GIT_AUTHOR_DATE':'2020-01-01T10:00:00+00:00','GIT_COMMITTER_DATE':'2020-01-01T10:00:00+00:00'}).stdout.decode().strip()
            run('init','-b','main'); arquivo=repo/'ação com espaço.java'
            arquivo.write_text('class A {}\n'); run('add','.'); run('commit','-m','raiz')
            run('branch','outra'); arquivo.write_text('class A { int n; }\n'); run('add','.'); run('commit','-m','alteração')
            run('commit','--amend','-m','alteração com controle \x1e no texto')
            alvo=run('rev-parse','HEAD'); run('checkout','outra')
            arquivo.write_text('class A { int bug; }\n'); run('add','.'); run('commit','-m','fix bug em outra branch')
            r=extrair(repo/'.git',[alvo])[0]
            self.assertEqual(r['la'],1); self.assertEqual(r['ld'],1); self.assertEqual(r['nf'],1)
            self.assertEqual(r['l_arq_correcoes_max'],0); self.assertEqual(r['l_arq_mudancas_max'],1)


if __name__=='__main__': unittest.main(verbosity=2)
