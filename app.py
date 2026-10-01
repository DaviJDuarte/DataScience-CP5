"""Modelo completo: exemplos offline e histórico Git compartilhado com o treinamento."""
import json
import os
import pandas as pd
import streamlit as st
from esquema import ROOT, FEATURES, DESCRICOES, GRUPOS
from modelo import carregar_modelo, prever
from github_api import repositorio, listar_commits, consultar, ErroGitHub
from extracao import ExtracaoIncompativel
from repositorios import caminho_repo, obter_repo, metricas_github

st.set_page_config(page_title="Risco de commits · modelo completo",page_icon="🔎",layout="centered")


@st.cache_resource
def modelo_cache(): return carregar_modelo()


@st.cache_data
def exemplos_cache(): return pd.read_csv(ROOT/"artefatos/exemplos.csv")


@st.cache_data(ttl=300,show_spinner=False)
def listar_cache(repo,pagina,token): return listar_commits(repo,pagina,token)


@st.cache_data(ttl=3600,show_spinner=False)
def extrair_cache(repo,sha,token): return metricas_github(repo,sha,token)


def token_opcional():
    try: return st.secrets.get("GITHUB_TOKEN",os.getenv("GITHUB_TOKEN",""))
    except FileNotFoundError: return os.getenv("GITHUB_TOKEN","")


def mostrar(pipeline,registro,real=None,esperado=None,link=None):
    resultado=prever(pipeline,pd.DataFrame([{f:registro[f] for f in FEATURES}])).iloc[0]
    score=float(resultado.score); classe=int(resultado.previsao)
    st.metric("Score de risco",f"{score:.6f}")
    st.write("**Priorizar revisão (1)**" if classe else "**Menor prioridade pelo modelo (0)**")
    st.caption("Limiar fixo: 0,5. O score não é uma probabilidade calibrada; classe 0 não garante código correto.")
    a,b,c=st.columns(3)
    a.metric("Linhas adicionadas",f"{registro['la']:,.0f}")
    b.metric("Linhas removidas",f"{registro['ld']:,.0f}")
    c.metric("Arquivos afetados",f"{registro['nf']:,.0f}")
    st.write(f"O arquivo com mais histórico foi alterado **{registro['l_arq_mudancas_max']:.0f} vezes** antes. "
             f"O máximo de mensagens anteriores com indício de correção foi **{registro['l_arq_correcoes_max']:.0f}**.")
    st.caption("Essas contagens contextualizam as entradas; não explicam causalmente a previsão. "
               "Indício de correção vem da mensagem, não de defeito confirmado.")
    if real is not None:
        st.write(f"Rótulo real: **{int(real)}** · **{'Acerto' if classe==int(real) else 'Erro'}**")
    if esperado is not None:
        if classe==int(esperado['previsao']) and abs(score-float(esperado['score']))<1e-12:
            st.success("Paridade confirmada: classe e score coincidem com o notebook.")
        else: st.error("Divergência em relação ao resultado salvo.")
    if link: st.link_button("Abrir commit no GitHub",link)
    with st.expander("Ver as 48 entradas utilizadas"):
        tabela=pd.DataFrame([{"Grupo":g,"Variável":f,"Descrição":desc,"Valor":float(registro[f])}
                            for g,grupo in GRUPOS.items() for f,desc in grupo.items()])
        st.dataframe(tabela,hide_index=True,width="stretch")
    path=ROOT/"artefatos/faixas.csv"
    if path.exists():
        faixas=pd.read_csv(path)
        with st.expander("Como interpretar este score"):
            atual=faixas[faixas.score_minimo<=score].iloc[-1]
            st.write(f"Faixa **{atual.faixa}**: {atual.taxa_buggy:.1%} de positivos observados na validação "
                     f"({int(atual.commits):,} commits nessa faixa).")
            st.caption("Taxa observada em outro período, não probabilidade individual nem garantia em novos projetos.")
            st.dataframe(faixas.rename(columns={"faixa":"Faixa","score_minimo":"Score mínimo",
                "commits":"Commits","taxa_buggy":"Fração de positivos","taxa_geral":"Fração geral"}),hide_index=True)


st.title("Risco de commits")
st.write("Tamanho da alteração, tipos de arquivo e histórico anterior de autores e arquivos.")
try: pipeline,meta=modelo_cache()
except (ValueError,FileNotFoundError) as exc:
    st.error(f"Modelo indisponível: {exc}. Execute treinamento.py."); st.stop()

modo=st.radio("Modo de uso",["Demonstração local","GitHub"],horizontal=True)
if modo=="Demonstração local":
    exemplos=exemplos_cache()
    indice=st.selectbox("Exemplo",exemplos.index.tolist(),format_func=lambda i:
                       f"{i+1} · {exemplos.loc[i,'caso']} · {exemplos.loc[i,'commit_id'][:8]}")
    linha=exemplos.loc[indice]
    st.caption(f"{linha.project} · {linha.data}")
    editar=st.checkbox("Editar um cenário com todas as entradas")
    if editar:
        st.info("Este é um cenário hipotético baseado no exemplo. Os valores precisam continuar coerentes entre si; "
                "editar só o tamanho não recalcula o histórico.")
        with st.form("cenario"):
            tabela=pd.DataFrame({"Variável":FEATURES,"Descrição":[DESCRICOES[f] for f in FEATURES],
                                "Valor":[float(linha[f]) for f in FEATURES]})
            alterado=st.data_editor(tabela,disabled=["Variável","Descrição"],hide_index=True,width="stretch",height=350)
            enviar=st.form_submit_button("Estimar cenário")
        if enviar:
            try: mostrar(pipeline,dict(zip(alterado["Variável"],alterado["Valor"])))
            except ValueError as exc: st.error(str(exc))
    else: mostrar(pipeline,linha,real=linha.buggy,esperado=linha)
else:
    url=st.text_input("URL de repositório público",placeholder="https://github.com/apache/camel")
    pagina=st.number_input("Página de commits",min_value=1,value=1,step=1)
    st.caption("30 commits da branch padrão por página. A análise completa usa histórico Git em cache; "
               "a primeira preparação de um repositório grande pode levar minutos e ocupar espaço em disco.")
    if st.button("Listar commits",type="primary"):
        st.session_state.pop("consulta",None)
        try:
            repo=repositorio(url)
            commits,mais=listar_cache(repo,pagina,token_opcional())
            st.session_state.consulta=(url,pagina,repo,commits,mais)
        except ErroGitHub as exc:
            st.error(str(exc)); st.info("A demonstração local continua disponível sem internet.")
    consulta=st.session_state.get("consulta")
    if consulta and consulta[0]==url and consulta[1]==pagina:
        _,_,repo,commits,mais=consulta
        st.caption("Há mais commits na próxima página." if mais else "Última página disponível.")
        i=st.selectbox("Commit",range(len(commits)),format_func=lambda j:
                       f"{commits[j]['sha'][:8]} · {commits[j]['commit']['message'].splitlines()[0][:90]} · "
                       f"{(commits[j]['commit'].get('author') or {}).get('date','Sem data')}")
        if not caminho_repo(repo).exists():
            st.warning("Este repositório ainda precisa ter seu histórico baixado. Nenhuma previsão usa histórico inventado.")
            if st.button("Baixar histórico público"):
                try:
                    with st.spinner("Baixando histórico Git, sem executar código do repositório…"):
                        obter_repo(repo,baixar=True)
                    st.success("Histórico disponível. Agora é possível analisar o commit.")
                except ExtracaoIncompativel as exc: st.error(str(exc))
        if st.button("Estimar risco do commit"):
            try:
                with st.spinner("Conferindo métricas e processando somente os ancestrais do commit…"):
                    valores,link=extrair_cache(repo,commits[i]['sha'],token_opcional())
                mostrar(pipeline,valores,link=link)
            except (ErroGitHub,ExtracaoIncompativel,ValueError,OSError) as exc:
                st.error(str(exc)); st.info("Use um exemplo local enquanto a consulta não estiver disponível.")

st.divider()
st.caption(f"{meta['modelo']} · {meta['estrategia']} · 48 entradas · histórico de primeiro pai.")
st.caption("Treinado com rótulos históricos de projetos Apache envolvendo Java. Não considera o tamanho total do "
           "repositório, não identifica a linha defeituosa e não garante qualidade em projetos da FIAP.")
with st.expander("Desempenho e limites do experimento"):
    final=json.loads((ROOT/'resultados/teste_final.json').read_text(encoding='utf-8'))
    m=final['metricas']
    st.write(f"No período final reavaliado: F1 **{m['f1']:.3f}**, acurácia **{m['acuracia']:.1%}**, "
             f"precision **{m['precision']:.1%}**, recall **{m['recall']:.1%}**.")
    st.caption("F1 resume precision e recall; não é porcentagem de acertos. A acurácia resume acertos no conjunto, "
               "enquanto o score acima se refere somente ao commit selecionado.")
    st.caption(f"Prever sempre a classe majoritária teria acurácia de {final['referencia_majoritaria']['acuracia']:.1%} "
               "e F1 zero. Por isso a seleção considera a detecção de bugs, não só acertos totais.")
    st.caption("Esse período já havia sido avaliado na versão inicial. A comparação atual é exploratória; "
               "uma validação confirmatória exige novos dados. Rótulos SZZ e mensagens de correção são imperfeitos.")
