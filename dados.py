"""Base original e cobertura; validação temporal por integração do commit."""
import hashlib
import json
import os
import platform
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import TimeSeriesSplit
from esquema import ROOT, FEATURES, VERSAO_EXTRATOR
from extracao import extrair, git

URL = "https://zenodo.org/records/5907847/files/apachejit_dataset_replication.zip?download=1"
CSV_SHA256 = "5097cbbbab8c4611a709b3cf95ac5c9fe5f62e619bab081911a2a247ddbed4e0"


def salvar_json(path, conteudo):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conteudo,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def pasta_repos():
    """Pasta dos históricos Git: variável CP5_REPOS_DIR ou `repos/` dentro do projeto."""
    if os.getenv("CP5_REPOS_DIR"):
        return Path(os.environ["CP5_REPOS_DIR"])
    return ROOT / "repos"


def baixar():
    pasta=ROOT/"data"; pasta.mkdir(exist_ok=True)
    csv=pasta/"apachejit_total.csv"
    if not csv.exists():
        pacote=pasta/"apachejit_dataset_replication.zip"
        if not pacote.exists():
            with requests.get(URL,stream=True,timeout=120) as r:
                r.raise_for_status()
                with pacote.with_suffix(".part").open("wb") as f:
                    for bloco in r.iter_content(1024*1024): f.write(bloco)
            pacote.with_suffix(".part").replace(pacote)
        if hashlib.md5(pacote.read_bytes()).hexdigest()!="528bf0ee04b15976be6bf15f8efddc65":
            raise ValueError("Pacote ApacheJIT v2 com checksum divergente.")
        with zipfile.ZipFile(pacote) as z: csv.write_bytes(z.read("apachejit/dataset/apachejit_total.csv"))
    if hashlib.sha256(csv.read_bytes()).hexdigest()!=CSV_SHA256:
        raise ValueError("CSV diferente do ApacheJIT v2 verificado. Não reutilize resultados.")
    return csv


def base_original():
    dados=pd.read_csv(baixar())
    y=dados.buggy.astype(str).str.lower().map({"true":1,"false":0,"1":1,"0":0})
    if y.isna().any(): raise ValueError("Rótulos desconhecidos.")
    dados["buggy"]=y.astype(int)
    dados["data"]=pd.to_datetime(dados.author_date,unit="s",utc=True)
    dados=dados.sort_values(["data","commit_id","project"],kind="stable")
    repetidos=dados[dados.duplicated("commit_id",keep=False)]
    if repetidos.groupby("commit_id")[["la","ld","nf","buggy","author_date"]].nunique().gt(1).any().any():
        raise ValueError("SHAs duplicados com informações conflitantes.")
    return dados.drop_duplicates("commit_id").reset_index(drop=True)


def hash_extrator():
    return hashlib.sha256((ROOT/"extracao.py").read_bytes()+(ROOT/"esquema.py").read_bytes()).hexdigest()


def presentes(repo, shas):
    if not repo.exists(): return set()
    resposta=git(repo,["cat-file","--batch-check=%(objectname) %(objecttype)"],input=("\n".join(shas)+"\n").encode()).stdout
    return {linha.split()[0].decode() for linha in resposta.splitlines() if linha.endswith(b" commit")}


def clonar_repos():
    """Preparação explícita via CLI; não ocorre ao abrir o notebook/app."""
    for projeto in sorted(base_original().project.unique()):
        nome=projeto.split("/")[1]; pasta=pasta_repos()/f"{nome}.git"
        if pasta.exists(): continue
        pasta.parent.mkdir(parents=True,exist_ok=True)
        print("Baixando histórico público:",projeto,flush=True)
        subprocess.run(["git","clone","--bare","--no-recurse-submodules",f"https://github.com/{projeto}.git",str(pasta)],
                       check=True,env={**os.environ,"GIT_TERMINAL_PROMPT":"0"},timeout=1800)


def extrair_base():
    dados=base_original(); distribuicao={}; faltantes=[]
    for projeto, grupo in dados.groupby("project",sort=True):
        nome=projeto.split("/")[1]; shas=grupo.commit_id.to_list()
        encontrados=presentes(pasta_repos()/f"{nome}.git",shas)
        distribuicao.setdefault(nome,[]).extend(s for s in shas if s in encontrados)
        faltam=set(shas)-encontrados
        if nome.startswith("hadoop-") and faltam:
            recuperados=presentes(pasta_repos()/"hadoop.git",sorted(faltam))
            distribuicao.setdefault("hadoop",[]).extend(sorted(recuperados)); faltam-=recuperados
        faltantes.extend({"commit_id":s,"project":projeto,"motivo":"objeto_git_indisponivel"} for s in sorted(faltam))
    if faltantes:
        salvar_json(ROOT/"resultados/objetos_faltantes.json",faltantes)
        raise ValueError(f"{len(faltantes)} commits sem objeto Git. Execute dados.py --clonar e investigue o relatório.")
    pasta=ROOT/"data/pistas"; pasta.mkdir(exist_ok=True)
    tabelas=[]; auditoria=[]
    for nome, shas in sorted(distribuicao.items(),key=lambda item:len(item[1])):
        if not shas: continue
        shas=sorted(set(shas)); path=pasta/f"{nome}.parquet"; meta=pasta/f"{nome}.json"
        assinatura={"extrator_sha256":hash_extrator(),"alvos_sha256":hashlib.sha256("\n".join(shas).encode()).hexdigest()}
        if path.exists() and meta.exists() and all(json.loads(meta.read_text())[k]==v for k,v in assinatura.items()):
            tabela=pd.read_parquet(path); info=json.loads(meta.read_text()); print(nome,"cache conferido",flush=True)
        else:
            print(f"Extraindo {nome}: {len(shas)} alvos e suas cadeias ancestrais",flush=True)
            inicio=time.perf_counter()
            tabela=pd.DataFrame(extrair(pasta_repos()/f"{nome}.git",shas))
            tabela["repositorio_git"]=f"apache/{nome}"
            tabela.to_parquet(path,index=False)
            info={**assinatura,"repo":nome,"alvos":len(tabela),"tempo_s":time.perf_counter()-inicio,
                  "git":git(pasta_repos()/f"{nome}.git",["--version"]).stdout.decode().strip()}
            salvar_json(meta,info)
        tabelas.append(tabela); auditoria.append(info)
        print(f"{nome}: {len(tabela)} extraídos; {tabela.motivo.ne('').sum()} incompatíveis",flush=True)
    pistas=pd.concat(tabelas,ignore_index=True)
    if pistas.commit_id.duplicated().any(): raise ValueError("Extração repetida de um mesmo SHA.")
    (ROOT/"artefatos").mkdir(exist_ok=True)
    compacto=ROOT/"artefatos/pistas_completas.parquet"
    pistas.to_parquet(compacto,index=False)
    salvar_json(ROOT/"resultados/extracao.json",{"versao":VERSAO_EXTRATOR,"extrator_sha256":hash_extrator(),
                "pistas_sha256":hashlib.sha256(compacto.read_bytes()).hexdigest(),
                "repositorios":auditoria,"linhas":len(pistas),"python":platform.python_version()})
    return pistas


def carregar_dividir():
    base=base_original()
    path=ROOT/"artefatos/pistas_completas.parquet"
    if not path.exists(): extrair_base()
    manifesto=json.loads((ROOT/"resultados/extracao.json").read_text(encoding="utf-8"))
    if manifesto["extrator_sha256"]!=hash_extrator():
        raise ValueError("Extrator alterado. Execute dados.py --extrair antes de treinar.")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifesto["pistas_sha256"]:
        raise ValueError("Características diferentes do manifesto da extração.")
    pistas=pd.read_parquet(path)
    corte=base.iloc[int(len(base)*0.8)].data
    bruto_treino=base[base.data<corte].reset_index(drop=True)
    juntos=base.merge(pistas,on="commit_id",how="left",suffixes=("_csv",""),validate="one_to_one")
    if juntos.repositorio_git.isna().any(): raise ValueError("Cobertura incompleta da extração.")
    for f in ["la","ld","nf"]:
        diff=juntos[f]!=juntos[f+"_csv"]
        juntos.loc[diff,"motivo"] = juntos.loc[diff,"motivo"] + f";{f}_diverge_csv"
    diff=juntos.author_date!=juntos.author_date_git
    juntos.loc[diff,"motivo"]=juntos.loc[diff,"motivo"]+";data_diverge_csv"
    juntos['data_autoria']=juntos['data']
    juntos['data']=pd.to_datetime(juntos.committer_date_git,unit='s',utc=True)
    juntos=juntos.sort_values(['data','commit_id','project'],kind='stable').reset_index(drop=True)
    datas=np.sort(juntos.loc[juntos.data<corte,'committer_date_git'].unique())
    janelas=[(datas[a],datas[b]) for a,b in TimeSeriesSplit(n_splits=3).split(datas)]
    excluidos=juntos[juntos.motivo.ne("")]
    excluidos[["commit_id","project","data","motivo"]].to_csv(ROOT/"resultados/exclusoes.csv",index=False)
    elegiveis=juntos[juntos.motivo.eq("")].copy()
    treino=elegiveis[elegiveis.data<corte].reset_index(drop=True)
    teste=elegiveis[elegiveis.data>=corte].reset_index(drop=True)
    folds=[(np.flatnonzero(treino.committer_date_git.isin(a)),np.flatnonzero(treino.committer_date_git.isin(b))) for a,b in janelas]
    assert not set(treino.commit_id)&set(teste.commit_id)
    assert treino.data.max()<teste.data.min()
    for a,b in folds: assert treino.iloc[a].data.max()<treino.iloc[b].data.min()
    info={"dimensoes_originais":[len(base),18],"n_treino_original":int((juntos.data<corte).sum()),
          "n_teste_original":int((juntos.data>=corte).sum()),"n_treino":len(treino),"n_teste":len(teste),
          "ordenacao":"committer_date Git (integração)","n_treino_por_autoria_antes_filtro":len(bruto_treino),
          "autoria_antes_integracao_depois_corte":int(((juntos.data_autoria<corte)&(juntos.data>=corte)).sum()),
          "n_excluidos":len(excluidos),"inicio_teste":str(corte),"csv_sha256":CSV_SHA256,
          "extrator_sha256":hash_extrator(),"sha_compartilhados":0,
          "avaliacao_final":"Holdout já avaliado na versão inicial; reavaliação exploratória, não teste intocado."}
    return treino,teste,folds,info


def auditar_treino(treino):
    from modelo import entradas
    entradas(treino)
    return {"tipos":treino[FEATURES].dtypes.astype(str).to_dict(),
            "ausencias":treino[FEATURES].isna().sum().astype(int).to_dict(),
            "classes":treino.buggy.value_counts().sort_index().to_dict(),
            "duplicatas_sha":int(treino.commit_id.duplicated().sum()),
            "estatisticas":treino[FEATURES].describe(percentiles=[0.5,0.95,0.99]).to_dict()}


def verificar_extracao(treino):
    """A checagem completa de la/ld/nf ocorreu antes de definir a elegibilidade."""
    for f in ["la","ld","nf"]: assert (treino[f]==treino[f+"_csv"]).all()
    path=ROOT/"resultados/compatibilidade.csv"
    treino[["commit_id","project","repositorio_git","la","ld","nf","la_csv","ld_csv","nf_csv"]].to_csv(path,index=False)
    return pd.read_csv(path)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="Preparar histórico público para o modelo completo.")
    p.add_argument("--clonar",action="store_true"); p.add_argument("--extrair",action="store_true")
    args=p.parse_args()
    if args.clonar: clonar_repos()
    if args.extrair: extrair_base()
    if args.extrair: print(carregar_dividir()[3])
