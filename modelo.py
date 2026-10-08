"""Um único esquema e pipeline para notebook, exemplos e GitHub."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from esquema import ROOT, FEATURES, NEGATIVAS, SEED, LIMIAR, VERSAO_EXTRATOR
from novas_pistas import ESTATICAS, colunas_entrada

FAIXAS=[("Baixo",0.0),("Moderado",0.2),("Alto",0.5),("Muito alto",0.7)]


def entradas(registros, variaveis=None):
    variaveis = FEATURES if variaveis is None else list(variaveis)
    if variaveis not in [FEATURES, FEATURES + list(ESTATICAS)]:
        raise ValueError("Esquema de entradas desconhecido.")
    frame=pd.DataFrame([registros] if isinstance(registros,dict) else registros)
    faltam=set(variaveis)-set(frame.columns)
    if faltam: raise ValueError("Histórico/entradas incompletos: "+", ".join(sorted(faltam)))
    frame=frame[variaveis].apply(pd.to_numeric,errors="raise").astype(float)
    if not np.isfinite(frame.to_numpy()).all(): raise ValueError("Não é permitido substituir histórico ausente por zeros ou NaN.")
    if (frame[[f for f in variaveis if f not in NEGATIVAS]]<0).any().any(): raise ValueError("Entradas negativas inválidas.")
    if (frame.nf<1).any(): raise ValueError("Um commit precisa afetar pelo menos um arquivo.")
    for f in ["la","ld","nf"]:
        if (frame[f]%1!=0).any(): raise ValueError("la, ld e nf devem ser inteiros.")
    for f in NEGATIVAS-{"q_fuso_h"}:
        if ((frame[f]<0)&(frame[f]!=-1)).any(): raise ValueError("Idade ausente deve ser -1.")
    for f in [c for c in FEATURES if "_frac_" in c]:
        if (frame[f]>1).any(): raise ValueError("Frações devem ficar entre 0 e 1.")
    if (frame.o_n_binarios!=0).any(): raise ValueError("Commits com arquivos binários estão fora do escopo.")
    if 'n_entropia' in frame:
        for f in ['n_entropia_normalizada', 'n_fracao_churn_max', 'n_fracao_ld_teste']:
            if (frame[f] > 1 + 1e-12).any(): raise ValueError("Dispersão normalizada deve ficar entre 0 e 1.")
        if ((frame.n_subsistemas < 1) | (frame.n_subsistemas > frame.nf)).any():
            raise ValueError("Quantidade de subsistemas incompatível com os arquivos.")
    return frame


def pasta_artefatos(atual=True):
    if atual:
        for nome in ["melhorias", "revisao_balanceamento"]:
            pasta = ROOT / "artefatos" / nome
            if (pasta / "metadados.json").exists(): return pasta
    return ROOT / "artefatos"


def carregar_modelo(atual=True, pasta=None):
    pasta = pasta_artefatos(atual) if pasta is None else pasta
    meta=json.loads((pasta/"metadados.json").read_text(encoding="utf-8"))
    esperadas = colunas_entrada(meta.get('pistas', 'originais'))
    if meta["variaveis"]!=esperadas or not 0 < meta["limiar"] < 1 or meta["versao_extrator"]!=VERSAO_EXTRATOR:
        raise ValueError("Modelo e esquema incompatíveis.")
    atual=hashlib.sha256((ROOT/'extracao.py').read_bytes()+(ROOT/'esquema.py').read_bytes()).hexdigest()
    if atual!=meta['divisao']['extrator_sha256']:
        raise ValueError("O extrator foi alterado desde o treinamento deste modelo.")
    for arquivo, campo in [('novas_pistas.py', 'novas_pistas_sha256'),
                           ('extrair_dispersao.py', 'extracao_dispersao_sha256')]:
        if campo in meta and hashlib.sha256((ROOT / arquivo).read_bytes()).hexdigest() != meta[campo]:
            raise ValueError("A engenharia de pistas foi alterada desde o treinamento.")
    path=pasta/"pipeline.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["pipeline_sha256"]:
        raise ValueError("Arquivo de modelo diferente do registrado.")
    pipeline=joblib.load(path)
    if list(pipeline.feature_names_in_)!=esperadas: raise ValueError("Ordem das entradas divergente.")
    if 'variaveis_modelo' in meta and list(pipeline.named_steps['modelo'].feature_names_in_) != meta['variaveis_modelo']:
        raise ValueError("Pistas internas do modelo divergentes.")
    if getattr(pipeline, "limiar_decisao_", LIMIAR) != meta["limiar"]:
        raise ValueError("Limiar do modelo diferente do registrado.")
    return pipeline,meta


def prever(pipeline,registros):
    X=entradas(registros, getattr(pipeline, 'feature_names_in_', FEATURES))
    scores=pipeline.predict_proba(X)[:,list(pipeline.classes_).index(1)].astype(float)
    limiar = getattr(pipeline, "limiar_decisao_", LIMIAR)
    return pd.DataFrame({"previsao":(scores>=limiar).astype(int),"score":scores}, index=X.index)
