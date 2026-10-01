"""Um único esquema e pipeline para notebook, exemplos e GitHub."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from esquema import ROOT, FEATURES, NEGATIVAS, SEED, LIMIAR, VERSAO_EXTRATOR

FAIXAS=[("Baixo",0.0),("Moderado",0.2),("Alto",0.5),("Muito alto",0.7)]


def entradas(registros):
    frame=pd.DataFrame([registros] if isinstance(registros,dict) else registros)
    faltam=set(FEATURES)-set(frame.columns)
    if faltam: raise ValueError("Histórico/entradas incompletos: "+", ".join(sorted(faltam)))
    frame=frame[FEATURES].apply(pd.to_numeric,errors="raise").astype(float)
    if not np.isfinite(frame.to_numpy()).all(): raise ValueError("Não é permitido substituir histórico ausente por zeros ou NaN.")
    if (frame[[f for f in FEATURES if f not in NEGATIVAS]]<0).any().any(): raise ValueError("Entradas negativas inválidas.")
    if (frame.nf<1).any(): raise ValueError("Um commit precisa afetar pelo menos um arquivo.")
    for f in ["la","ld","nf"]:
        if (frame[f]%1!=0).any(): raise ValueError("la, ld e nf devem ser inteiros.")
    for f in NEGATIVAS-{"q_fuso_h"}:
        if ((frame[f]<0)&(frame[f]!=-1)).any(): raise ValueError("Idade ausente deve ser -1.")
    for f in [c for c in FEATURES if "_frac_" in c]:
        if (frame[f]>1).any(): raise ValueError("Frações devem ficar entre 0 e 1.")
    if (frame.o_n_binarios!=0).any(): raise ValueError("Commits com arquivos binários estão fora do escopo.")
    return frame


def carregar_modelo():
    meta=json.loads((ROOT/"artefatos/metadados.json").read_text(encoding="utf-8"))
    if meta["variaveis"]!=FEATURES or meta["limiar"]!=LIMIAR or meta["versao_extrator"]!=VERSAO_EXTRATOR:
        raise ValueError("Modelo e esquema incompatíveis.")
    atual=hashlib.sha256((ROOT/'extracao.py').read_bytes()+(ROOT/'esquema.py').read_bytes()).hexdigest()
    if atual!=meta['divisao']['extrator_sha256']:
        raise ValueError("O extrator foi alterado desde o treinamento deste modelo.")
    path=ROOT/"artefatos/pipeline.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["pipeline_sha256"]:
        raise ValueError("Arquivo de modelo diferente do registrado.")
    pipeline=joblib.load(path)
    if list(pipeline.feature_names_in_)!=FEATURES: raise ValueError("Ordem das entradas divergente.")
    return pipeline,meta


def prever(pipeline,registros):
    X=entradas(registros)
    scores=pipeline.predict_proba(X)[:,list(pipeline.classes_).index(1)].astype(float)
    return pd.DataFrame({"previsao":(scores>=LIMIAR).astype(int),"score":scores})
