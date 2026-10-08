"""Revisão reproduzível: pesos de classe, limiar temporal e avaliação até 2018.

python revisar_balanceamento.py [--refazer]
Os resultados originais são preservados; o app usa a revisão concluída.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

from dados import carregar_dividir, salvar_json
from esquema import ROOT, FEATURES, LIMIAR
from modelo import FAIXAS, carregar_modelo, prever
from treinamento import construir

RESULTADOS = ROOT / "resultados/revisao_balanceamento"
ARTEFATOS = ROOT / "artefatos/revisao_balanceamento"
FIM_PRINCIPAL = pd.Timestamp("2019-01-01", tz="UTC")
LIMIARES = np.round(np.arange(0.10, 0.901, 0.01), 2)
CANDIDATOS = [
    {"id": "sem_pesos_fixo", "class_weight": None, "ajustar_limiar": False},
    {"id": "sem_pesos_ajustado", "class_weight": None, "ajustar_limiar": True},
    {"id": "balanced_fixo", "class_weight": "balanced", "ajustar_limiar": False},
    {"id": "balanced_ajustado", "class_weight": "balanced", "ajustar_limiar": True},
]


def medir(y, scores, limiar=LIMIAR):
    pred = (np.asarray(scores) >= limiar).astype(int)
    return {"f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "acuracia": float(accuracy_score(y, pred)),
            "acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores))}


def dividir_limiar(treino):
    """Últimos 20% das datas únicas; nunca divide commits com a mesma data."""
    datas = np.sort(treino.committer_date_git.unique())
    if len(datas) < 5:
        raise ValueError("Poucas datas para validação temporal do limiar.")
    corte = datas[int(len(datas) * 0.8)]
    ajuste = treino[treino.committer_date_git < corte]
    validacao = treino[treino.committer_date_git >= corte]
    if ajuste.buggy.nunique() != 2 or validacao.buggy.nunique() != 2:
        raise ValueError("Cada trecho interno precisa conter ambas as classes.")
    return ajuste, validacao


def escolher_limiar(y, scores):
    curva = pd.DataFrame({"limiar": LIMIARES,
                          "f1": [f1_score(y, np.asarray(scores) >= t, zero_division=0) for t in LIMIARES]})
    curva["distancia_05"] = (curva.limiar - 0.5).abs().round(8)
    melhor = curva.sort_values(["f1", "distancia_05", "limiar"],
                              ascending=[False, True, False], kind="stable").iloc[0]
    return float(melhor.limiar), curva.drop(columns="distancia_05")


def ajustar(treino, parametros, pesos, ajustar_limiar):
    params = {**parametros, "class_weight": pesos}
    limiar, curva, auditoria = LIMIAR, pd.DataFrame(), {}
    if ajustar_limiar:
        anterior, interno = dividir_limiar(treino)
        auxiliar = construir("LightGBM", params)
        auxiliar.fit(anterior[FEATURES], anterior.buggy)
        limiar, curva = escolher_limiar(interno.buggy, auxiliar.predict_proba(interno[FEATURES])[:, 1])
        auditoria = {"n_ajuste_interno": len(anterior), "n_validacao_interna": len(interno),
                     "fim_ajuste_interno": str(anterior.data.max()),
                     "inicio_validacao_interna": str(interno.data.min()),
                     "fim_validacao_interna": str(interno.data.max())}
    pipeline = construir("LightGBM", params)
    pipeline.fit(treino[FEATURES], treino.buggy)
    pipeline.limiar_decisao_ = limiar
    contagens = treino.buggy.value_counts().sort_index()
    auditoria["contagens_fit"] = contagens.to_dict()
    auditoria["pesos_fit"] = {int(c): float(len(treino)/(2*n)) if pesos else 1.0
                              for c, n in contagens.items()}
    return pipeline, curva, auditoria


def separar_periodos(teste):
    principal = teste[teste.data < FIM_PRINCIPAL].reset_index(drop=True)
    sensibilidade = teste[teste.data >= FIM_PRINCIPAL].reset_index(drop=True)
    if principal.empty or sensibilidade.empty:
        raise ValueError("Esperados holdout até 2018 e período de sensibilidade 2019+.")
    return principal, sensibilidade


def comparar(treino, folds, parametros):
    linhas, previsoes, curvas = [], [], []
    for candidato in CANDIDATOS:
        for k, (it, iv) in enumerate(folds, 1):
            passado, futuro = treino.iloc[it], treino.iloc[iv]
            pipe, curva, auditoria = ajustar(passado, parametros, candidato["class_weight"],
                                            candidato["ajustar_limiar"])
            pred = prever(pipe, futuro)
            m = medir(futuro.buggy, pred.score, pipe.limiar_decisao_)
            linhas.append({**candidato, "fold": k, "limiar": pipe.limiar_decisao_, **m,
                           **auditoria, "fim_treino": str(passado.data.max()),
                           "inicio_validacao": str(futuro.data.min()), "n_validacao": len(futuro)})
            previsoes.append(futuro[["commit_id", "data", "buggy"]].assign(
                candidato=candidato["id"], fold=k, score=pred.score,
                previsao=pred.previsao, limiar=pipe.limiar_decisao_))
            if not curva.empty:
                curvas.append(curva.assign(candidato=candidato["id"], fold=k))
            print(f"{candidato['id']} · fold {k}: F1={m['f1']:.4f}, limiar={pipe.limiar_decisao_:.2f}", flush=True)
    detalhe = pd.DataFrame(linhas)
    resumo = []
    for candidato in CANDIDATOS:
        d = detalhe[detalhe.id == candidato["id"]]
        resumo.append({**candidato, "f1_validacao": float(d.f1.mean()), "desvio_f1": float(d.f1.std(ddof=0)),
                       "precision": float(d.precision.mean()), "recall": float(d.recall.mean()),
                       "acuracia_balanceada": float(d.acuracia_balanceada.mean()),
                       "average_precision": float(d.average_precision.mean())})
    tabela = pd.DataFrame(resumo)
    tabela["usa_pesos"] = tabela.class_weight.notna()
    melhor = tabela.sort_values(["f1_validacao", "desvio_f1", "ajustar_limiar", "usa_pesos"],
                                ascending=[False, True, True, True], kind="stable").iloc[0]
    candidato = next(c for c in CANDIDATOS if c["id"] == melhor.id)
    tabela.drop(columns="usa_pesos").to_csv(RESULTADOS / "comparacao.csv", index=False)
    detalhe.to_json(RESULTADOS / "folds.json", orient="records", indent=2, force_ascii=False)
    pd.concat(curvas, ignore_index=True).to_csv(RESULTADOS / "curvas_limiar.csv", index=False)
    oof = pd.concat(previsoes, ignore_index=True)
    oof.to_csv(RESULTADOS / "previsoes_validacao.csv", index=False)
    return candidato, melhor, oof[oof.candidato == candidato["id"]]


def avaliar_periodo(pipe, original, frame, periodo):
    pred, anterior = prever(pipe, frame), prever(original, frame)
    tabela = pd.concat([frame[["commit_id", "project", "data", *FEATURES, "buggy"]], pred], axis=1)
    tabela["score_original"] = anterior.score
    tabela["previsao_original"] = anterior.previsao
    tabela["caso"] = np.select([
        (tabela.buggy == 1) & (tabela.previsao == 1),
        (tabela.buggy == 0) & (tabela.previsao == 0),
        (tabela.buggy == 0) & (tabela.previsao == 1)],
        ["Verdadeiro positivo", "Verdadeiro negativo", "Falso positivo"], default="Falso negativo")
    resumo = {"periodo": periodo, "n_teste": len(frame), "taxa_buggy": float(frame.buggy.mean()),
              "inicio": str(frame.data.min()), "fim": str(frame.data.max()),
              "metricas": medir(frame.buggy, pred.score, pipe.limiar_decisao_),
              "original_mesmos_commits": medir(frame.buggy, anterior.score),
              "referencia_majoritaria": medir(frame.buggy, np.zeros(len(frame))),
              "matriz_confusao": confusion_matrix(frame.buggy, pred.previsao, labels=[0, 1]).tolist(),
              "matriz_original": confusion_matrix(frame.buggy, anterior.previsao, labels=[0, 1]).tolist(),
              "ressalva": "Holdout conhecido: reavaliação exploratória; não é teste novo/intocado."}
    return resumo, tabela


def salvar_faixas(oof):
    nomes, minimos = zip(*FAIXAS)
    grupo = pd.cut(oof.score, [*minimos, np.inf], right=False, labels=list(nomes))
    tabela = oof.groupby(grupo, observed=False).agg(commits=("buggy", "size"), taxa_buggy=("buggy", "mean"))
    tabela = tabela.rename_axis("faixa").reset_index()
    tabela.insert(1, "score_minimo", minimos)
    tabela["taxa_geral"] = oof.buggy.mean()
    tabela.to_csv(ARTEFATOS / "faixas.csv", index=False)


def assinatura_experimento(treino, teste, folds, config, info):
    fontes = [Path(__file__).name, "PROTOCOLO_REVISAO.md", "treinamento.py", "modelo.py", "dados.py"]
    fontes_hash = {nome: hashlib.sha256((ROOT / nome).read_bytes()).hexdigest() for nome in fontes}
    versoes = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm"]}
    conteudo = {"fontes": fontes_hash, "versoes": versoes, "config_original": config, "divisao": info,
                "pipeline_original_sha256": hashlib.sha256((ROOT / "artefatos/pipeline.joblib").read_bytes()).hexdigest(),
                "pistas_sha256": hashlib.sha256((ROOT / "artefatos/pistas_completas.parquet").read_bytes()).hexdigest(),
                "treino_ids": hashlib.sha256("\n".join(treino.commit_id).encode()).hexdigest(),
                "teste_ids": hashlib.sha256("\n".join(teste.commit_id).encode()).hexdigest(),
                "folds": [hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest() for a, b in folds]}
    return hashlib.sha256(json.dumps(conteudo, sort_keys=True).encode()).hexdigest(), conteudo


def executar(refazer=False):
    treino, teste, folds, info = carregar_dividir()
    original, meta_original = carregar_modelo(atual=False)
    if meta_original["modelo"] != "LightGBM":
        raise ValueError("A revisão mantém o LightGBM escolhido na versão original.")
    if treino.data.max() >= FIM_PRINCIPAL:
        raise ValueError("O protocolo exige treino original anterior a 2019.")
    principal, sensibilidade = separar_periodos(teste)
    config_original = json.loads((ROOT / "resultados/selecao.json").read_text(encoding="utf-8"))
    assinatura, manifesto = assinatura_experimento(treino, teste, folds, config_original, info)
    if (ARTEFATOS / "metadados.json").exists() and not refazer:
        meta = json.loads((ARTEFATOS / "metadados.json").read_text(encoding="utf-8"))
        if meta.get("assinatura_revisao") != assinatura:
            raise ValueError("Revisão mudou. Use --refazer para recalcular sem reutilizar resultados antigos.")
        carregar_modelo()
        print("Revisão já concluída; assinatura e modelo conferidos.", flush=True)
        return json.loads((RESULTADOS / "teste_final.json").read_text(encoding="utf-8"))
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    ARTEFATOS.mkdir(parents=True, exist_ok=True)
    salvar_json(RESULTADOS / "manifesto.json", {"assinatura": assinatura, **manifesto})
    pd.concat([treino[["commit_id", "data", "buggy"]].assign(conjunto="treino"),
               principal[["commit_id", "data", "buggy"]].assign(conjunto="teste_ate_2018"),
               sensibilidade[["commit_id", "data", "buggy"]].assign(conjunto="sensibilidade_2019_mais")]
              ).to_csv(RESULTADOS / "particoes.csv", index=False)
    por_ano = pd.concat([treino, teste]).assign(ano=lambda d: d.data.dt.year).groupby("ano").agg(
        commits=("buggy", "size"), bugs=("buggy", "sum"), taxa_buggy=("buggy", "mean"))
    por_ano.to_csv(RESULTADOS / "distribuicao_anual.csv")
    candidato, melhor, oof = comparar(treino, folds, meta_original["parametros"])
    pipe, curva, auditoria = ajustar(treino, meta_original["parametros"], candidato["class_weight"],
                                    candidato["ajustar_limiar"])
    config = {**candidato, "modelo": "LightGBM", "estrategia": "Revisão de pesos e limiar temporal",
              "parametros": {**meta_original["parametros"], "class_weight": candidato["class_weight"]},
              "limiar": pipe.limiar_decisao_, "f1_validacao": float(melhor.f1_validacao),
              "desvio_f1": float(melhor.desvio_f1), "ajuste_final": auditoria,
              "regra": "Maior F1 médio externo; menor desvio; limiar fixo e sem pesos em empate."}
    salvar_json(RESULTADOS / "selecao.json", config)
    curva.to_csv(RESULTADOS / "curva_limiar_final.csv", index=False)
    # A configuração está congelada. Nenhuma decisão abaixo usa os resultados do holdout.
    resumo, previsoes = avaliar_periodo(pipe, original, principal, "Avaliação principal: 06/10/2017 a 31/12/2018 (UTC).")
    separado, previsoes_2019 = avaliar_periodo(pipe, original, sensibilidade, "Sensibilidade: integração em 2019 ou depois.")
    salvar_json(RESULTADOS / "teste_final.json", resumo)
    salvar_json(RESULTADOS / "sensibilidade_2019.json", separado)
    previsoes.to_csv(RESULTADOS / "previsoes_teste.csv", index=False)
    previsoes_2019.to_csv(RESULTADOS / "previsoes_2019.csv", index=False)
    exemplos = previsoes.groupby("caso", sort=True).head(1)
    restantes = previsoes.drop(exemplos.index)
    perto = restantes.assign(distancia=(restantes.score - pipe.limiar_decisao_).abs()).sort_values("distancia").head(1)
    claro = restantes.drop(perto.index).sort_values("score").head(1)
    exemplos = pd.concat([exemplos, perto, claro]).drop(columns="distancia", errors="ignore")
    exemplos.to_csv(ARTEFATOS / "exemplos.csv", index=False)
    salvar_faixas(oof)
    joblib.dump(pipe, ARTEFATOS / "pipeline.joblib", compress=3)
    meta = {**meta_original, **config, "assinatura_revisao": assinatura,
            "resultado_avaliacao": "resultados/revisao_balanceamento/teste_final.json",
            "divisao": {**info, "n_teste": len(principal), "n_sensibilidade_2019": len(sensibilidade),
                        "fim_teste_exclusivo": str(FIM_PRINCIPAL)},
            "pipeline_sha256": hashlib.sha256((ARTEFATOS / "pipeline.joblib").read_bytes()).hexdigest()}
    # Metadados publicados por último: sinalizam que a revisão está completa para o app.
    salvar_json(ARTEFATOS / "metadados.json", meta)
    recarregado, _ = carregar_modelo()
    p = prever(recarregado, exemplos)
    np.testing.assert_allclose(p.score, exemplos.score, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(p.previsao, exemplos.previsao)
    print(json.dumps({"selecao": config, "principal": resumo, "sensibilidade_2019": separado},
                     ensure_ascii=False, indent=2), flush=True)
    return resumo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refazer", action="store_true", help="Recalcular somente a revisão, mantendo a versão original.")
    executar(parser.parse_args().refazer)
