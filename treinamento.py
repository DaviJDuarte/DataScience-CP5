"""Experimento completo. Resultados salvos permitem retomar sem abrir o teste."""
import hashlib
import importlib.metadata
import json
import platform
from time import perf_counter
import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from dados import carregar_dividir, auditar_treino, verificar_extracao, salvar_json
from modelo import ROOT, FEATURES, FAIXAS, LIMIAR, SEED, prever
from esquema import VERSAO_EXTRATOR, DESCRICOES, GRUPOS, NEGATIVAS

RESULTADOS = ROOT / "resultados"
MODELOS = ["Random Forest", "XGBoost", "LightGBM"]
N_JOBS = 2
GRADES = {
    "Random Forest": {"n_estimators": [150, 300], "max_depth": [8, None], "min_samples_leaf": [1, 5]},
    "XGBoost": {"n_estimators": [150, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
    "LightGBM": {"n_estimators": [150, 300], "num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
}


def slug(nome):
    return nome.lower().replace(" ", "_")


def construir(nome, parametros=None):
    comuns = {"random_state": SEED, "n_jobs": N_JOBS, "n_estimators": 150}
    if nome == "Random Forest":
        params = {**comuns, "max_depth": None, "min_samples_leaf": 1, **(parametros or {})}
        modelo = RandomForestClassifier(**params)
    elif nome == "XGBoost":
        params = {**comuns, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist",
                  "objective": "binary:logistic", "eval_metric": "logloss", **(parametros or {})}
        modelo = XGBClassifier(**params)
    elif nome == "LightGBM":
        params = {**comuns, "num_leaves": 31, "learning_rate": 0.1, "verbosity": -1,
                  "deterministic": True, "force_col_wise": True, **(parametros or {})}
        modelo = LGBMClassifier(**params)
    elif nome == "Sempre positivo":
        modelo = DummyClassifier(strategy="constant", constant=1, random_state=SEED)
    else:
        modelo = DummyClassifier(strategy="most_frequent", random_state=SEED)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")
    return Pipeline([("imputacao", imputer), ("modelo", modelo)])


def metricas(y, p):
    pred = (np.asarray(p) >= LIMIAR).astype(int)
    return {"f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "acuracia": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p)}


def pontuar(estimator, X, y):
    return metricas(y, estimator.predict_proba(X)[:, 1])


def avaliar_cv(nome, params, treino, folds):
    inicio = perf_counter()
    scores = cross_validate(construir(nome, params), treino[FEATURES], treino.buggy,
                            cv=folds, scoring=pontuar, n_jobs=1, return_train_score=True,
                            error_score="raise")
    return scores, perf_counter() - inicio


def resumo_cv(nome, estrategia, params, scores, tempo):
    return {"modelo": nome, "estrategia": estrategia, "parametros": params,
            "f1_treino": float(np.mean(scores["train_f1"])),
            "f1_validacao": float(np.mean(scores["test_f1"])),
            "desvio_f1": float(np.std(scores["test_f1"])),
            "gap": float(np.mean(scores["train_f1"]) - np.mean(scores["test_f1"])),
            "tempo_s": float(tempo), "ajuste_medio_s": float(np.mean(scores["fit_time"])),
            "f1_folds": scores["test_f1"].tolist(),
            **{f"validacao_{m}": float(np.mean(scores[f"test_{m}"]))
               for m in ["precision", "recall", "acuracia", "roc_auc", "average_precision"]}}


def preparar():
    RESULTADOS.mkdir(exist_ok=True)
    treino, teste, folds, info = carregar_dividir()
    verificar_extracao(treino)
    # Impede reutilizar resultados em dados ou protocolo diferentes.
    assinatura = hashlib.sha256((info["csv_sha256"] + info["extrator_sha256"] + str(FEATURES)
                                + (ROOT / "PROTOCOLO.md").read_text(encoding="utf-8")
                                + json.dumps(GRADES, sort_keys=True)).encode()).hexdigest()
    path = RESULTADOS / "assinatura.txt"
    if path.exists() and path.read_text() != assinatura:
        raise ValueError("Dados/protocolo diferentes dos resultados salvos. Use outra pasta de experimento.")
    path.write_text(assinatura)
    salvar_json(RESULTADOS / "divisao.json", info)
    salvar_json(RESULTADOS / "fonte.json", {
        "titulo": "ApacheJIT: A Large Dataset for Just-In-Time Defect Prediction",
        "doi": "10.5281/zenodo.5907847", "versao": "v2",
        "origem_versao": "Página do registro no Zenodo",
        "registro": "https://zenodo.org/records/5907847",
        "metadados": "https://zenodo.org/api/records/5907847",
        "licenca_metadados": {"id": "cc-by-4.0"}, "metadados_verificados_em": "2026-10-01",
        "arquivo": "apachejit/dataset/apachejit_total.csv",
        "md5_pacote_verificado": "528bf0ee04b15976be6bf15f8efddc65",
        "observacao": "HTML sem texto de licença; CC BY 4.0 conferida na API oficial na data indicada."})
    salvar_json(RESULTADOS / "auditoria_treino.json", auditar_treino(treino))
    pd.concat([treino[["commit_id", "project", "data"]].assign(conjunto="treino"),
               teste[["commit_id", "project", "data"]].assign(conjunto="teste")]).to_csv(
                   RESULTADOS / "particoes.csv", index=False)
    linhas = []
    arrays = {}
    for k, (tr, va) in enumerate(folds, 1):
        arrays[f"treino_{k}"] = tr
        arrays[f"validacao_{k}"] = va
        linhas.append({"fold": k, "n_treino": len(tr), "n_validacao": len(va),
                       "treino_inicio": str(treino.iloc[tr].data.min()),
                       "treino_fim": str(treino.iloc[tr].data.max()),
                       "validacao_inicio": str(treino.iloc[va].data.min()),
                       "validacao_fim": str(treino.iloc[va].data.max())})
    np.savez_compressed(RESULTADOS / "folds.npz", **arrays)
    pd.DataFrame(linhas).to_csv(RESULTADOS / "folds.csv", index=False)
    return treino, teste, folds, info


def baselines(treino, folds):
    linhas = []
    for nome in [*MODELOS, "Dummy", "Sempre positivo"]:
        path = RESULTADOS / f"baseline_{slug(nome)}.json"
        if path.exists():
            linha = json.loads(path.read_text(encoding="utf-8"))
        else:
            print("Baseline:", nome, flush=True)
            scores, tempo = avaliar_cv(nome, {}, treino, folds)
            linha = resumo_cv(nome, "Baseline", {}, scores, tempo)
            linha["hiperparametros_completos"] = construir(nome).named_steps["modelo"].get_params()
            salvar_json(path, linha)
        linhas.append(linha)
    return linhas


def buscar_grid(nome, treino, folds):
    path = RESULTADOS / f"grid_{slug(nome)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("Grid Search:", nome, "(8 combinações x 3 folds)", flush=True)
    busca = GridSearchCV(construir(nome), {f"modelo__{k}": v for k, v in GRADES[nome].items()},
                         scoring=pontuar, refit=False, cv=folds, n_jobs=1,
                         return_train_score=True, error_score="raise")
    inicio = perf_counter()
    busca.fit(treino[FEATURES], treino.buggy)
    tempo = perf_counter() - inicio
    tabela = pd.DataFrame(busca.cv_results_)
    tabela.to_csv(RESULTADOS / f"grid_{slug(nome)}_candidatos.csv", index=False)
    joblib.dump(busca.cv_results_, RESULTADOS / f"grid_{slug(nome)}_cv.joblib")
    melhor = int(tabela.mean_test_f1.argmax())
    params = {k.removeprefix("modelo__"): v for k, v in busca.cv_results_["params"][melhor].items()}
    scores = {f"{lado}_{metrica}": np.array([busca.cv_results_[f"split{k}_{lado}_{metrica}"][melhor]
               for k in range(3)]) for lado in ["train", "test"]
               for metrica in ["f1", "precision", "recall", "acuracia", "roc_auc", "average_precision"]}
    scores["fit_time"] = np.array([busca.cv_results_["mean_fit_time"][melhor]])
    linha = resumo_cv(nome, "Grid Search", params, scores, tempo)
    salvar_json(path, linha)
    return linha


def sugerir(trial, nome):
    params = {"n_estimators": trial.suggest_int("n_estimators", 150, 300, step=50)}
    if nome == "Random Forest":
        params.update(max_depth=trial.suggest_categorical("max_depth", [8, 16, None]),
                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                      max_features=trial.suggest_categorical("max_features", ["sqrt", 1.0]))
    else:
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.2, log=True)
        if nome == "XGBoost":
            params.update(max_depth=trial.suggest_int("max_depth", 2, 6),
                          min_child_weight=trial.suggest_int("min_child_weight", 1, 10))
        else:
            params.update(num_leaves=trial.suggest_int("num_leaves", 15, 63),
                          min_child_samples=trial.suggest_int("min_child_samples", 10, 60))
    return params


def buscar_optuna(nome, treino, folds):
    path = RESULTADOS / f"optuna_{slug(nome)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler_path = RESULTADOS / f"sampler_{slug(nome)}.joblib"
    sampler = joblib.load(sampler_path) if sampler_path.exists() else optuna.samplers.TPESampler(seed=SEED)
    estudo = optuna.create_study(study_name=slug(nome), direction="maximize", sampler=sampler,
                                 storage=f"sqlite:///{(RESULTADOS / 'optuna.sqlite3').as_posix()}",
                                 load_if_exists=True)
    # Trials interrompidos não contam para os 20 completos.
    for t in estudo.get_trials(deepcopy=False):
        if t.state == optuna.trial.TrialState.RUNNING:
            estudo.tell(t.number, state=optuna.trial.TrialState.FAIL)
    def objetivo(trial):
        params = sugerir(trial, nome)
        scores, tempo = avaliar_cv(nome, params, treino, folds)
        linha = resumo_cv(nome, "Optuna", params, scores, tempo)
        trial.set_user_attr("resultado", linha)
        return linha["f1_validacao"]
    def checkpoint(study, trial):
        joblib.dump(study.sampler, sampler_path)
        study.trials_dataframe().to_csv(RESULTADOS / f"optuna_{slug(nome)}_trials.csv", index=False)
        print(f"Optuna {nome}: trial {trial.number + 1}/20, F1={trial.value:.4f}", flush=True)
    completos = sum(t.state == optuna.trial.TrialState.COMPLETE for t in estudo.trials)
    estudo.optimize(objetivo, n_trials=max(0, 20 - completos), n_jobs=1, callbacks=[checkpoint])
    linha = dict(estudo.best_trial.user_attrs["resultado"])
    linha["tempo_melhor_trial_s"] = linha["tempo_s"]
    linha["tempo_s"] = sum(t.duration.total_seconds() for t in estudo.trials if t.duration)
    linha["trials_completos"] = sum(t.state == optuna.trial.TrialState.COMPLETE for t in estudo.trials)
    salvar_json(path, linha)
    return linha


def comparacao(treino, folds):
    linhas = baselines(treino, folds)
    for nome in MODELOS:
        linhas.append(buscar_grid(nome, treino, folds))
        linhas.append(buscar_optuna(nome, treino, folds))
    tabela = pd.DataFrame([r for r in linhas if r["modelo"] in MODELOS])
    assert len(tabela) == 9
    tabela.to_csv(RESULTADOS / "comparacao.csv", index=False)
    return tabela


def selecionar(tabela):
    proximos = tabela[tabela.f1_validacao >= tabela.f1_validacao.max() - 0.005].copy()
    proximos["gap_positivo"] = proximos.gap.clip(lower=0)
    proximos["arvores"] = proximos.parametros.map(lambda p: p.get("n_estimators", 150))
    vencedor = proximos.sort_values(["desvio_f1", "gap_positivo", "arvores", "ajuste_medio_s"],
                                    kind="stable").iloc[0]
    config = {"modelo": vencedor.modelo, "estrategia": vencedor.estrategia,
              "parametros": vencedor.parametros, "variaveis": FEATURES, "limiar": LIMIAR,
              "f1_validacao": vencedor.f1_validacao, "desvio_f1": vencedor.desvio_f1,
              "gap": vencedor.gap, "regra": "Tolerância 0,005; desvio, gap positivo, árvores e tempo."}
    salvar_json(RESULTADOS / "selecao.json", config)
    return config


def diagnosticar(config, treino, folds):
    path = RESULTADOS / "diagnosticos.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    previsoes, aprendizado, importancias = [], [], []
    for k, (it, iv) in enumerate(folds, 1):
        for fracao in [0.25, 0.5, 1.0]:
            # Prefixos crescentes do passado; toda validação permanece posterior.
            indices = it[:max(2, int(len(it) * fracao))]
            modelo = construir(config["modelo"], config["parametros"])
            modelo.fit(treino.iloc[indices][FEATURES], treino.iloc[indices].buggy)
            p_train = modelo.predict_proba(treino.iloc[indices][FEATURES])[:, 1].astype(float)
            p_val = modelo.predict_proba(treino.iloc[iv][FEATURES])[:, 1].astype(float)
            aprendizado.append({"fold": k, "fracao": fracao, "n_treino": len(indices),
                                "f1_treino": metricas(treino.iloc[indices].buggy, p_train)["f1"],
                                "f1_validacao": metricas(treino.iloc[iv].buggy, p_val)["f1"]})
            if fracao == 1:
                previsoes.append(treino.iloc[iv][["commit_id", "buggy"]].assign(
                    fold=k, score=p_val, previsao=(p_val >= LIMIAR).astype(int)))
                imp = modelo.named_steps["modelo"].feature_importances_.astype(float)
                importancias.append(imp / imp.sum())
    oof = pd.concat(previsoes)
    assert not oof.commit_id.duplicated().any()
    oof.to_csv(RESULTADOS / "previsoes_validacao.csv", index=False)
    pd.DataFrame(aprendizado).to_csv(RESULTADOS / "curva_aprendizado.csv", index=False)
    pd.DataFrame({"variavel": FEATURES, "importancia": np.mean(importancias, axis=0)}).to_csv(
        RESULTADOS / "importancia.csv", index=False)
    resumo = {"metricas_validacao_agregada": metricas(oof.buggy, oof.score),
              "matriz_validacao": confusion_matrix(oof.buggy, oof.previsao, labels=[0, 1]).tolist(),
              "n_validacao": len(oof), "n_sem_previsao": len(treino) - len(oof)}
    salvar_json(path, resumo)
    return resumo


def escolher_exemplos(todas):
    """Um caso por célula da matriz, o score mais próximo do limiar e um negativo de score baixo."""
    exemplos = todas.groupby("caso", sort=True).head(1)
    restantes = todas.drop(exemplos.index).assign(distancia=(todas.score - LIMIAR).abs())
    exemplos = pd.concat([exemplos, restantes.sort_values("distancia").head(max(0, 5-len(exemplos)))])
    # Acerto com folga: os anteriores são os primeiros em ordem cronológica e ficam perto do limiar.
    restantes = restantes.drop(exemplos.index, errors="ignore")
    claro = restantes[(restantes.caso == "Verdadeiro negativo") & (restantes.score < 0.1)].head(1)
    return pd.concat([exemplos, claro]).drop(columns="distancia", errors="ignore")


def avaliar_final(config, treino, teste, info):
    path = RESULTADOS / "teste_final.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("Configuração congelada. Reavaliação exploratória do holdout conhecido, uma vez nesta versão.", flush=True)
    pipeline = construir(config["modelo"], config["parametros"])
    pipeline.fit(treino[FEATURES], treino.buggy)
    # A mesma função de inferência será chamada pelo app.
    pred = prever(pipeline, teste[FEATURES])
    resumo = {"metricas": metricas(teste.buggy, pred.score),
              "referencia_majoritaria": metricas(teste.buggy, np.full(len(teste), int(treino.buggy.mode()[0]))),
              "matriz_confusao": confusion_matrix(teste.buggy, pred.previsao, labels=[0, 1]).tolist(),
              "n_teste": len(teste), "avaliacoes_nesta_versao": 1,
              "ressalva": "Período já avaliado na versão inicial; não é teste novo/intocado."}
    todas = pd.concat([teste[["commit_id", "project", "data", *FEATURES, "buggy"]], pred], axis=1)
    todas["caso"] = np.select([
        (todas.buggy == 1) & (todas.previsao == 1), (todas.buggy == 0) & (todas.previsao == 0),
        (todas.buggy == 0) & (todas.previsao == 1)],
        ["Verdadeiro positivo", "Verdadeiro negativo", "Falso positivo"], default="Falso negativo")
    exemplos = escolher_exemplos(todas)
    todas.to_csv(RESULTADOS / "previsoes_teste.csv", index=False)
    artefatos = ROOT / "artefatos"
    artefatos.mkdir(exist_ok=True)
    exemplos.to_csv(artefatos / "exemplos.csv", index=False)
    joblib.dump(pipeline, artefatos / "pipeline.joblib", compress=3)
    pacotes = ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "optuna", "streamlit",
               "matplotlib", "requests", "joblib", "nbformat", "nbclient", "ipykernel", "immutables", "pyarrow"]
    versoes = {p: importlib.metadata.version(p) for p in pacotes}
    meta = {**config, "versoes": versoes, "python": platform.python_version(),
            "versao_extrator": VERSAO_EXTRATOR, "descricoes": DESCRICOES,
            "entrada": {f: {"tipo": "numerico", "aceita_sentinela_menos_um": f in NEGATIVAS} for f in FEATURES},
            "divisao": info, "modelo_ajustado_apenas_no_treino": True,
            "pipeline_sha256": hashlib.sha256((artefatos / "pipeline.joblib").read_bytes()).hexdigest(),
            "score": "Estimativa do modelo; probabilidade não calibrada."}
    salvar_json(artefatos / "metadados.json", meta)
    salvar_json(path, resumo)
    (ROOT / "requirements.txt").write_text("\n".join(f"{p}=={v}" for p,v in versoes.items()) + "\n", encoding="utf-8")
    recarregado = joblib.load(artefatos / "pipeline.joblib")
    iguais = prever(recarregado, exemplos[FEATURES])
    assert np.array_equal(iguais.previsao, exemplos.previsao)
    np.testing.assert_allclose(iguais.score, exemplos.score, rtol=0, atol=1e-12)
    return resumo


def faixas_de_risco():
    """Taxa de positivos por faixa de score nas previsões de validação; é o que o app exibe.

    Usa só validação: não muda o modelo nem o limiar, apenas dá contexto ao score.
    """
    path = ROOT / "artefatos/faixas.csv"
    if path.exists():
        return pd.read_csv(path)
    oof = pd.read_csv(RESULTADOS / "previsoes_validacao.csv")
    nomes, minimos = zip(*FAIXAS)
    grupo = pd.cut(oof.score, [*minimos, np.inf], right=False, labels=list(nomes))
    tabela = oof.groupby(grupo, observed=False).agg(commits=("buggy", "size"), taxa_buggy=("buggy", "mean"))
    tabela = tabela.rename_axis("faixa").reset_index()
    tabela.insert(1, "score_minimo", minimos)
    tabela["taxa_geral"] = oof.buggy.mean()
    path.parent.mkdir(exist_ok=True)
    tabela.to_csv(path, index=False)
    return pd.read_csv(path)


def executar():
    treino, teste, folds, info = preparar()
    comparar_entradas(treino, folds)
    tabela = comparacao(treino, folds)
    config = selecionar(tabela)
    diagnosticar(config, treino, folds)
    faixas_de_risco()
    resultado = avaliar_final(config, treino, teste, info)
    print(json.dumps({"selecao": config, "teste": resultado}, ensure_ascii=False, indent=2))


def comparar_entradas(treino, folds):
    """Ablação com configuração fixa e referência antiga no mesmo conjunto elegível."""
    path = RESULTADOS / "comparacao_entradas.csv"
    if path.exists(): return pd.read_csv(path)
    antigos = {"n_estimators":250,"max_depth":2,"learning_rate":0.18857058875004495,"min_child_weight":10}
    fixos = {"n_estimators":400,"max_depth":5,"learning_rate":0.05,"min_child_weight":5,
             "subsample":0.8,"colsample_bytree":0.8}
    leve = [*GRUPOS["Tamanho"], *GRUPOS["Tipo da alteração"], *GRUPOS["Mensagem"]]
    conjuntos = [("Modelo inicial — mesma cobertura", ["la","ld","nf"], antigos),
                 ("Tamanho — configuração fixa", ["la","ld","nf"], fixos),
                 ("Tamanho + tipo + mensagem", leve, fixos),
                 ("Completo — ancestral", FEATURES, fixos),
                 ("Completo + palavras atuais — sensibilidade", FEATURES + [c for c in treino if c.startswith("p_")], fixos)]
    linhas = []
    for nome, colunas, params in conjuntos:
        print("Comparação de entradas:", nome, flush=True)
        inicio = perf_counter()
        scores = cross_validate(construir("XGBoost",params),treino[colunas],treino.buggy,
                                cv=folds,scoring=pontuar,n_jobs=1,return_train_score=True,error_score="raise")
        linha = resumo_cv("XGBoost",nome,params,scores,perf_counter()-inicio)
        linha["n_entradas"]=len(colunas)
        linhas.append(linha)
    tabela=pd.DataFrame(linhas); tabela.to_csv(path,index=False)
    return tabela


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--desenvolver",action="store_true",help="Compara e congela o modelo, sem avaliar o período final.")
    args=p.parse_args()
    if args.desenvolver:
        treino, _, folds, _ = preparar()
        comparar_entradas(treino,folds)
        config=selecionar(comparacao(treino,folds))
        diagnosticar(config,treino,folds)
        faixas_de_risco()
        print(json.dumps(config,ensure_ascii=False,indent=2))
    else:
        executar()
