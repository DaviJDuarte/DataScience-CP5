"""Gráficos do relatório. Exploração usa somente treino; desempenho é identificado."""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
from esquema import ROOT, DESCRICOES

PASTA = ROOT / "figuras"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold"})


def salvar(fig, nome):
    PASTA.mkdir(exist_ok=True)
    fig.savefig(PASTA / f"{nome}.png", dpi=160, bbox_inches="tight")
    return fig


def pct(x):
    return f"{x:.1%}".replace(".", ",")


def mil(x):
    return f"{int(x):,}".replace(",", ".")


def taxas(treino, coluna, limites, nomes):
    grupo = pd.cut(treino[coluna], limites, labels=nomes, include_lowest=True)
    return treino.groupby(grupo, observed=False).buggy.agg(commits="size", taxa="mean")


def explorar(treino):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    classes = treino.buggy.value_counts().reindex([0, 1])
    ax[0].bar(["Sem marca de bug", "Com marca de bug"], classes, color=["#4d7ea8", "#df8058"])
    ax[0].set(title="1 · Classes no treino", ylabel="Commits")
    for i, n in enumerate(classes): ax[0].text(i, n, f"{mil(n)}\n{pct(n/len(treino))}", ha="center", va="bottom")
    ax[0].set_ylim(0, classes.max()*1.2)
    ax[1].boxplot([np.log1p(treino.loc[treino.buggy.eq(v), "la"]+treino.loc[treino.buggy.eq(v), "ld"])
                   for v in [0, 1]], tick_labels=["Sem marca de bug", "Com marca de bug"], showfliers=False)
    ax[1].set(title="2 · Volume da alteração", ylabel="log(1 + adições + remoções)")
    t = taxas(treino, "l_arq_mudancas_max", [-1, 0, 9, 49, np.inf], ["0", "1–9", "10–49", "50+"])
    ax[2].bar(t.index.astype(str), t.taxa*100, color="#4d7ea8")
    for i, r in enumerate(t.itertuples()): ax[2].text(i, r.taxa*100, pct(r.taxa), ha="center", va="bottom")
    ax[2].set(title="3 · Histórico dos arquivos", xlabel="Máximo de mudanças anteriores", ylabel="Commits com marca de bug (%)", ylim=(0, max(55,t.taxa.max()*120)))
    return salvar(fig, "01_exploracao")


def historico(treino):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    especificacoes = [("l_arq_correcoes_max", [-1,0,4,14,np.inf], ["0","1–4","5–14","15+"], "Mensagens de correção já vistas nos arquivos"),
                      ("o_la_producao", [-1,0,49,199,999,np.inf], ["0","1–49","50–199","200–999","1.000+"], "Linhas Java de produção adicionadas")]
    for eixo, (c,bins,labels,titulo) in zip(ax, especificacoes):
        t = taxas(treino,c,bins,labels)
        eixo.bar(t.index.astype(str),t.taxa*100,color="#4d7ea8")
        for i,r in enumerate(t.itertuples()): eixo.text(i,r.taxa*100,f"{pct(r.taxa)}\nn={mil(r.commits)}",ha="center",va="bottom",fontsize=8)
        eixo.set(title=titulo,ylabel="Commits com marca de bug (%)",ylim=(0,max(70,t.taxa.max()*125)))
    return salvar(fig,"02_historico")


def comparar(tabela):
    t=tabela.sort_values("f1_validacao")
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    ax.barh(t.modelo+" · "+t.estrategia,t.f1_validacao,xerr=t.desvio_f1,color="#4d7ea8",capsize=3)
    ax.set(xlim=(0,0.8),xlabel="F1 médio de validação ± variação entre os três simulados",title="As nove configurações nos mesmos simulados")
    return salvar(fig,"03_comparacao")


def diagnosticos():
    p=ROOT/"resultados"
    curva=pd.read_csv(p/"curva_aprendizado.csv")
    imp=pd.read_csv(p/"importancia.csv").nlargest(12,"importancia").sort_values("importancia")
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout="constrained")
    for campo,nome in [("f1_treino","Treino"),("f1_validacao","Validação")]:
        t=curva.groupby("fracao")[campo].agg(["mean","std"])
        ax[0].errorbar(t.index*100,t["mean"],yerr=t["std"],marker="o",capsize=4,label=nome)
    ax[0].set(xlabel="Parte do passado usada no treino (%)",ylabel="F1",ylim=(0,1),title="Curva de aprendizado")
    ax[0].legend()
    ax[1].barh(imp.variavel.map(DESCRICOES),imp.importancia,color="#4d7ea8")
    ax[1].set(title="As 12 pistas mais usadas pelo modelo",xlabel="Importância para o modelo (não indica causa)")
    return salvar(fig,"04_diagnosticos")


def validacao():
    oof=pd.read_csv(ROOT/"resultados/previsoes_validacao.csv")
    fig,ax=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    ConfusionMatrixDisplay.from_predictions(oof.buggy,oof.previsao,labels=[0,1],display_labels=["Sem bug","Com bug"],ax=ax[0],colorbar=False,cmap="Blues")
    ax[0].set(title="Validação (três simulados) · alerta a partir de 0,5",xlabel="Previsão do modelo",ylabel="Marca na base")
    PrecisionRecallDisplay.from_predictions(oof.buggy,oof.score,ax=ax[1],name="Modelo escolhido",plot_chance_level=True)
    objetos,nomes=ax[1].get_legend_handles_labels()
    ax[1].legend(objetos,[n.replace('Chance level','Alertar ao acaso') for n in nomes])
    ax[1].set(title="Curva precision–recall na validação",xlabel="Recall",ylabel="Precision")
    return salvar(fig,"05_validacao")


def final():
    resumo=json.loads((ROOT/"resultados/teste_final.json").read_text(encoding="utf-8"))
    fig,ax=plt.subplots(figsize=(5,4),layout="constrained")
    ConfusionMatrixDisplay(np.array(resumo["matriz_confusao"]),display_labels=["Sem bug","Com bug"]).plot(ax=ax,colorbar=False,cmap="Blues")
    ax.set(title="Período final (reavaliação) · alerta a partir de 0,5",xlabel="Previsão do modelo",ylabel="Marca na base")
    return salvar(fig,"06_periodo_final")
