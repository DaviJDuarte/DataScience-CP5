# Checkpoint 5 — Quais commits revisar primeiro?

Classificação de risco de commits com ApacheJIT. O objetivo é equilibrar bugs encontrados e falsos
alarmes para ajudar a priorizar revisão e testes. O modelo não localiza o defeito e o score não é uma
probabilidade calibrada.

**Entrega principal:** [notebook executado](Checkpoint05_resolvido.ipynb), organizado nos sete exercícios.
**Código:** [GitHub](https://github.com/DaviJDuarte/DataScience-CP5). **Aplicativo publicado:** [Streamlit](https://datascience-cp5-davi.streamlit.app/).
Os resultados abaixo são os da versão publicada neste repositório; o aplicativo publicado usa o mesmo modelo e o mesmo limiar.

| Integrante | RM |
|---|---|
| Beatriz Cortêz Gomes | 561431 |
| Bruno Henrique Campos Alves | 563986 |
| Davi de Jesus Duarte | 566316 |
| Gabriel Augusto Gonçalves Pereira | 564126 |
| Raphaela Oliveira Tatto | 572059 |

## Modelo e resultados

Selecionado: **LightGBM**, conjunto **razoes**, pesos **balanced**,
limiar **0,71**. São 48 entradas e 64
características após a transformação no pipeline.

A busca ampliada comparou seis controles e 36 tentativas Optuna (24 LightGBM, 12 XGBoost).
O F1 médio na validação passou de **0,6061** na versão
anterior para **0,6168**. Pesos, parâmetros, conjunto de pistas e limiar foram
escolhidos somente no treino e suas janelas de validação.

Avaliação nos **mesmos 11.292 commits**, de outubro de 2017 até dezembro de 2018:

| Métrica | Original, limiar 0,50 | Versão anterior, limiar ajustado | Modelo final |
|---|---|---|---|
| F1 | 0,5877 | 0,5851 | 0,5843 |
| Precision | 0,5351 | 0,5201 | 0,5520 |
| Recall | 0,6516 | 0,6687 | 0,6206 |
| Average Precision | 0,6210 | 0,6210 | 0,6159 |
| ROC-AUC | 0,8424 | 0,8424 | 0,8376 |

O novo modelo não melhorou o F1 desse período em relação à versão
anterior: diferença **-0,0008**. Encontrou 1.523 commits marcados buggy, gerou 1.236 falsos
alarmes e deixou 931 positivos sem alerta. A diferença para a versão anterior é de
**-118 bugs encontrados** e **-278 falsos alarmes**.

O bootstrap pareado de semanas dá intervalo descritivo de 95% de [-0,0111; 0,0095] para a diferença
de F1. **O período já era conhecido:** o intervalo não corrige adaptações de experimentos anteriores.
É evidência exploratória, não uma confirmação independente. A busca não foi retomada para melhorar esse resultado.

## Dados, 2019 e desbalanceamento

- ApacheJIT v2: 106.674 commits; 835 exclusões por extração incompatível, registradas por motivo.
- Treino: 84.106 commits anteriores ao corte de 06/10/2017, ordenados pela integração no Git.
- Três folds temporais expansivos. Um trecho interno de cada treino escolhe seu limiar sem usar a janela seguinte.
- Avaliação principal: 11.292 commits, 21,7% positivos.
- 2019 separado: 10.441 commits, 10,1% positivos; F1 final 0,4150,
  contra 0,4096 na versão anterior.

2019 nunca fez parte do treino. Separá-lo muda o escopo da avaliação; não melhora o aprendizado.
A mudança na frequência de positivos pode envolver atraso da identificação de bugs ou diferenças
entre commits. Não classificamos todo o ano como dado incorreto.

Testamos pesos iguais e balanceados. No segundo caso, o peso de uma amostra é
`n / (2 × n_da_classe)`, recalculado somente nas linhas usadas em cada ajuste. A validação e a
avaliação conservam suas distribuições reais. Pesos não corrigem rótulos errados nem garantem maior precision.

## O que a auditoria dos rótulos encontrou

Reproduzimos os rótulos dos 106.674 commits a partir do pacote oficial, sem divergências.
Hadoop, HDFS e MapReduce precisam ser lidos juntos: os nomes da fonte misturam repositório e componente.
A família tem 3.060 positivos em
16.192 commits (18,9%).
Corrigimos esse agrupamento nos diagnósticos; o modelo não usa o nome do projeto.

Há 64 positivos de treino com vínculos para revisão.
Entre eles, 11 não têm nenhum relato com resolução Fixed,
e 8 têm uma associação com possível confusão entre número de PR e Jira.
Apenas 1 depende exclusivamente de vínculos sem a chave do bug na mensagem.
As flags não confirmam que esses commits sejam limpos, e não foram usadas para alterar rótulos.

**Limitação temporal:** 3.099 dos
24.338 positivos do treino (12,7%)
têm a primeira correção vinculada depois do corte. As características usam o passado, mas os alvos
são retrospectivos. A avaliação não simula estritamente o que era conhecido em cada data.
A auditoria não alterou o pipeline, os commits avaliados nem o F1.
Veja o método, as evidências e a fila priorizada em [AUDITORIA_ROTULOS.md](AUDITORIA_ROTULOS.md).

## O que foi testado

1. Random Forest, XGBoost e LightGBM: baselines, Grid Search e Optuna, nas mesmas janelas.
2. Pesos de classe e limiar escolhido em validação temporal interna.
3. Um piloto de oito alternativas de regularização e janelas de treino recentes.
4. Comparação controlada de 48 características originais, 64 com razões e 72 com razões e dispersão.
5. Busca ampliada de regularização, tamanho das árvores, amostragem, pesos e características.

As oito pistas do diff foram extraídas do Git para todos os 105.839 commits elegíveis, com paridade
das métricas estáticas originais. Incluem entropia, concentração de linhas, subsistemas e remoções
em testes. As 16 razões são calculadas no mesmo pipeline no treino e no aplicativo.
Não usamos projeto, SHA, identidade do autor ou rótulos futuros como preditores.

Com os mesmos parâmetros, adicionar razões e dispersão levou o F1 de validação de
0,6061 para 0,6142,
e Average Precision de 0,6330 para
0,6525. A escolha final considera o F1 do procedimento completo.
Todos os candidatos ficam registrados, inclusive os que pioraram o resultado.

## Executar

Python 3.13, dentro da pasta do projeto, no PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Para conferir a entrega e executar o notebook:

```powershell
.\.venv\Scripts\python.exe -m unittest testar_extracao testar_balanceamento testar_melhorias testar_projeto testar_rotulos -v
.\.venv\Scripts\python.exe executar_notebook.py
```

Os caches de características e os resultados acompanham o projeto. O notebook funciona sem clonar
repositórios e repete o ajuste final para conferir classe e score com o artefato salvo.
`buscar_melhorias.py` executa ou retoma a busca; `finalizar_melhorias.py` produz ou confere o modelo.
Os comandos rejeitam resultados com assinaturas incompatíveis. Para uma busca diferente, use outra pasta
de experimento e registre o novo protocolo.

No modo GitHub, o app aceita repositórios públicos, exige histórico completo e compara suas contagens
com a API. Use o botão de baixar histórico quando necessário. O cache padrão é `repos/`; para usar outro:

```powershell
$env:CP5_REPOS_DIR = "D:\minha_pasta\repos"
```

`GITHUB_TOKEN` é opcional para aumentar a cota da API. Código dos repositórios não é executado.
Merges, arquivos binários, histórico incompleto e métricas divergentes são recusados.

## Arquivos e limites

| Caminho | Conteúdo |
|---|---|
| `Checkpoint05_resolvido.ipynb` | Explicações, código e saídas dos sete exercícios |
| `PROTOCOLO_MELHORIAS.md` | Regras da busca ampliada definidas antes da execução |
| `resultados/melhorias/` | 42 configurações, folds, seleção, previsões e avaliação |
| `AUDITORIA_ROTULOS.md`, `auditar_rotulos.py` | Achados e reprodução da proveniência dos rótulos |
| `resultados/auditoria_rotulos/` | Evidências Git/Jira, fila de revisão e diagnóstico por família |
| `artefatos/melhorias/` | Pipeline atual, metadados, seis exemplos e faixas de score |
| `novas_pistas.py`, `extrair_dispersao.py` | Transformações e extração compartilhada do diff |
| `resultados/revisao_balanceamento/` | Comparação anterior de pesos e limiar |
| `artefatos/pipeline.joblib` | Modelo original preservado |

`carregar_modelo()` carrega a entrega atual. `carregar_modelo(atual=False)` carrega o original;
o argumento `pasta` permite carregar explicitamente outro conjunto de artefatos.

Rótulos SZZ são retrospectivos e imperfeitos. A reprodução das regras de coleta verifica consistência
interna, não a verdade de cada defeito. O diagnóstico por família preserva todos os registros e todos
os erros. A fila de revisão separa sinais de problema de correções de rótulo efetivamente confirmadas.

O modelo foi treinado com projetos Apache e não demonstra desempenho em projetos FIAP ou em outras
linguagens. Uma avaliação confirmatória exige commits futuros com rótulos independentes.
Slides PDF/PPTX anteriores não foram atualizados; a entrega atual está no notebook e neste README.

Fontes: [ApacheJIT v2](https://zenodo.org/records/5907847),
[LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html),
[XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html),
[limiar de decisão](https://scikit-learn.org/stable/modules/classification_threshold.html).
