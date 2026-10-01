# Checkpoint 5 — Quais commits revisar primeiro?

Um modelo que olha para um commit e dá um **score de risco de 0 a 1**: quanto maior, maior a chance de
aquele commit ter introduzido um bug. A ideia é ajudar uma equipe a decidir o que revisar e testar primeiro.
Ele não encontra o bug e não substitui a revisão.

O modelo usa 48 pistas de cada commit: tamanho, tipo de arquivo, mensagem, horário, experiência do autor
e histórico dos arquivos alterados. Comparamos Random Forest, XGBoost e LightGBM, ajustados com Grid Search e Optuna.

## Links

- **GitHub:** https://github.com/DaviJDuarte/DataScience-CP5
- **Streamlit:** pendente de publicação

## Integrantes

| Nome | RM |
| --- | --- |
| Beatriz Cortêz Gomes | 561431 |
| Bruno Henrique Campos Alves | 563986 |
| Davi de Jesus Duarte | 566316 |
| Gabriel Augusto Gonçalves Pereira | 564126 |
| Raphaela Oliveira Tatto | 572059 |

## Resultado em resumo

Modelo escolhido: **LightGBM ajustado com Optuna**.

| | Só o tamanho do commit (3 pistas) | Com histórico (48 pistas) |
| --- | --- | --- |
| F1 nos simulados de validação | 0,4703 | **0,5609** |

No período mais recente dos dados, o modelo de 48 pistas teve F1 **0,5229**:
43,5% dos alertas eram bugs e 65,5% dos bugs foram alertados.

Como ler: o F1 vai de 0 a 1 e junta "quantos alertas estavam certos" com "quantos bugs foram encontrados".
Ele **não** é a porcentagem de acertos. O histórico dos arquivos e dos autores melhora o modelo, mas ele continua
errando bastante. Serve para ordenar o que revisar primeiro, não para afirmar que um commit tem bug.

## Rodar o aplicativo

Precisa de Python 3.13. O modelo treinado já vem no projeto; não é preciso treinar nada.

No Windows (PowerShell), dentro da pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No Linux ou macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Abra no navegador o endereço que aparece no terminal. Para encerrar, `Ctrl+C`.

## Usar o aplicativo

**Demonstração local.** Escolha um dos 6 exemplos. Eles vêm do período mais recente, que o modelo não
viu no treino, e incluem acertos e erros. A tela mostra o score, a classe, a faixa de risco e as 48 pistas.
Funciona sem internet.

**GitHub.** Informe um repositório público, liste os commits e escolha um. Para calcular o histórico, o aplicativo
precisa de uma cópia do histórico do repositório. Se ainda não houver, use o botão **Baixar histórico público**
(precisa do Git instalado; em repositórios grandes demora e ocupa espaço). O código do repositório nunca é executado.

- As cópias ficam na pasta `repos/` do projeto. Para usar outra pasta, defina `CP5_REPOS_DIR` antes de abrir o
  aplicativo. No PowerShell: `$env:CP5_REPOS_DIR = "D:\minha_pasta\repos"`.
- `GITHUB_TOKEN` é opcional e só aumenta o limite de consultas à API. Nunca coloque o token no código.
- Merges, commits com arquivos binários e históricos incompletos são recusados com uma mensagem.

## Como o projeto funciona

**1. Dados.** [ApacheJIT v2](https://zenodo.org/records/5907847) (DOI 10.5281/zenodo.5907847, licença CC BY 4.0):
106.674 commits reais de 14 projetos Apache, cada um com a marca `buggy` (1 = introduziu bug, 0 = não).
A marca foi criada pelos autores da base, voltando de correções de bugs até o commit que escreveu as linhas
corrigidas. Ela tem erros, e bugs ainda não corrigidos ficam sem marca.

**2. Pistas.** Para cada commit calculamos 48 números, a partir do histórico Git público dos projetos:

| Grupo | Quantas | Exemplos |
| --- | --- | --- |
| Tamanho | 3 | linhas adicionadas, linhas removidas, arquivos alterados |
| Tipo da alteração | 14 | arquivos de teste, linhas em código de produção |
| Momento | 6 | hora e dia da semana |
| Mensagem | 4 | tamanho da mensagem do commit |
| Autor | 9 | commits anteriores da pessoa, se já mexeu nesses arquivos |
| Histórico dos arquivos | 12 | vezes que os arquivos já foram alterados, por quantas pessoas, mensagens antigas de correção |

**3. O que ficou de fora.** Tudo que entregaria a resposta ou que não existiria na vida real: identificadores
(SHA, e-mail, nome do projeto), as marcas `buggy`/`fix` de outros commits e qualquer informação do futuro.
O histórico de um commit conta só o que veio antes dele na linha principal do repositório.
As palavras da mensagem do próprio commit também ficaram de fora, por cautela.

**4. Limpeza.** 835 commits (0,78%) saíram, quase todos por terem arquivos binários, em que não
dá para contar linhas. A lista está em `resultados/exclusoes.csv`. Sobraram 105.839.

**5. Como testamos.** Os commits são ordenados por data. Os 84.106 mais antigos são o treino; os
21.733 mais recentes (a partir de 06/10/2017) são o período final. Dentro do treino fazemos três
"simulados": o modelo aprende com um trecho do passado e é avaliado no trecho seguinte. Os três algoritmos
usam as mesmas pistas, os mesmos simulados e a mesma métrica (F1, com alerta a partir de score 0,5).

**6. Modelos e ajuste.** Para cada algoritmo: uma versão com configuração padrão (baseline), um Grid Search
com 8 combinações e um Optuna com 20 tentativas. As regras completas, escritas antes da comparação,
estão em [`PROTOCOLO.md`](PROTOCOLO.md).

## Resultados

### As nove configurações

Médias dos três simulados. "Variação" é o desvio entre os simulados; "gap" é F1 de treino menos F1 de validação
(um gap grande indica que o modelo decorou o treino).

| Modelo | Estratégia | F1 treino | F1 validação | Variação | Gap |
| --- | --- | --- | --- | --- | --- |
| Random Forest | Baseline | 1,0000 | 0,5536 | 0,0485 | 0,4464 |
| XGBoost | Baseline | 0,7337 | 0,5371 | 0,0657 | 0,1967 |
| LightGBM | Baseline | 0,7223 | 0,5472 | 0,0587 | 0,1751 |
| Random Forest | Grid Search | 1,0000 | 0,5536 | 0,0485 | 0,4464 |
| Random Forest | Optuna | 0,9803 | 0,5549 | 0,0476 | 0,4254 |
| XGBoost | Grid Search | 0,8162 | 0,5458 | 0,0629 | 0,2703 |
| XGBoost | Optuna | 0,7522 | 0,5514 | 0,0578 | 0,2008 |
| LightGBM | Grid Search | 0,7231 | 0,5526 | 0,0575 | 0,1705 |
| LightGBM | Optuna | 0,7320 | 0,5609 | 0,0515 | 0,1711 |

Escolhido: **LightGBM · Optuna**. Regra: entre as configurações a até 0,005 do melhor F1,
vence a de menor variação, depois menor gap, menos árvores e menor tempo. Parâmetros:

```json
{
  "n_estimators": 250,
  "learning_rate": 0.07902978859187117,
  "num_leaves": 27,
  "min_child_samples": 60
}
```

O Random Forest inicial tira 1,000 no treino e 0,554 na validação: decorou os dados (sobreajuste).
O modelo escolhido tem gap de 0,171, menor, mas não zero. As diferenças entre as nove configurações são
pequenas perto da variação entre os simulados.

### As pistas novas ajudaram?

Mesmos commits e mesmos simulados, mudando só as pistas. As quatro últimas linhas usam um XGBoost de configuração fixa.

| Pistas usadas | Quantidade | F1 | Acurácia | ROC-AUC |
| --- | --- | --- | --- | --- |
| Só tamanho, configuração da 1ª versão | 3 | 0,4703 | 0,7438 | 0,7721 |
| Só tamanho | 3 | 0,4619 | 0,7411 | 0,7693 |
| Tamanho + tipo de arquivo + tamanho da mensagem | 21 | 0,4735 | 0,7539 | 0,7825 |
| Tudo acima + histórico de autores e arquivos | 48 | 0,5509 | 0,7615 | 0,7999 |
| Tudo acima + palavras da mensagem (só para comparar) | 56 | 0,5528 | 0,7622 | 0,8026 |

O ganho vem do histórico de autores e arquivos. As palavras da mensagem acrescentam quase nada e não entram no modelo final.

### Período final

**Atenção:** esse período já tinha sido usado para avaliar a primeira versão do projeto. O resultado abaixo é
uma reavaliação, não uma prova inédita. A configuração foi escolhida antes e não foi mudada depois.

| Métrica | Valor | O que significa |
| --- | --- | --- |
| F1 (principal) | 0,5229 | equilíbrio entre os dois abaixo |
| Precision | 0,4351 | dos alertas, quantos eram bugs |
| Recall | 0,6552 | dos bugs, quantos foram alertados |
| Acurácia | 0,8068 | respostas certas no total |
| ROC-AUC | 0,8367 | qualidade da ordenação (0,5 é sorte; 1 é perfeito) |
| Average Precision | 0,5086 | ordenação olhando para os commits com bug |

Em cada 1.000 commits desse período, 162 têm marca de bug. O modelo alerta 243:
acerta 106 e dá 137 alarmes falsos. Deixa passar 56.

**Por que a acurácia (80,7%) é menor que a de responder sempre "sem bug" (83,8%)?**
Porque poucos commits têm bug, e um alerta só aumenta a acurácia quando mais da metade dos alertas está certa.
A diferença aparece no último ano:

| Ano | Commits | Com marca de bug | Alertados | Acurácia do modelo | Acurácia de sempre "sem bug" |
| --- | --- | --- | --- | --- | --- |
| 2017 | 2.007 | 25,2% | 27,4% | 82,8% | 74,8% |
| 2018 | 9.285 | 21,0% | 26,3% | 79,6% | 79,0% |
| 2019 | 10.441 | 10,1% | 22,0% | 81,3% | 89,9% |

Em 2019 só 10,1% dos commits têm marca de bug, em parte porque bugs recentes ainda não tinham sido
corrigidos quando a base foi fechada. Responder sempre "sem bug" acerta muito e não encontra nenhum bug; por isso
a métrica principal é o F1.

### Faixas de risco mostradas no aplicativo

Calculadas nos simulados de validação, não no período final.

| Faixa | Score | Commits | Tinham marca de bug |
| --- | --- | --- | --- |
| Baixo | a partir de 0,0 | 33.597 | 13,2% |
| Moderado | a partir de 0,2 | 14.943 | 34,0% |
| Alto | a partir de 0,5 | 6.524 | 54,0% |
| Muito alto | a partir de 0,7 | 8.876 | 71,3% |

A média geral nesses commits era 30,3%.

## Reproduzir o experimento

O notebook **`Checkpoint05_resolvido.ipynb`** tem os sete exercícios com as saídas executadas e mostra o código de cada etapa.

```powershell
.\.venv\Scripts\python.exe treinamento.py --desenvolver
.\.venv\Scripts\python.exe treinamento.py
.\.venv\Scripts\python.exe executar_notebook.py
.\.venv\Scripts\python.exe testar_extracao.py
.\.venv\Scripts\python.exe testar_projeto.py
```

- Resultados já calculados são lidos de `resultados/`. `--desenvolver` para antes da avaliação final.
- Para refazer tudo do zero, copie a pasta e apague na cópia os resultados de treinamento, mantendo `resultados/extracao.json`.
- As 48 pistas de todos os commits estão em `artefatos/pistas_completas.parquet` (cerca de 16 MB), então não é
  preciso baixar os repositórios. Se o CSV original não estiver em `data/`, ele é baixado do Zenodo com conferência de checksum.
- Para refazer também a extração das pistas: `python dados.py --clonar --extrair`. Isso baixa 15 repositórios
  públicos e ocupa vários GB.

## Testes

- `testar_projeto.py`: divisão por tempo, modelo salvo, exemplos, buscas completas e aplicativo funcionando sem internet.
- `testar_extracao.py`: o histórico de um commit nunca usa informação do futuro nem de outras branches; renomeações mantêm o histórico.
- A conferência das 48 pistas entre a extração em lote e a de um commit só está em `resultados/paridade_git_real.json`.

## Limitações

- As marcas de bug vêm de um método automático e têm erros.
- O período final já era conhecido da primeira versão.
- Os projetos são da Apache, em Java. Não testamos em outros tipos de projeto.
- "Arquivo de teste" e "mensagem de correção" são decididos por regras simples, que erram em alguns casos.
- O score não é uma probabilidade exata e o limiar ficou fixo em 0,5.

## Arquivos

| Arquivo ou pasta | Conteúdo |
| --- | --- |
| `Checkpoint05_resolvido.ipynb` | Os sete exercícios, com código, saídas e leitura dos resultados |
| `app.py` | Aplicativo Streamlit |
| `extracao.py`, `esquema.py` | Cálculo das 48 pistas, igual no treino e no aplicativo |
| `dados.py`, `treinamento.py`, `modelo.py` | Preparação dos dados, buscas, escolha e previsão |
| `artefatos/` | Modelo salvo, exemplos, faixas de risco e pistas de todos os commits |
| `resultados/` | Tudo o que foi medido: buscas, partições, previsões, exclusões |
| `figuras/` | Gráficos do notebook |
| `PROTOCOLO.md` | Regras do experimento, em linguagem técnica |

## Pequeno glossário

- **Commit:** um pacote de alterações salvo no histórico do projeto.
- **Score e limiar:** o modelo dá um número de 0 a 1; a partir de 0,5 (o limiar) vira alerta.
- **Precision / recall / F1:** alertas certos / bugs encontrados / equilíbrio entre os dois.
- **Validação cruzada temporal:** os três simulados feitos dentro do treino, sempre avaliando em datas posteriores.
- **Sobreajuste (overfitting):** o modelo decora o treino e vai pior em dados novos.
- **Subajuste (underfitting):** falta informação ou capacidade para ir bem até no treino.
- **Hiperparâmetros:** os "botões" do modelo, como número de árvores e profundidade.
- **Grid Search / Optuna:** duas formas de procurar bons hiperparâmetros: cardápio fixo ou tentativas guiadas pelos resultados anteriores.
- **Vazamento (data leakage):** deixar o modelo ver algo que entrega a resposta ou que ele não teria na prática.
