# CP5 — Auditoria da origem dos rótulos

A auditoria explicou a distribuição estranha do Hadoop e encontrou ligações que merecem revisão.
Os 106.674 rótulos publicados foram reproduzidos exatamente, mas isso confirma a aplicação das regras
de coleta, não a verdade de cada defeito. Também encontramos uma limitação na avaliação temporal:
parte das marcas positivas depende de correções posteriores ao corte de treino.

O notebook agora usa o agrupamento correto nos diagnósticos e explica essa limitação.
O CSV original, as características, o pipeline e as previsões foram preservados. O F1 da avaliação
principal continua 0,5843. Esta é uma auditoria posterior aos experimentos, sem nova seleção de modelo.

## 1. O que foi conferido

Usamos o [pacote oficial ApacheJIT v2](https://zenodo.org/records/5907847), os notebooks e o coletor
incluídos nele, e os objetos dos repositórios Git locais. O programa `auditar_rotulos.py` lê esses
arquivos e reimplementa as regras; não executa código do pacote nem dos repositórios examinados.

| Verificação | Resultado |
|---|---:|
| Commits publicados | 106.674 |
| Positivos / negativos | 28.239 / 78.435 |
| SHAs repetidos | 0 |
| Divergências em `buggy` ou `fix` | 0 |
| Commits sem proveniência | 0 |
| Associações relato → correção conferidas no Git | 44.202 |
| Objetos Git ausentes | 0 |
| Rótulos alterados | 0 |

As datas de correção de todos os vínculos aceitos foram conferidas com o timestamp do committer
no Git. Isso permite examinar disponibilidade temporal sem depender do fuso do export Jira.

Os rótulos positivos vêm do SZZ: correção → linhas alteradas → commits anteriores que escreveram
essas linhas. A reprodução aplica a ordem temporal, limites de quantidade de vínculos e filtro
de alterações sintáticas (AST) usados pelos autores. Os limites calculados são 13 candidatos por
correção e 8 correções por candidato. Uma alteração pode corrigir um problema e introduzir outro;
por isso `fix=True` não implica `buggy=False`.

## 2. Hadoop: o problema era comparar nomes de origens diferentes

| Nome no CSV | Commits | Positivos |
|---|---:|---:|
| `apache/hadoop` | 11.964 | 0 |
| `apache/hadoop-hdfs` | 2.907 | 2.222 |
| `apache/hadoop-mapreduce` | 1.321 | 838 |
| **Família Hadoop** | **16.192** | **3.060 (18,9%)** |

Na coleta dos positivos, o nome acompanha o projeto do relato, como HDFS ou MAPREDUCE.
Muitos negativos residuais acompanham o nome do repositório compartilhado, Hadoop. A coluna final
não permite reconstruir com segurança o componente de cada negativo. Somamos os três nomes em
uma família para o relatório, sem inventar componentes individuais.

O artigo apresenta 14 projetos; o CSV final tem 15 nomes e o relatório agrupado tem 13 famílias.
São convenções diferentes de agrupamento. A soma Hadoop coincide com a soma HDFS + MapReduce
da tabela do artigo. O nome do projeto nunca entra como característica do classificador.

Na validação externa, a família reúne 10.169 commits, com precision 0,3314, recall 0,7335 e
F1 0,4565. Há 3.809 falsos alarmes e 686 positivos não alertados. O problema de desempenho continua;
somar os grupos não muda nenhuma previsão nem o F1 global. A tabela com os nomes originais foi
preservada em `resultados/melhorias/erros_por_projeto.csv`.

## 3. Ligações suspeitas: revisar a evidência antes de trocar a marca

Há **64 commits positivos para revisão, todos no treino**. As categorias abaixo se sobrepõem.

| Sinal | Commits positivos | Interpretação |
|---|---:|---|
| Nenhum relato associado com resolução `Fixed` | 11 | Divergência de metadados frente ao critério descrito no artigo |
| Vínculo sem a chave do bug, com possível confusão entre Jira e PR | 8 | A mensagem cita outro bug e um `#número` igual ao número do relato coletado |
| Correção mais de um dia antes da abertura do relato | 46 | Datas a conferir; migração de tracker ou datas Git incomuns podem explicar |

No conjunto completo de 44.202 associações, 47 mensagens não contêm a chave nem uma variação
de pontuação; 43 têm o padrão de possível colisão com número de PR. Apenas cinco dessas
associações alcançam os oito positivos acima depois dos filtros SZZ/AST. Sete desses oito commits
têm outra associação que cita a chave correspondente. Um depende exclusivamente do vínculo suspeito.

Exemplo prioritário: o commit
[`0c1c000a8bbd7ba9da9f4b528554fd06c3b0f400`](https://github.com/apache/hive/commit/0c1c000a8bbd7ba9da9f4b528554fd06c3b0f400)
recebe marca positiva por uma correção associada a `HIVE-2107` no pacote. A mensagem da
[correção `b4b353e4`](https://github.com/apache/hive/commit/b4b353e4a9756fc3a008fc60445c7ec433e61065)
cita `HIVE-24886` e encerra o PR `apache/hive#2107`. Isso sustenta a suspeita de associação errada
entre identificadores. Não demonstra, por si só, que o commit antigo não introduziu nenhum bug.

Variações como `HBASE=3755` e `HDFS:1699` são aceitas na conferência. A ausência do hífen não
é tratada como erro. A comparação de datas Jira tolera um dia, pois o fuso do export não está
declarado; flags de cronologia continuam sendo indicações para revisão, não erros confirmados.
As resoluções são as do pacote histórico, não uma consulta ao estado atual do Jira.

Arquivos para conferir os casos:

- [`fila_commits.csv`](resultados/auditoria_rotulos/fila_commits.csv): uma linha por commit, com prioridade,
  partição e flags. O caso que só tem vínculo sem chave aparece primeiro.
- [`positivos_para_revisao.csv`](resultados/auditoria_rotulos/positivos_para_revisao.csv): todas as evidências
  desses commits, inclusive ligações alternativas, assuntos e URLs do Git/Jira.
- [`vinculos_para_revisao.csv`](resultados/auditoria_rotulos/vinculos_para_revisao.csv): ligações sinalizadas
  na coleta completa, incluindo as que não chegaram aos positivos publicados.

O nome “evidência sem alerta” significa apenas que esses testes não sinalizaram o vínculo. Não é
certificação de correção semântica. Não removemos exemplos para favorecer o modelo nem inferimos
novas marcas a partir dos seus scores.

## 4. Os 334 negativos que apareceram como candidatos SZZ

São candidatos brutos rejeitados pelas regras dos autores: 71 por ordem temporal, 260 pelo limite
de candidatos por correção e 3 pelo limite de correções por candidato. Nenhum negativo publicado
sobrevive a todos os filtros. Não encontramos contradição interna que justifique convertê-los
automaticamente em positivos.

O SZZ pode gerar candidatos incorretos, e filtros também podem descartar bugs reais. Portanto,
esses 334 casos são outra fila de investigação, registrada em
[`negativos_candidatos_descartados.csv`](resultados/auditoria_rotulos/negativos_candidatos_descartados.csv).
Os demais negativos também não têm uma certificação de ausência de defeito.

## 5. A limitação mais relevante para interpretar a validação

“Commit anterior ao corte” não significa “rótulo conhecido no corte”. Comparamos a primeira
correção vinculada aceita de cada positivo com o último commit de cada conjunto de treino.

| Trecho de aprendizado | Positivos | Primeira correção vinculada depois do corte | Fração |
|---|---:|---:|---:|
| Fold 1, até 21/03/2011 | 4.965 | 2.618 | 52,7% |
| Fold 2, até 21/02/2014 | 12.036 | 3.322 | 27,6% |
| Fold 3, até 10/09/2015 | 17.998 | 4.163 | 23,1% |
| Treino final, até 06/10/2017 | 24.338 | 3.099 | 12,7% |

Exemplo: um commit escrito em 2016, rotulado por uma correção de 2018, pertence ao treino cronológico
de 2017, mas essa evidência de seu alvo ainda não existia no corte. A primeira correção vinculada
no pacote não garante a data real da primeira descoberta; outras evidências podem não ter sido coletadas.

As características continuam limitadas ao commit e seu histórico anterior. Porém, os alvos usados
no ajuste são retrospectivos. Os F1 registrados são válidos como medidas dessa tarefa retrospectiva;
não demonstram o desempenho de um sistema treinado apenas com rótulos disponíveis naquela data.
Essa limitação é independente de o período de avaliação já ter sido consultado no desenvolvimento.

Separar 2019 ou configurar pesos de classe não resolve esse problema. Também seria incorreto
apagar só os positivos com correção futura ou convertê-los diretamente em negativos.

## 6. O que pode melhorar a próxima versão

O próximo experimento defensável é reconstruir rótulos respeitando a data em que a evidência ficou
disponível, com a mesma janela de observação para as duas classes, e revisar os vínculos prioritários.
Isso exige tratar commits ainda sem tempo de acompanhamento como casos não resolvidos, e estabelecer
outro protocolo antes de escolher modelos. As datas de correções futuras servem à auditoria e à
rotulagem, nunca como entradas do classificador. Uma avaliação confirmatória precisa de novos dados.

A auditoria não prova que o modelo chegou ao limite. Ela mostra problemas concretos de origem e
avaliação que uma nova busca de hiperparâmetros não corrige. A pequena fila de vínculos suspeitos,
por si só, também não sustenta a promessa de um grande aumento no F1.

## Reprodução e rastreabilidade

O notebook lê os resultados persistidos. Para repetir a auditoria completa, use o ZIP oficial v2
e os mesmos clones bare usados na extração; informe os caminhos locais:

```powershell
.\.venv\Scripts\python.exe auditar_rotulos.py --pacote "C:\dados\apachejit_dataset_replication.zip" --repos "D:\dados\repos"
.\.venv\Scripts\python.exe -m unittest testar_rotulos -v
```

`carregar_auditoria()` confere o hash do código, dos arquivos produzidos, da base, do pipeline e
das previsões de validação. O manifesto registra também os hashes dos arquivos originais de coleta.

- MD5 do pacote: `528bf0ee04b15976be6bf15f8efddc65`.
- SHA-256 do CSV: `5097cbbbab8c4611a709b3cf95ac5c9fe5f62e619bab081911a2a247ddbed4e0`.
- SHA-256 do pipeline antes/depois: `380e3ae6e03ab8439a49a694391afcc21d5c035b7ae26ea5fb3074100afe0054`.
- Etapas e contagens: `resultados/auditoria_rotulos/etapas.csv`.
- Proveniência por commit e evidências completas: `proveniencia.parquet` e `evidencias_positivos.parquet` na mesma pasta.

As contagens deste relatório foram calculadas dos arquivos locais. O método da base é descrito no
[artigo ApacheJIT](https://arxiv.org/html/2203.00101) e no
[código dos autores](https://github.com/hosseinkshvrz/apachejit).
A [issue upstream #1](https://github.com/hosseinkshvrz/apachejit/issues/1) levanta uma dúvida sobre
um filtro da construção; ela não foi tratada como uma errata confirmada nem como autorização para
mudar as regras da fonte.
