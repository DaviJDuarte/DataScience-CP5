# Experimento de novas pistas e busca ampliada

Definido antes da execução desta busca. É uma continuação exploratória: os folds e o
período final já foram vistos em versões anteriores. Nenhum resultado será apresentado
como uma confirmação independente.

1. Manter exatamente os 84.106 commits de treino, os três folds temporais e os 11.292
   commits da avaliação principal até 2018. Relatar 2019 separadamente. Não excluir
   projetos difíceis, corrigir rótulos por previsão ou usar projeto/SHA/autor como entrada.
2. Comparar três conjuntos: 48 pistas originais; essas pistas mais 16 razões determinísticas;
   e o conjunto anterior mais oito medidas da dispersão do diff (72 pistas). Extrair o diff
   com o mesmo reconhecimento de renomeações; exigir paridade de todas as métricas
   estáticas originais para cada commit. Não utilizar as métricas prontas do CSV sem paridade.
3. Comparação controlada: os três conjuntos com os parâmetros do modelo anterior e com
   os parâmetros do melhor piloto de regularização (seis configurações). Assim é possível
   separar o efeito das pistas do efeito dos parâmetros.
4. Busca adicional Optuna TPE, semente 42, sem poda: 24 configurações LightGBM e 12 XGBoost.
   Variar conjunto de pistas, número de árvores (200–800), taxa de aprendizado (0,02–0,15),
   tamanho das árvores, mínimo de amostras/peso da folha, amostragem de linhas/colunas,
   regularização L1/L2 e pesos de classe (nenhum ou balanceado). A configuração exata de
   cada trial fica salva. Pesos balanceados são recalculados apenas no trecho de cada fit.
5. Em cada fold, reservar os últimos 20% das datas únicas do seu treino para escolher o
   limiar de 0,10 a 0,90, passo 0,01, pelo F1. Empate: mais próximo de 0,50; depois maior
   limiar. Reajustar o estimador em todo o treino do fold e avaliar na janela seguinte.
   Não usar a validação externa para early stopping nem para escolher seu próprio limiar.
6. Selecionar a configuração pelo maior F1 médio nos três folds. Desempatar por menor
   desvio, menos pistas e menos árvores. Relatar precision, recall e Average Precision
   para distinguir mudança de alertas de melhoria de ordenação. Preservar o controle
   anterior como candidato: ampliar a busca não obriga trocar o modelo.
7. Congelar a escolha, calcular o limiar final com o mesmo procedimento interno no treino
   completo e ajustar o pipeline final. Só então comparar com os modelos anteriores nos
   mesmos commits do período final. Não continuar a busca por causa desse resultado.
8. Salvar auditoria de partições, previsões por fold, parâmetros, hashes e versões.
   Integrar o candidato selecionado ao notebook e app com as mesmas entradas e limiar.
   Uma comparação por projeto é diagnóstico de cobertura dos rótulos, nunca critério
   para excluir linhas ou inserir identidade do projeto no modelo.

As razões usam zero quando não há denominador; idades sem histórico continuam como
sentinela -1 nas pistas originais. Nas razões por idade, a idade é limitada a zero e soma-se
um dia ao denominador. As medidas do diff usam somente o commit e seu primeiro pai.

Referências: [regularização LightGBM](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html),
[parâmetros XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html),
[ajuste do limiar](https://scikit-learn.org/stable/modules/classification_threshold.html).
