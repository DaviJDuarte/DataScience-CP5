# Revisão de 08/10/2026 — classes e limiar

Objetivo: equilibrar bugs encontrados e falsos alarmes pelo F1 positivo. Esta revisão
é posterior à inspeção do holdout original e não produz uma avaliação confirmatória.

1. Manter as 48 entradas, o treino e os três folds temporais originais. Manter os
   hiperparâmetros do LightGBM selecionado anteriormente. A comparação RF/XGBoost/
   LightGBM original continua como referência histórica, sem repetir as buscas.
2. Retirar commits com integração a partir de 01/01/2019 UTC da avaliação principal.
   Preservá-los em relatório de sensibilidade separado. O corte original de treino
   em 06/10/2017 não muda: 2019 nunca esteve no treino. Possível atraso de rótulos
   é uma hipótese, não prova de que todos os dados de 2019 estejam errados.
3. Comparar quatro candidatos: class_weight=None ou balanced, cada um com limiar
   fixo 0,5 ou ajustado. O balanceamento usa n/(2*n_classe), calculado pelo LightGBM
   somente nas linhas do fit. Não reamostrar nem balancear validação ou teste.
4. Para cada fold externo, usar os últimos 20% das datas únicas do respectivo treino
   como validação interna temporal do limiar; ajustar o modelo nos 80% anteriores.
   Buscar limiares de 0,10 a 0,90, passo 0,01, pelo maior F1 interno. Empates:
   mais próximo de 0,5, depois maior limiar. Reajustar no treino externo inteiro e
   medir no fold externo futuro. O limiar nunca usa os rótulos da validação externa.
5. Escolher o candidato pelo maior F1 médio nos três folds externos, depois menor
   desvio, limiar fixo e ausência de pesos em empate exato. Para o candidato ajustado,
   repetir a divisão interna temporal no treino completo para obter o limiar final,
   então reajustar o modelo em todo o treino. Congelar antes de avaliar o holdout.
6. Comparar modelo original e revisão nos mesmos commits até 2018 e, separadamente,
   em 2019+. Relatar F1, precision, recall, acurácia balanceada, acurácia, ROC-AUC,
   Average Precision, prevalência e matriz de confusão. Não atribuir à aprendizagem
   o ganho causado somente pela mudança do período avaliado.
7. Salvar revisão em resultados/revisao_balanceamento e artefatos/revisao_balanceamento.
   O app usa a revisão concluída, inclusive seu limiar, exemplos e faixas; os arquivos
   originais permanecem reproduzíveis. A assinatura cobre código, protocolo, versões,
   configuração, dados e folds; não reutilizar cache de outra revisão.

Os hiperparâmetros do modelo base já foram escolhidos nos três folds; a validação
externa desta revisão isola o ajuste do limiar, mas não é uma avaliação independente
de todo o histórico de seleção. Pesos e limiar não corrigem rótulos errados nem
eliminam mudança de distribuição. Scores permanecem sem calibração probabilística.
