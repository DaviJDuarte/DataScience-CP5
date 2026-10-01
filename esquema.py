"""As mesmas 48 entradas no treinamento e na aplicação, com ordem explícita."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42
LIMIAR = 0.5
VERSAO_EXTRATOR = "ancestral-primeiro-pai-v1"
GRUPOS = {
    "Tamanho": {"la": "Linhas adicionadas", "ld": "Linhas removidas", "nf": "Arquivos afetados"},
    "Tipo da alteração": {
        "o_n_java": "Arquivos Java", "o_n_teste": "Arquivos de teste", "o_n_producao": "Arquivos Java fora de testes",
        "o_n_outros": "Arquivos não Java", "o_tem_teste": "Presença de teste",
        "o_frac_la_teste": "Fração das adições em testes", "o_la_producao": "Adições em Java de produção",
        "o_ld_producao": "Remoções em Java de produção", "o_n_extensoes": "Extensões distintas",
        "o_n_pastas": "Pastas distintas", "o_n_binarios": "Arquivos binários (alvos recusados)",
        "o_n_renomeados": "Arquivos renomeados", "o_max_la_arquivo": "Maior adição em um arquivo",
        "o_frac_remocao": "Fração de linhas removidas"},
    "Momento": {
        "q_hora": "Hora local de autoria", "q_dia_semana": "Dia da semana (segunda=0)",
        "q_fim_semana": "Fim de semana", "q_madrugada": "Entre 22h e 6h",
        "q_fuso_h": "Fuso do autor em horas", "q_atraso_commit_h": "Intervalo autoria-integração (horas)"},
    "Mensagem": {"m_caracteres": "Caracteres da mensagem", "m_palavras": "Palavras da mensagem",
                 "m_linhas": "Linhas da mensagem", "m_assunto_caracteres": "Caracteres do assunto"},
    "Autor": {
        "w_autor_diferente_committer": "Autor diferente do integrador",
        "w_autor_commits": "Eventos anteriores do autor na cadeia ancestral",
        "w_autor_dias_desde_primeiro": "Dias desde primeiro evento do autor",
        "w_autor_dias_desde_ultimo": "Dias desde último evento do autor",
        "w_autor_commits_7d": "Eventos anteriores do autor em sete dias",
        "w_autor_nos_arquivos": "Mudanças anteriores do autor nos arquivos",
        "w_autor_frac_arquivos_conhecidos": "Fração de arquivos já alterados pelo autor",
        "w_autor_nas_pastas": "Eventos anteriores do autor nas pastas",
        "w_autor_frac_correcoes": "Fração de mensagens anteriores do autor com indício de correção"},
    "Histórico dos arquivos": {
        "l_n_arquivos_novos": "Arquivos sem histórico anterior",
        "l_frac_arquivos_novos": "Fração de arquivos sem histórico",
        "l_arq_mudancas_max": "Máximo de mudanças anteriores por arquivo",
        "l_arq_mudancas_media": "Média de mudanças anteriores (arquivos conhecidos)",
        "l_arq_idade_media_d": "Idade média do histórico dos arquivos, em dias",
        "l_arq_dias_desde_ultima_min": "Menor intervalo desde a última mudança",
        "l_arq_dias_desde_ultima_media": "Intervalo médio desde a última mudança",
        "l_arq_autores_max": "Máximo de autores anteriores por arquivo",
        "l_arq_autores_media": "Média de autores anteriores por arquivo",
        "l_arq_correcoes_max": "Máximo de mensagens anteriores com indício de correção",
        "l_arq_correcoes_media": "Média de mensagens anteriores com indício de correção",
        "l_arq_frac_correcoes": "Fração média de mensagens anteriores com indício de correção"},
}
DESCRICOES = {k: v for grupo in GRUPOS.values() for k, v in grupo.items()}
FEATURES = list(DESCRICOES)
NEGATIVAS = {"q_fuso_h", "w_autor_dias_desde_primeiro", "w_autor_dias_desde_ultimo",
             "l_arq_idade_media_d", "l_arq_dias_desde_ultima_min", "l_arq_dias_desde_ultima_media"}
