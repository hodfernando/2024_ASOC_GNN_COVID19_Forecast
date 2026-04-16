import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# Função para adicionar resultados ao dicionário
def adicionar_resultados(modelo, resultados, situacao, y_real, y_pred):
    rmse = np.array(
        [np.sqrt(mean_squared_error(y_real[ind], y_pred[ind])) for ind in range(y_real.shape[0])])
    resultados['Modelo'].append(modelo)
    resultados['Situação'].append(situacao)
    resultados['RMSE Max'].append(rmse.max())
    resultados['RMSE Min'].append(rmse.min())
    resultados['RMSE Mean'].append(rmse.mean())
    resultados['RMSE Std'].append(rmse.std())


# Obtém o diretório atual
pasta_atual = os.path.dirname(os.path.realpath(__file__))

# Retorna o diretório pai (pasta do projeto)
pasta_projeto = os.path.dirname(pasta_atual)

# Define o caminho para a pasta 'results' dentro do projeto
pasta_results = os.path.join(pasta_projeto, 'results', 'Brazil', 'regression')

# Modelos a serem processados
modelos = ['GCLSTM', 'GCRN']

# Dicionário para armazenar os resultados
resultados_tempo = {
    'Modelo': [],
    'Situação': [],
    'RMSE Max': [],
    'RMSE Min': [],
    'RMSE Mean': [],
    'RMSE Std': [],
}

for modelo in modelos:
    # Caminho para os dados do Bracis
    pasta_bracis = os.path.join(pasta_results, 'Bracis', modelo)
    y_pred_bracis = np.load(os.path.join(pasta_bracis, 'y_pred_no_norm_2020-2022.npy')).mean(axis=0)
    y_real_bracis = np.load(os.path.join(pasta_bracis, 'y_real_no_norm_2020-2022.npy')).mean(axis=0)

    # Caminho para os dados sem janela deslizante
    pasta_sem_janela = os.path.join(pasta_results, 'Sem janela', modelo, 'lags_14_out_1')
    y_pred_no_window = np.load(os.path.join(pasta_sem_janela, 'y_pred_no_norm.npy')).mean(axis=(0, 3))
    y_real_no_window = np.load(os.path.join(pasta_sem_janela, 'y_real_no_norm.npy')).mean(axis=(0, 3))

    # Caminho para os dados com janela deslizante
    pasta_com_janela = os.path.join(pasta_results, modelo, 'lags_14_out_1')
    y_pred_window = np.load(os.path.join(pasta_com_janela, 'y_pred_no_norm.npy')).mean(axis=(0, 3))
    y_real_window = np.load(os.path.join(pasta_com_janela, 'y_real_no_norm.npy')).mean(axis=(0, 3))

    # Caminho para os dados com janela deslizante e backbone
    pasta_backbone = os.path.join(pasta_results, 'backbone_threshold_1', modelo, 'lags_14_out_1')
    y_pred_backbone_list = []
    y_real_backbone_list = []

    # Carregar todos os arquivos de repetição dinamicamente
    for file in os.listdir(pasta_backbone):
        if file.startswith('y_pred_no_norm_rep_') and file.endswith('.npy'):
            y_pred_backbone_list.append(np.load(os.path.join(pasta_backbone, file)))
        elif file.startswith('y_real_no_norm_rep_') and file.endswith('.npy'):
            y_real_backbone_list.append(np.load(os.path.join(pasta_backbone, file)))

    # Calcular a média dos arrays carregados
    y_pred_backbone = np.mean(y_pred_backbone_list, axis=(0, 3))
    y_real_backbone = np.mean(y_real_backbone_list, axis=(0, 3))

    # Adiciona os resultados ao dicionário
    adicionar_resultados(modelo, resultados_tempo, 'Bracis', y_real_bracis, y_pred_bracis)
    adicionar_resultados(modelo, resultados_tempo, 'Sem Janela', y_real_no_window, y_pred_no_window)
    adicionar_resultados(modelo, resultados_tempo, 'Com Janela', y_real_window, y_pred_window)
    adicionar_resultados(modelo, resultados_tempo, 'Backbone', y_real_backbone, y_pred_backbone)

# Cria o DataFrame
df_resultados_tempo = pd.DataFrame(resultados_tempo)

# Mostra a tabela
print(df_resultados_tempo)
print(50 * '-')

# Salva os resultados em um arquivo CSV
df_resultados_tempo.to_csv(os.path.join(pasta_atual, 'result_compare_teste.csv'), index=False)
