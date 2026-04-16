import os
from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# Função para calcular RMSE
def calcular_rmse(y_real, y_pred):
    return np.sqrt(mean_squared_error(y_real, y_pred))


# Obtém o diretório atual do script (pasta 'codes')
pasta_atual = os.path.dirname(os.path.realpath(__file__))

# Retorna o diretório pai (pasta do projeto)
pasta_projeto = os.path.dirname(pasta_atual)

# Define o pais
countries = ['Brazil', 'China']

# Define o tipo de tarefa
task_type = 'regression'

# Define se a rede fará extração de backbone e o threshold
backbone = True
threshold = 0.01

# Definindo numero de lags e saidas
lags = 14
outs = 14
nameModels = ['GCRN', 'GCLSTM', 'LSTM']

for country in countries:
    print(f"\nProcessando pais: {country}")

    # Define o caminho para a pasta 'results' dentro do projeto
    if backbone:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type,
                                     'backbone_threshold_{:.0f}'.format(threshold * 100))
    else:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type)

    # Verifica se o diretório de pasta_results existe, se não, cria-o
    if not os.path.exists(os.path.join(pasta_results, "Figures")):
        os.makedirs(os.path.join(pasta_results, "Figures"))

    # Lista para armazenar os heatmaps
    heatmaps = []

    for nameModel in nameModels:
        print(f"\nModelo: {nameModel}")

        file_metrics = os.path.join(pasta_results,
                                    f'results_metrics_{nameModel}_{task_type}_{country}_{threshold}.npy') \
            if backbone else os.path.join(pasta_results,
                                          f'results_metrics_{nameModel}_{task_type}_{country}.npy')

        if not os.path.exists(file_metrics):
            # Iterável dos valores de lags e outs
            iteravel = product(range(1, lags + 1), range(1, outs + 1))

            # Inicializa uma lista para armazenar os RMSE médios por lag e out
            heatmap_data = np.zeros((lags, outs), dtype=np.float32)

            for lag, out in iteravel:
                print(f"Lag: {lag}, Out: {out}")

                # Define o caminho para a pasta 'results' dentro do projeto
                pasta_results_model = os.path.join(pasta_results, nameModel, f'lags_{lag}_out_{out}')

                if not os.path.exists(pasta_results_model):
                    print(f"Não existe a pasta {pasta_results_model}")
                    break
                else:
                    if not os.listdir(pasta_results_model):
                        print(f"A pasta {pasta_results_model} está vazia")
                        break
                    else:
                        # Extraindo os números de repetição (rep_num) e usando set para evitar duplicatas
                        npy_files_rep = sorted(
                            set([int(file.split('_')[-1].split('.')[0]) for file in os.listdir(pasta_results_model) if
                                 file.endswith('.npy')]))

                        if npy_files_rep.__len__() == 0:
                            print(f"A pasta {pasta_results_model} não possui arquivos .npy")
                            continue

                        # Carregar os dados em matrizes
                        y_real = []
                        y_pred = []

                        # Carrega todos os pares y_real e y_pred juntos
                        for rep_num in npy_files_rep:
                            y_real_path = os.path.join(pasta_results_model, f'y_real_no_norm_rep_{rep_num}.npy')
                            y_pred_path = os.path.join(pasta_results_model, f'y_pred_no_norm_rep_{rep_num}.npy')

                            # Carrega os arquivos .npy
                            y_real.append(np.load(y_real_path))
                            y_pred.append(np.load(y_pred_path))

                        # Converter as listas em arrays numpy
                        y_real = np.array(y_real)  # Shape: (num_reps, num_tests, num_nodes, num_outputs)
                        y_pred = np.array(y_pred)  # Shape: (num_reps, num_tests, num_nodes, num_outputs)

                        # Calcula o RMSE para cada cidade ao longo das rodadas
                        rmse = calcular_rmse(y_real.mean(axis=(0, 3)), y_pred.mean(axis=(0, 3)))

                        # Calcula a média dos RMSEs para essa combinação de lag e out
                        heatmap_data[lag - 1, out - 1] = rmse.mean()

            # Salva os resultados em um arquivo .npy
            np.save(file_metrics, heatmap_data)
        else:
            heatmap_data = np.load(file_metrics, allow_pickle=True)

        heatmaps.append(heatmap_data)

        # Criar figura individual para cada modelo
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="ticks", font_scale=1.0)

        # Flip the heatmap vertically to show lags in descending order
        flipped_heatmap = np.flipud(heatmap_data)

        # Plot heatmap
        ax = sns.heatmap(flipped_heatmap, annot=False, fmt=".0f", cmap="YlOrBr",
                         xticklabels=range(1, outs + 1),
                         yticklabels=range(lags, 0, -1),
                         cbar=True)

        plt.title(f"Average RMSE {nameModel} - {country}", fontsize=18)
        plt.xlabel("Prediction horizon", fontsize=18)
        plt.ylabel("Window size", fontsize=18)
        plt.tick_params(labelsize=16)

        # Adjust layout
        plt.tight_layout()

        # Salvar a figura individual
        fig_path = os.path.join(pasta_results, "Figures", f"heatmap_rmse_{nameModel}_{country}.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Estatísticas globais (máximo, mínimo, média, desvio padrão, quartis)
        print(f"\nEstatísticas globais para {nameModel} - {country}:")
        print(f"Máximo RMSE: {heatmap_data.max():.2f}")
        print(f"Mínimo RMSE: {heatmap_data.min():.2f}")
        print(f"Média RMSE: {heatmap_data.mean():.2f}")
        print(f"Desvio padrão RMSE: {heatmap_data.std():.2f}")
        print(f"1º quartil RMSE: {np.percentile(heatmap_data, 25):.2f}")
        print(f"Mediana (2º quartil) RMSE: {np.percentile(heatmap_data, 50):.2f}")
        print(f"3º quartil RMSE: {np.percentile(heatmap_data, 75):.2f}")
        print("-" * 40)
