import os
from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from scipy.special import expit  # Sigmoid function for log loss

# Obtém o diretório atual do script (pasta 'codes')
pasta_atual = os.path.dirname(os.path.realpath(__file__))

# Retorna o diretório pai (pasta do projeto)
pasta_projeto = os.path.dirname(pasta_atual)

# Define o pais
country = 'Brazil'

# Define o tipo de tarefa
task_type = 'classification'

# Define se a rede fará extração de backbone e o threshold
backbone = True
threshold = 0.01

# Define o caminho para a pasta 'results' dentro do projeto
if backbone:
    pasta_results = os.path.join(pasta_projeto, 'results', country, task_type,
                                 'backbone_threshold_{:.0f}'.format(threshold * 100))
else:
    pasta_results = os.path.join(pasta_projeto, 'results', country, task_type)

# Verifica se o diretório de pasta_results existe, se não, cria-o
if not os.path.exists(os.path.join(pasta_results, "Figures")):
    os.makedirs(os.path.join(pasta_results, "Figures"))

# Definindo numero de lags e saidas
lags = 14
outs = 14
nameModels = ['GCRN', 'GCLSTM']  # 'GCRN', 'GCLSTM'

# Lista das métricas a serem calculadas
metricas = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'Log Loss']
resultados_metricas = {metrica: np.zeros((lags, outs), dtype=np.float32) for metrica in metricas}

# Lista para armazenar os heatmaps
heatmaps = []

for nameModel in nameModels:
    print(f"Modelo: {nameModel}")

    file_metrics = os.path.join(pasta_results,
                                f'results_metrics_{nameModel}_{task_type}_{country}_{threshold}.npy') \
        if backbone else os.path.join(pasta_results,
                                      f'results_metrics_{nameModel}_{task_type}_{country}.npy')

    if not os.path.exists(file_metrics):
        # Iterável dos valores de lags e outs
        iteravel = product(range(1, lags + 1), range(1, outs + 1))

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
                    accuracy, f1, precision, recall, roc_auc, logloss = [], [], [], [], [], []

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
                        y_real_path = os.path.join(pasta_results_model, f'y_real_rep_{rep_num}.npy')
                        y_pred_path = os.path.join(pasta_results_model, f'y_pred_rep_{rep_num}.npy')

                        # Carrega os arquivos .npy
                        y_real.append(np.load(y_real_path))
                        y_pred.append(np.load(y_pred_path))

                    # Converter as listas em arrays numpy
                    y_real = np.array(y_real)  # Shape: (num_reps, num_nodes, num_outputs)
                    y_pred = np.array(y_pred)  # Shape: (num_reps, num_nodes, num_outputs)

                    # Flatten os arrays e classificação binária
                    y_true_flat = y_real.ravel()  # Flatten de todos os y_real
                    y_pred_flat = y_pred.ravel()  # Flatten de todos os y_pred

                    # Classificação binária para y_pred (apenas um threshold de 0.5)
                    y_pred_classes = np.where(y_pred_flat >= 0.5, 1, 0)

                    # Cálculo das métricas uma única vez para todos os dados
                    accuracy = accuracy_score(y_true_flat, y_pred_classes)
                    f1 = f1_score(y_true_flat, y_pred_classes, average='weighted', zero_division=0)
                    precision = precision_score(y_true_flat, y_pred_classes, average='weighted', zero_division=0)
                    recall = recall_score(y_true_flat, y_pred_classes, average='weighted', zero_division=0)

                    # Para ROC AUC, tratamos como problema binário com classes
                    roc_auc = roc_auc_score(y_true_flat, y_pred_classes)

                    # Para Log Loss, aplicamos a função sigmoide em y_pred para transformar em probabilidade
                    y_pred_probs = expit(y_pred_flat)  # Sigmoide para converter em probabilidades entre 0 e 1
                    logloss = log_loss(y_true_flat, y_pred_probs)

                    # Armazenando os resultados para o heatmap
                    resultados_metricas['Accuracy'][lag - 1, out - 1] = np.mean(accuracy)
                    resultados_metricas['F1 Score'][lag - 1, out - 1] = np.mean(f1)
                    resultados_metricas['Precision'][lag - 1, out - 1] = np.mean(precision)
                    resultados_metricas['Recall'][lag - 1, out - 1] = np.mean(recall)
                    resultados_metricas['ROC AUC'][lag - 1, out - 1] = np.mean(roc_auc)
                    resultados_metricas['Log Loss'][lag - 1, out - 1] = np.mean(logloss)

        # Salva os resultados em um arquivo .npy
        np.save(file_metrics, resultados_metricas)
    else:
        resultados_metricas = np.load(file_metrics, allow_pickle=True).item()

    heatmaps.append(resultados_metricas)

# Plotando heatmaps para cada métrica
for metrica in metricas:
    # Criar uma figura com subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(heatmaps), figsize=(15, 6), sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.91, .16, .03, .74])

    vmin = min([heatmap[metrica].min() for heatmap in heatmaps])
    vmax = max([heatmap[metrica].max() for heatmap in heatmaps])

    # Plotar os heatmaps nos subplots
    for i, heatmap in enumerate(heatmaps):
        heatmap = np.flipud(heatmap[metrica])
        sns.set_theme(style="ticks", font_scale=1.0)
        sns.heatmap(heatmap, ax=axes[i], annot=False, fmt=".2f", cmap="YlOrBr", xticklabels=range(1, outs + 1),
                    yticklabels=range(lags, 0, -1), cbar=i == 0, vmin=vmin, vmax=vmax, cbar_ax=None if i else cbar_ax)
        axes[i].set_title(f"Average {metrica} {nameModels[i]}", fontsize=18)
        axes[i].set_xlabel("Prediction horizon", fontsize=18)
        axes[i].set_ylabel("Window size", fontsize=18)
        axes[i].tick_params(labelsize=16)

        # Estatísticas globais (máximo, mínimo, média, desvio padrão, quartis)
        print(f"Estatísticas globais para {nameModels[i]}:")
        print(f"Máximo {metrica}: {heatmap.max():.2f}")
        print(f"Mínimo {metrica}: {heatmap.min():.2f}")
        print(f"Média {metrica}: {heatmap.mean():.2f}")
        print(f"Desvio padrão {metrica}: {heatmap.std():.2f}")
        print(f"1º quartil {metrica}: {np.percentile(heatmap, 25):.2f}")
        print(f"Mediana (2º quartil) {metrica}: {np.percentile(heatmap, 50):.2f}")
        print(f"3º quartil {metrica}: {np.percentile(heatmap, 75):.2f}")
        print("-" * 40)

    # Ajustar o layout e mostrar a figura
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.show()

    # Salvar a figura
    fig.savefig(os.path.join(pasta_results, "Figures", f"heatmap_{metrica.lower()}_pixel.pdf"), dpi=300, bbox_inches='tight')
