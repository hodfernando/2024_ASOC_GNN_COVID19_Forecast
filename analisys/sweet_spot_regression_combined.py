# -*- coding: utf-8 -*-

import os
from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Ferramenta para criar eixos dedicados para a barra de cores, resolvendo o problema de achatamento
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- 1. FUNÇÕES DE CÁLCULO DE MÉTRICAS ---

def calcular_rmse(y_real, y_pred):
    """Calcula a Raiz do Erro Quadrático Médio (RMSE)."""
    return np.sqrt(mean_squared_error(y_real, y_pred))


def calcular_mae(y_real, y_pred):
    """Calcula o Erro Absoluto Médio (MAE)."""
    return mean_absolute_error(y_real, y_pred)


# --- 2. CONFIGURAÇÃO DO SCRIPT ---

# Determina os caminhos do projeto de forma dinâmica
try:
    # __file__ é definido quando o script é executado como um arquivo .py
    pasta_atual = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Fallback para ambientes interativos como Jupyter Notebooks
    pasta_atual = os.getcwd()

pasta_projeto = os.path.dirname(pasta_atual)

# Parâmetros principais do experimento
countries = ['Brazil', 'China']
task_type = 'regression'
backbone = True
threshold = 0.01
nameModels = ['GCRN', 'GCLSTM', 'LSTM']

# Configurações de grade para os heatmaps
lags = 14  # Número de passos de tempo de entrada (eixo Y do heatmap)
outs = 14  # Número de passos de tempo de saída (eixo X do heatmap)

# --- 3. PROCESSAMENTO DOS DADOS ---

# Loop principal para cada país
for country in countries:
    print(f"\n{'=' * 20} Processando País: {country.upper()} {'=' * 20}")

    # Constrói o caminho para a pasta de resultados com base nas configurações
    if backbone:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type,
                                     'backbone_threshold_{:.0f}'.format(threshold * 100))
    else:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type)

    # Cria a pasta 'Figures' para salvar os gráficos, se ela não existir
    if not os.path.exists(os.path.join(pasta_results, "Figures")):
        os.makedirs(os.path.join(pasta_results, "Figures"))

    # Dicionários para armazenar os dados de todos os modelos para plotagem
    all_models_data_rmse = {}
    all_models_data_mae = {}

    # Loop para cada modelo
    for nameModel in nameModels:
        print(f"\n--- Modelo: {nameModel} ---")

        # Define os nomes dos arquivos .npy que guardam os resultados pré-calculados
        if backbone:
            metric_rmse_path = os.path.join(pasta_results,
                                            f'metric_rmse_{nameModel}_{task_type}_{country}_{threshold}.npy')
            metric_mae_path = os.path.join(pasta_results,
                                           f'metric_mae_{nameModel}_{task_type}_{country}_{threshold}.npy')
        else:
            metric_rmse_path = os.path.join(pasta_results, f'metric_rmse_{nameModel}_{task_type}_{country}.npy')
            metric_mae_path = os.path.join(pasta_results, f'metric_mae_{nameModel}_{task_type}_{country}.npy')

        # Verifica se os arquivos de métricas já existem para evitar reprocessamento
        if not (os.path.exists(metric_rmse_path) and os.path.exists(metric_mae_path)):
            print("Arquivos de métricas não encontrados. Calculando a partir dos dados brutos...")

            # Inicializa as matrizes (heatmaps) com zeros
            heatmap_rmse = np.zeros((lags, outs), dtype=np.float32)
            heatmap_mae = np.zeros((lags, outs), dtype=np.float32)

            # Itera sobre todas as combinações de lag (janela) e out (horizonte)
            iteravel = product(range(1, lags + 1), range(1, outs + 1))
            for lag, out in iteravel:
                print(f"  Processando Lag: {lag}, Out: {out}")

                pasta_results_model = os.path.join(pasta_results, nameModel, f'lags_{lag}_out_{out}')

                if not os.path.exists(pasta_results_model) or not os.listdir(pasta_results_model):
                    print(f"    Aviso: Pasta não encontrada ou vazia para {nameModel} L{lag} O{out}. Pulando.")
                    continue

                # Identifica os números de repetição dos experimentos
                rep_indices = sorted(
                    set([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(pasta_results_model) if
                         'y_real_no_norm_rep' in f]))

                if not rep_indices:
                    print(f"    Aviso: Nenhum arquivo de resultados encontrado em {pasta_results_model}. Pulando.")
                    continue

                # Carrega os dados de todas as repetições
                y_real = np.array(
                    [np.load(os.path.join(pasta_results_model, f'y_real_no_norm_rep_{rep}.npy')) for rep in
                     rep_indices])
                y_pred = np.array(
                    [np.load(os.path.join(pasta_results_model, f'y_pred_no_norm_rep_{rep}.npy')) for rep in
                     rep_indices])

                # Calcula a média dos resultados sobre as repetições (axis=0) e saídas (axis=3)
                # Esta é uma forma de obter uma métrica de erro geral para a configuração
                y_real_mean = y_real.mean(axis=(0, 3))
                y_pred_mean = y_pred.mean(axis=(0, 3))

                # Calcula e armazena as métricas na matriz do heatmap
                heatmap_rmse[lag - 1, out - 1] = calcular_rmse(y_real_mean, y_pred_mean)
                heatmap_mae[lag - 1, out - 1] = calcular_mae(y_real_mean, y_pred_mean)

            # Salva as matrizes calculadas para uso futuro
            print(f"Salvando métricas em arquivos .npy...")
            np.save(metric_rmse_path, heatmap_rmse)
            np.save(metric_mae_path, heatmap_mae)
        else:
            # Carrega as métricas se os arquivos já existirem
            print("Carregando métricas pré-calculadas...")
            heatmap_rmse = np.load(metric_rmse_path)
            heatmap_mae = np.load(metric_mae_path)

        # Armazena os dados nos dicionários principais
        all_models_data_rmse[nameModel] = heatmap_rmse
        all_models_data_mae[nameModel] = heatmap_mae

    # --- 4. GERAÇÃO DOS HEATMAPS E ESTATÍSTICAS ---

    # Agrupa os dados e nomes das métricas para gerar os gráficos em um loop
    metric_data_groups = {
        'RMSE': all_models_data_rmse,
        'MAE': all_models_data_mae
    }

    for metric_name, all_models_data in metric_data_groups.items():
        print(f"\nGerando heatmap e estatísticas para a métrica: {metric_name}")

        # Cria a figura que conterá os subplots lado a lado
        fig, axes = plt.subplots(1, len(nameModels), figsize=(6.5 * len(nameModels), 6), sharey=True)
        if len(nameModels) == 1: axes = [axes]  # Garante que 'axes' seja sempre iterável

        # Determina o valor mínimo e máximo em todos os modelos para a métrica atual
        # Isso garante que a escala de cores seja a mesma para todos os subplots, permitindo uma comparação justa.
        vmin = min(data.min() for data in all_models_data.values() if data.size > 0)
        vmax = max(data.max() for data in all_models_data.values() if data.size > 0)

        # Loop para plotar cada heatmap
        for i, (nameModel, data) in enumerate(all_models_data.items()):
            ax = axes[i]
            # Inverte a matriz verticalmente para que a janela de tamanho 1 fique na base do gráfico
            flipped_heatmap = np.flipud(data)

            # Define se a anotação dos valores deve ser ligada.
            # Para heatmaps muito grandes (ex: 30x30), é melhor definir como False.
            show_annotations = False

            # --- CORREÇÃO DO TAMANHO DO HEATMAP ---
            # Apenas para o último subplot, criamos um eixo dedicado para a barra de cores.
            # Isso impede que o heatmap principal seja "achatado".
            if i == len(nameModels) - 1:
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%",
                                              pad=0.1)  # 'size' e 'pad' controlam o tamanho da barra e o espaçamento
                sns.heatmap(flipped_heatmap, ax=ax, annot=show_annotations, fmt=".2f", cmap="YlOrBr",
                            vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax)
            else:
                sns.heatmap(flipped_heatmap, ax=ax, annot=show_annotations, fmt=".2f", cmap="YlOrBr",
                            vmin=vmin, vmax=vmax, cbar=False)

            # Define os títulos e rótulos
            ax.set_title(nameModel, fontsize=16)
            ax.set_xlabel("Prediction horizon", fontsize=14)
            # O rótulo do eixo Y aparece apenas no primeiro gráfico
            if i == 0:
                ax.set_ylabel("Window size", fontsize=14)

            # Define os rótulos dos eixos (ticks)
            ax.set_xticklabels(range(1, outs + 1))
            ax.set_yticklabels(range(lags, 0, -1))
            ax.tick_params(labelsize=12)

        # Ajusta o layout para garantir que nada se sobreponha
        plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adiciona um pequeno ajuste para a barra de cores

        # Salva a figura combinada
        fig_path = os.path.join(pasta_results, "Figures", f"{country.lower()}_heatmap_{metric_name.lower()}.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Heatmap salvo em: {fig_path}")

        # --- ESTATÍSTICAS GLOBAIS ---
        print(f"\n--- Estatísticas Globais para {metric_name} - {country} ---")
        for nameModel, data in all_models_data.items():
            if data.size > 0:
                print(f"\nModelo: {nameModel}")
                print(f"  - Máximo: {data.max():.4f}")
                print(f"  - Mínimo: {data.min():.4f}")
                print(f"  - Média: {data.mean():.4f}")
                print(f"  - Desvio Padrão: {data.std():.4f}")
                print(
                    f"  - Quartis (25%, 50%, 75%): {np.percentile(data, 25):.4f}, {np.median(data):.4f}, {np.percentile(data, 75):.4f}")
        print("-" * 50)