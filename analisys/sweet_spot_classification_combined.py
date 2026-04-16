# -*- coding: utf-8 -*-

import os
from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from scipy.special import expit  # Função sigmoide para a Log Loss
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Ferramenta para corrigir o tamanho do heatmap

# --- 1. CONFIGURAÇÃO DO SCRIPT ---

# Determina os caminhos do projeto de forma dinâmica
try:
    # __file__ é definido quando o script é executado diretamente
    pasta_atual = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Fallback para ambientes interativos (como Jupyter Notebooks)
    pasta_atual = os.getcwd()

pasta_projeto = os.path.dirname(pasta_atual)

# Parâmetros principais do experimento
countries = ['Brazil', 'China']
task_type = 'classification'
backbone = True
threshold = 0.01

# Configurações de janela (lags) e horizonte (outs)
lags = 14
outs = 14
nameModels = ['GCRN', 'GCLSTM', 'LSTM']

# Métricas a serem calculadas e plotadas
metricas = ['F1 Score', 'Precision', 'Recall']  # 'Accuracy', 'ROC AUC', 'Log Loss'

# --- 2. PROCESSAMENTO E CÁLCULO DE MÉTRICAS ---

# Loop principal sobre cada país
for country in countries:
    print(f"\n{'=' * 20} Processando País: {country.upper()} {'=' * 20}")

    # Constrói o caminho para a pasta de resultados com base nas configurações
    if backbone:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type,
                                     'backbone_threshold_{:.0f}'.format(threshold * 100))
    else:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type)

    # Cria a pasta 'Figures' dentro da pasta de resultados, se ela não existir
    if not os.path.exists(os.path.join(pasta_results, "Figures")):
        os.makedirs(os.path.join(pasta_results, "Figures"))

    # Dicionário para armazenar os dados de todos os modelos para plotagem posterior
    all_models_data = {}

    # Loop sobre cada modelo
    for nameModel in nameModels:
        print(f"\n--- Modelo: {nameModel} ---")

        # Define o nome do arquivo .npy que armazena os resultados pré-calculados
        file_metrics = os.path.join(pasta_results,
                                    f'results_metrics_{nameModel}_{task_type}_{country}_{threshold}.npy') \
            if backbone else os.path.join(pasta_results,
                                          f'results_metrics_{nameModel}_{task_type}_{country}.npy')

        # --- Bloco de Cálculo ou Carregamento ---
        # Verifica se o arquivo de métricas já existe para evitar reprocessamento
        if not os.path.exists(file_metrics):
            print(f"Arquivo de métricas não encontrado. Calculando a partir dos dados brutos...")

            # Inicializa um dicionário de matrizes para armazenar os resultados de cada métrica
            resultados_metricas = {metrica: np.zeros((lags, outs), dtype=np.float32) for metrica in metricas}

            # Itera sobre todas as combinações de lag (janela) e out (horizonte)
            iteravel = product(range(1, lags + 1), range(1, outs + 1))
            for lag, out in iteravel:
                print(f"  Processando Lag: {lag}, Out: {out}")

                # Caminho para os resultados brutos da combinação específica
                pasta_results_model = os.path.join(pasta_results, nameModel, f'lags_{lag}_out_{out}')

                if not os.path.exists(pasta_results_model) or not os.listdir(pasta_results_model):
                    print(f"    Aviso: Pasta não encontrada ou vazia para {nameModel} L{lag} O{out}. Pulando.")
                    continue

                # Identifica os números das repetições dos experimentos a partir dos nomes dos arquivos
                rep_indices = sorted(
                    set([int(file.split('_')[-1].split('.')[0]) for file in os.listdir(pasta_results_model) if
                         file.endswith('.npy') and 'y_real_rep' in file]))

                if not rep_indices:
                    print(f"    Aviso: Nenhum arquivo .npy de resultados encontrado em {pasta_results_model}. Pulando.")
                    continue

                # Carrega todos os dados de y_real e y_pred para todas as repetições
                y_real_list = [np.load(os.path.join(pasta_results_model, f'y_real_rep_{rep}.npy')) for rep in
                               rep_indices]
                y_pred_list = [np.load(os.path.join(pasta_results_model, f'y_pred_rep_{rep}.npy')) for rep in
                               rep_indices]

                # Concatena os resultados de todas as repetições para um cálculo único e robusto
                y_true_flat = np.concatenate(y_real_list).ravel()
                y_pred_flat = np.concatenate(y_pred_list).ravel()

                # Converte as predições contínuas em classes binárias (0 ou 1) usando um limiar de 0.5
                y_pred_classes = np.where(y_pred_flat >= 0.5, 1, 0)

                # Calcula as métricas de classificação
                f1 = f1_score(y_true_flat, y_pred_classes, average='weighted', zero_division=0)
                precision = precision_score(y_true_flat, y_pred_classes, average='weighted', zero_division=0)
                recall = recall_score(y_true_flat, y_pred_classes, average='weighted', zero_division=0)

                # Armazena o valor da métrica na sua respectiva posição na matriz do heatmap
                # A indexação é lag-1 e out-1 porque os loops começam em 1
                resultados_metricas['F1 Score'][lag - 1, out - 1] = f1
                resultados_metricas['Precision'][lag - 1, out - 1] = precision
                resultados_metricas['Recall'][lag - 1, out - 1] = recall

            # Salva o dicionário de matrizes em um único arquivo .npy para uso futuro
            print(f"Salvando métricas calculadas em: {file_metrics}")
            np.save(file_metrics, resultados_metricas)
        else:
            # Se o arquivo já existe, carrega os dados pré-calculados
            print(f"Carregando métricas pré-calculadas de: {file_metrics}")
            resultados_metricas = np.load(file_metrics, allow_pickle=True).item()

        # Armazena os resultados do modelo atual no dicionário geral
        all_models_data[nameModel] = resultados_metricas

    # --- 3. GERAÇÃO DOS HEATMAPS COMPARATIVOS ---

    # Loop sobre cada métrica para criar uma figura comparativa
    for metrica in metricas:
        print(f"\nGerando heatmap comparativo para a métrica: {metrica}")

        # Cria a figura com subplots lado a lado (1 linha, N colunas para N modelos)
        fig, axes = plt.subplots(1, len(nameModels), figsize=(6 * len(nameModels), 6), sharey=True)

        # Garante que 'axes' seja sempre uma lista, mesmo com um único modelo
        if len(nameModels) == 1:
            axes = [axes]

        # Encontra os valores mínimo e máximo em todos os modelos para esta métrica
        # Isso garante que a escala de cores (vmin, vmax) seja consistente em todos os subplots
        vmin = min(
            all_models_data[model][metrica].min() for model in nameModels if all_models_data[model][metrica].size > 0)
        vmax = max(
            all_models_data[model][metrica].max() for model in nameModels if all_models_data[model][metrica].size > 0)

        # Loop sobre os modelos para plotar cada heatmap
        for i, nameModel in enumerate(nameModels):
            ax = axes[i]  # Seleciona o eixo atual

            # Inverte a matriz verticalmente para que o lag 1 fique na parte de baixo
            flipped_heatmap = np.flipud(all_models_data[nameModel][metrica])

            # --- CORREÇÃO DO TAMANHO DO HEATMAP ---
            # Para o último subplot, criamos um eixo separado para a barra de cores.
            # Isso impede que a barra de cores "achate" o gráfico.
            if i == len(nameModels) - 1:
                # Usa make_axes_locatable para criar um novo eixo à direita do eixo do gráfico
                divider = make_axes_locatable(ax)
                cbar_ax = divider.append_axes("right", size="5%",
                                              pad=0.1)  # 'size' e 'pad' controlam o tamanho e espaçamento
                # Plota o heatmap e direciona a barra de cores para o novo eixo 'cbar_ax'
                sns.heatmap(flipped_heatmap, ax=ax, annot=False, cmap="YlOrBr", vmin=vmin, vmax=vmax,
                            cbar=True, cbar_ax=cbar_ax)
            else:
                # Para os outros gráficos, a barra de cores não é desenhada
                sns.heatmap(flipped_heatmap, ax=ax, annot=False, cmap="YlOrBr", vmin=vmin, vmax=vmax,
                            cbar=False)

            # Configurações de títulos e rótulos para cada subplot
            ax.set_title(nameModel, fontsize=16)
            ax.set_xlabel("Prediction horizon", fontsize=14)
            # O rótulo do eixo Y é mostrado apenas no primeiro gráfico para evitar redundância
            if i == 0:
                ax.set_ylabel("Window size", fontsize=14)

            # Define os rótulos dos eixos (ticks) para serem os valores reais de lag e out
            ax.set_xticklabels(range(1, outs + 1))
            ax.set_yticklabels(range(lags, 0, -1))

            # Melhora a legibilidade dos ticks
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)

        # Ajusta o layout para evitar sobreposição de títulos e rótulos
        plt.tight_layout()

        # Salva a figura combinada em alta resolução
        fig_path = os.path.join(pasta_results, "Figures",
                                f"{country.lower()}_heatmap_{metrica.replace(' ', '_').lower()}.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fecha a figura para liberar memória

    # --- 4. EXIBIÇÃO DE ESTATÍSTICAS GLOBAIS ---

    # Imprime um resumo estatístico para cada modelo e métrica
    for metrica in metricas:
        print(f"\n{'=' * 10} Estatísticas Globais para a Métrica: {metrica} - País: {country} {'=' * 10}")
        for nameModel in nameModels:
            dados_metrica = all_models_data[nameModel][metrica]
            if dados_metrica.size > 0:
                print(f"\nModelo: {nameModel}")
                print(f"  - Máximo: {dados_metrica.max():.4f}")
                print(f"  - Mínimo: {dados_metrica.min():.4f}")
                print(f"  - Média: {dados_metrica.mean():.4f}")
                print(f"  - Desvio Padrão: {dados_metrica.std():.4f}")
                print(
                    f"  - Quartis (25%, 50%, 75%): {np.percentile(dados_metrica, 25):.4f}, {np.median(dados_metrica):.4f}, {np.percentile(dados_metrica, 75):.4f}")
            else:
                print(f"\nModelo: {nameModel}\n  - Sem dados para calcular estatísticas.")
        print("-" * 60)
