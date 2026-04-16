import os
import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Função de Geração de Gráficos (em Inglês) - VERSÃO CORRIGIDA
# --------------------------------------------------------------------------- #

def regenerate_cd_plot_english(results_df, metric_col, metric_name, analysis_folder, model_names_ordered, window_size,
                               prediction_horizon):
    """
    Gera um Diagrama de Diferença Crítica (CD Plot) com textos em inglês e
    correções para evitar sobreposição de texto.

    Esta versão inclui:
    1. Lógica para alternar a altura dos nomes dos modelos quando seus ranks são muito próximos.
    2. Layout ajustado para acomodar corretamente a legenda inferior.

    Args:
        results_df (pd.DataFrame): DataFrame com resultados para UMA combinação de Window/Horizon.
        metric_col (str): Nome da coluna da métrica (ex: 'RMSE_mean').
        metric_name (str): Nome legível da métrica (ex: 'RMSE').
        analysis_folder (str): Caminho para salvar o gráfico PDF.
        model_names_ordered (list): Lista de nomes de modelos.
        window_size (int): O tamanho da janela de entrada.
        prediction_horizon (int): O horizonte de previsão.
    """
    print(
        f"\n--- Regenerating Plot for {metric_name.upper()} (Window: {window_size}, Horizon: {prediction_horizon}) ---")

    if results_df.empty:
        print("  Empty results DataFrame. Skipping plot.")
        return

    # Cria a coluna 'block_id_for_pivot' para o teste de Friedman
    results_df['block_id_for_pivot'] = "K" + results_df['K'].astype(str) + "_hc" + results_df['hidden_channel'].astype(
        str)
    pivot_table = results_df.pivot_table(index='block_id_for_pivot', columns='model', values=metric_col)
    pivot_table_cleaned = pivot_table.dropna()

    if pivot_table_cleaned.empty or len(pivot_table_cleaned) < 2 or len(pivot_table_cleaned.columns) < 2:
        print("  Insufficient data for Friedman test. Skipping plot.")
        return

    actual_models_in_pivot = [col for col in model_names_ordered if col in pivot_table_cleaned.columns]
    if len(actual_models_in_pivot) < 2:
        print("  Less than two models available. Skipping plot.")
        return
    pivot_table_cleaned = pivot_table_cleaned[actual_models_in_pivot]

    # --- Análise Estatística ---
    stat, p_value = np.nan, 1.0
    try:
        stat, p_value = friedmanchisquare(*[pivot_table_cleaned[col] for col in actual_models_in_pivot])
        print(f"  Friedman Test: Statistic={stat:.4f}, p-value={p_value:.4g}")
    except ValueError as e:
        print(f"  Friedman test failed: {e}.")

    mean_ranks = pivot_table_cleaned.rank(axis=1, method='average', ascending=True).mean().sort_values()
    print(f"  Average Ranks ({metric_name}):\n{mean_ranks}")

    nemenyi_results = None
    if not pd.isna(p_value) and p_value < 0.05:
        try:
            nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_table_cleaned.to_numpy())
            nemenyi_results.columns = actual_models_in_pivot
            nemenyi_results.index = actual_models_in_pivot
        except Exception as e:
            print(f"  Nemenyi post-hoc test failed: {e}")

    # --- Plotagem do Gráfico ---
    fig, ax = plt.subplots(figsize=(10, max(3, len(actual_models_in_pivot) * 0.6)), facecolor='white')
    try:
        # --- LÓGICA PARA EVITAR SOBREPOSIÇÃO DOS NOMES DOS MODELOS ---
        label_y_offsets = {}
        sorted_models = mean_ranks.index.tolist()
        base_y, elevated_y = 0.3, 0.45  # Alturas normal e elevada
        rank_threshold = 0.25  # Se a diferença de rank for menor que isso, os nomes são muito próximos

        if sorted_models:
            # O primeiro modelo sempre fica na altura base
            label_y_offsets[sorted_models[0]] = base_y
            for i in range(1, len(sorted_models)):
                prev_model = sorted_models[i - 1]
                curr_model = sorted_models[i]
                rank_diff = abs(mean_ranks[curr_model] - mean_ranks[prev_model])

                # Se muito próximo do anterior, alterna a altura
                if rank_diff < rank_threshold:
                    if label_y_offsets[prev_model] == base_y:
                        label_y_offsets[curr_model] = elevated_y
                    else:
                        label_y_offsets[curr_model] = base_y
                else:  # Se não, reseta para a altura base
                    label_y_offsets[curr_model] = base_y
        # --- FIM DA LÓGICA ---

        # Plota os pontos e os nomes dos modelos com os offsets calculados
        ax.scatter(mean_ranks.values, np.zeros(len(mean_ranks)), s=120, color='black', zorder=3)
        for model, rank_val in mean_ranks.items():
            ax.text(rank_val, label_y_offsets.get(model, base_y), model, ha='center', va='bottom', fontsize=10)

        # Configurações do gráfico
        ax.set_title(
            f"Critical Difference Diagram - {metric_name}\n(Window: {window_size}, Horizon: {prediction_horizon})",
            fontsize=12)
        ax.set_xlabel(f"Average Rank for {metric_name} (lower is better)", fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(mean_ranks.min() - 0.7, mean_ranks.max() + 0.7)
        ax.set_ylim(-0.8, 1.2)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Desenha as linhas de conexão do Nemenyi
        line_threshold = 0.05
        if nemenyi_results is not None:
            y_offset, y_step = -0.15, -0.08
            for i in range(len(sorted_models)):
                for j in range(i + 1, len(sorted_models)):
                    model1, model2 = sorted_models[i], sorted_models[j]
                    try:
                        pval = nemenyi_results.loc[model1, model2]
                        if pval >= line_threshold:
                            ax.plot([mean_ranks[model1], mean_ranks[model2]], [y_offset, y_offset], color='black',
                                    lw=1.5)
                            y_offset += y_step
                            if y_offset < -0.75: y_offset = -0.15 - np.random.rand() * 0.02
                    except KeyError:
                        continue

        # --- CORREÇÃO PARA A LEGENDA INFERIOR ---
        # Ajusta o layout para criar mais espaço na parte inferior
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Aumenta a margem inferior de 0.08 para 0.1

        p_val_caption = f"{p_value:.3g}" if not pd.isna(p_value) else "N/A"
        caption = f"Models connected by a line are not statistically different (Nemenyi, p >= {line_threshold}).\nFriedman p-value: {p_val_caption}"

        # Posiciona o texto em uma coordenada segura dentro do espaço criado
        plt.figtext(0.5, 0.01, caption, ha='center', fontsize=9, wrap=True)

        # Salva o arquivo PDF
        filename = f"cd_plot_geral_lag{window_size}_out{prediction_horizon}_{metric_name.lower()}.pdf"
        filepath = os.path.join(analysis_folder, filename)
        plt.savefig(filepath, dpi=300,
                    facecolor='white')  # bbox_inches='tight' é removido para usar nosso layout customizado
        print(f"  ✅ Figure saved successfully: {filepath}")

    except Exception as e:
        print(f"  ❌ Error generating plot: {e}")
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Função Principal (main)
# --------------------------------------------------------------------------- #

def main():
    """
    Função principal para regenerar os gráficos a partir dos dados CSV processados.
    """
    # --- Configurações (devem ser as mesmas do script de análise) ---
    countries = ['Brazil', 'China']
    namemodels = ['GCRN', 'GCLSTM', 'LSTM']

    # Determina o caminho base do projeto
    try:
        current_folder = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        current_folder = os.getcwd()

    # Lógica aprimorada para encontrar a pasta raiz do projeto
    project_base_path = current_folder
    for _ in range(3):  # Tenta subir até 3 níveis
        if os.path.basename(project_base_path).lower() in ["scripts", "code", "notebooks"]:
            project_base_path = os.path.dirname(project_base_path)
        else:
            break
    if 'results_tune_1' not in os.listdir(project_base_path):  # Heurística final
        project_base_path = os.path.dirname(current_folder)

    print(f"Project base path detected: {project_base_path}")

    # Loop principal sobre os países
    for country in countries:
        print(f"\n{'=' * 20} PROCESSING COUNTRY: {country.upper()} {'=' * 20}")

        analysis_folder_path = os.path.join(project_base_path, 'results_tune_1', 'analysis_results', country)
        csv_filename = f'all_results_detailed_stats_{country}.csv'
        csv_filepath = os.path.join(analysis_folder_path, csv_filename)

        if not os.path.exists(csv_filepath):
            print(f"Error: Summary CSV file not found for {country}.\nExpected at: {csv_filepath}")
            continue

        print(f"Loading summary data from: {csv_filepath}")
        results_df_country = pd.read_csv(csv_filepath)

        # Mapeia os nomes das colunas 'lag' e 'out' para a nova terminologia
        if 'lag' not in results_df_country.columns or 'out' not in results_df_country.columns:
            print("Error: 'lag' or 'out' columns not found in the CSV. Skipping.")
            continue

        results_df_country.rename(columns={'lag': 'Window size', 'out': 'Prediction horizon'}, inplace=True)

        unique_configs = results_df_country[['Window size', 'Prediction horizon']].drop_duplicates().values

        # Loop sobre cada configuração para gerar os gráficos
        for window_size, prediction_horizon in unique_configs:
            df_subset = results_df_country[
                (results_df_country['Window size'] == window_size) &
                (results_df_country['Prediction horizon'] == prediction_horizon)
                ].copy()

            # Gera os gráficos para RMSE e MAE
            for metric_col, metric_name in [('RMSE_mean', 'RMSE'), ('MAE_mean', 'MAE')]:
                regenerate_cd_plot_english(
                    results_df=df_subset,
                    metric_col=metric_col,
                    metric_name=metric_name,
                    analysis_folder=analysis_folder_path,
                    model_names_ordered=namemodels,
                    window_size=window_size,
                    prediction_horizon=prediction_horizon
                )

    print("\nAll figures have been regenerated.")


if __name__ == '__main__':
    main()
