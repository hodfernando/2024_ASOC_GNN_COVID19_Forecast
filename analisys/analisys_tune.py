import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_real, y_pred):
    """
    Calcula RMSE e MAE entre arrays com shape (num_reps, num_tests, num_nodes, num_outputs)
    """
    # Garante que os arrays são numpy arrays
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    # Verifica se temos dados suficientes
    if y_real.size == 0 or y_pred.size == 0:
        return {
            'RMSE_mean': np.nan,
            'RMSE_std': np.nan,
            'RMSE_min': np.nan,
            'RMSE_max': np.nan,
            'MAE_mean': np.nan,
            'MAE_std': np.nan,
            'MAE_min': np.nan,
            'MAE_max': np.nan
        }

    # Calcula médias ao longo das repetições e outputs
    y_real_mean = y_real.mean(axis=(0, 3))  # (num_tests, num_nodes)
    y_pred_mean = y_pred.mean(axis=(0, 3))  # (num_tests, num_nodes)

    # Calcula métricas
    rmse = np.sqrt(mean_squared_error(y_real_mean, y_pred_mean)) #, multioutput='raw_values'
    mae = mean_absolute_error(y_real_mean, y_pred_mean) #, multioutput='raw_values'

    # Calcula métricas por repetição para estatísticas
    rmse_values = []
    mae_values = []

    for rep in range(y_real.shape[0]):
        y_real_rep = y_real[rep].mean(axis=2)  # (num_tests, num_nodes)
        y_pred_rep = y_pred[rep].mean(axis=2)  # (num_tests, num_nodes)

        rmse_rep = np.sqrt(mean_squared_error(y_real_rep.flatten(), y_pred_rep.flatten()))
        mae_rep = mean_absolute_error(y_real_rep.flatten(), y_pred_rep.flatten())

        rmse_values.append(rmse_rep)
        mae_values.append(mae_rep)

    metrics = {
        'RMSE_mean': rmse,
        'RMSE_std': np.std(rmse_values),
        'RMSE_min': np.min(rmse_values),
        'RMSE_max': np.max(rmse_values),
        'MAE_mean': mae,
        'MAE_std': np.std(mae_values),
        'MAE_min': np.min(mae_values),
        'MAE_max': np.max(mae_values)
    }

    return metrics


def main():
    # Configurações iniciais
    country = 'Brazil'
    task_type = 'regression'
    backbone = True
    threshold = 0.01

    # Parâmetros dos modelos
    max_reps = 2  # Número máximo de repetições a procurar
    K_values = [1, 2, 3, 4]
    hidden_channels = [32, 64, 128, 256]
    namemodels = ['GCRN', 'GCLSTM', 'LSTM']

    pasta_atual = os.path.dirname(os.path.realpath(__file__))
    pasta_projeto = os.path.dirname(pasta_atual)

    if backbone:
        pasta_results = os.path.join(pasta_projeto, 'results_tune_1', country, task_type,
                                     'backbone_threshold_{:.0f}'.format(threshold * 100))
    else:
        pasta_results = os.path.join(pasta_projeto, 'results_tune_1', country, task_type)

    print(f"Procurando resultados em: {pasta_results}")

    if not os.path.exists(pasta_results):
        print(f"Erro: O diretório não existe: {pasta_results}")
        print("Pastas encontradas no diretório pai:")
        print(os.listdir(os.path.dirname(pasta_results)))
        return

    # Lista para armazenar todos os resultados
    results = []

    # Criar o iterável conforme especificado (sem reps)
    iteravel = product([7, 14], [7, 14], namemodels, [task_type], K_values, hidden_channels)

    # Iterar sobre todas as combinações de parâmetros
    for lag, out, model, _, K, hc in iteravel:
        # Listas para acumular resultados de todas as repetições encontradas
        y_real_all = []
        y_pred_all = []

        # Procurar por todas as repetições disponíveis (até max_reps)
        for rep in range(max_reps):
            # Construir o caminho para os arquivos
            model_dir = os.path.join(
                pasta_results,
                f'k_{K}_hc_{hc}',
                model,
                f'lags_{lag}_out_{out}'
            )

            # Verificar se o diretório existe
            if not os.path.exists(model_dir):
                continue

            # Tentar carregar os arquivos
            try:
                y_real_path = os.path.join(model_dir, f'y_real_no_norm_rep_{rep}.npy')
                y_pred_path = os.path.join(model_dir, f'y_pred_no_norm_rep_{rep}.npy')

                if os.path.exists(y_real_path) and os.path.exists(y_pred_path):
                    y_real = np.load(y_real_path)
                    y_pred = np.load(y_pred_path)

                    y_real_all.append(y_real)
                    y_pred_all.append(y_pred)

                    print(f"Encontrado: K={K}, hc={hc}, model={model}, lag={lag}, out={out}, rep={rep}")
                    print(f"Shape y_real: {y_real.shape}")
                    print(f"Shape y_pred: {y_pred.shape}")

            except Exception as e:
                print(f"Erro ao processar rep={rep} para K={K}, hc={hc}, model={model}, lag={lag}, out={out}: {str(e)}")
                continue

        # Se encontrou pelo menos uma repetição
        if y_real_all:
            # Concatena todas as repetições encontradas
            y_real_concat = np.stack(y_real_all, axis=0)  # (num_reps, num_tests, num_nodes, num_outputs)
            y_pred_concat = np.stack(y_pred_all, axis=0)  # (num_reps, num_tests, num_nodes, num_outputs)

            print(f"Total de repetições encontradas: {len(y_real_all)}")
            print(f"Shape final y_real: {y_real_concat.shape}")
            print(f"Shape final y_pred: {y_pred_concat.shape}")

            # Calcular métricas RMSE e MAE
            metrics = calculate_metrics(y_real_concat, y_pred_concat)

            # Adicionar metadados e métricas aos resultados
            stats = {
                'K': K,
                'hidden_channel': hc,
                'model': model,
                'lag': lag,
                'out': out,
                **metrics
            }

            results.append(stats)
        else:
            print(f"Nenhuma repetição encontrada para K={K}, hc={hc}, model={model}, lag={lag}, out={out}")

    if not results:
        print("Nenhum dado válido encontrado para análise.")
        return

    # Criar DataFrame com todos os resultados
    results_df = pd.DataFrame(results)

    # Criar pasta para salvar as tabelas
    analysis_folder = os.path.join(pasta_projeto, 'results_tune_1', 'analysis_results')
    os.makedirs(analysis_folder, exist_ok=True)

    # Salvar resultados
    stats_file = os.path.join(analysis_folder, 'all_results_stats.csv')
    results_df.to_csv(stats_file, index=False)
    print(f"\nAnálise concluída. Resultados salvos em:\n{stats_file}")

    # Exemplo de tabela agregada
    agg_stats = results_df.groupby(['model', 'K', 'hidden_channel', 'lag', 'out']).agg({
        'RMSE_mean': ['mean', 'std', 'min', 'max'],
        'RMSE_std': ['mean'],
        'MAE_mean': ['mean', 'std', 'min', 'max'],
        'MAE_std': ['mean']
    })

    agg_file = os.path.join(analysis_folder, 'aggregated_stats.csv')
    agg_stats.to_csv(agg_file)
    print(f"Estatísticas agregadas salvas em:\n{agg_file}")


if __name__ == '__main__':
    main()