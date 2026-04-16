import os
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from itertools import product


def calculate_node_metrics(y_real, y_pred, node_metrics_dict):
    """
    Calcula RMSE e MAE para cada nó e correlaciona com métricas de rede
    Retorna um DataFrame com os resultados por município
    """
    # Garante que os arrays são numpy arrays
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    if y_real.size == 0 or y_pred.size == 0:
        return pd.DataFrame()

    # Cálculo vetorizado mais eficiente
    errors = y_real - y_pred
    rmse_nodes = np.sqrt(np.mean(errors ** 2, axis=(0, 1, 3)))
    mae_nodes = np.mean(np.abs(errors), axis=(0, 1, 3))

    # Cria DataFrame com os resultados
    results = []
    for codigo, idx in node_metrics_dict['codigo_para_indice'].items():
        results.append({
            'codigo_municipio': codigo,
            'RMSE': rmse_nodes[idx],
            'MAE': mae_nodes[idx],
            'betweenness': node_metrics_dict['betweenness'].get(codigo, 0),
            'degree': node_metrics_dict['degree'].get(codigo, 0),
            'strength': node_metrics_dict['strength'].get(codigo, 0)
        })

    return pd.DataFrame(results)


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

    # Carrega informações do grafo e mapeamento
    pasta_pre_processed = os.path.join(pasta_projeto, 'pre_processed', country)

    try:
        if backbone:
            # Caminhos para os arquivos a serem salvos
            caminho_codigos = os.path.join(pasta_pre_processed, f'codigos_municipios_back_{threshold}.pkl')
            caminho_mapeamento = os.path.join(pasta_pre_processed, f'codigo_para_indice_back_{threshold}.pkl')
            caminho_grafo = os.path.join(pasta_pre_processed, f'grafo_back_{threshold}.pkl')
        else:
            caminho_codigos = os.path.join(pasta_pre_processed, 'codigos_municipios.pkl')
            caminho_mapeamento = os.path.join(pasta_pre_processed, 'codigo_para_indice.pkl')
            caminho_grafo = os.path.join(pasta_pre_processed, 'grafo.pkl')

        with open(caminho_codigos, 'rb') as f:
            codigos_municipios = pickle.load(f)
        with open(caminho_mapeamento, 'rb') as f:
            codigo_para_indice = pickle.load(f)
        with open(caminho_grafo, 'rb') as f:
            G = pickle.load(f)

        # Cria mapeamento reverso (índice -> código)
        indice_para_codigo = {v: k for k, v in codigo_para_indice.items()}

        # PRÉ-CALCULA as métricas de rede CORRETAMENTE
        print("\nCalculando métricas de rede...")

        # Calcula métricas usando os índices do grafo
        betweenness_idx = nx.betweenness_centrality(G, weight='weight')
        degree_idx = dict(G.degree())
        strength_idx = dict(G.degree(weight='weight'))

        # Converte para usar códigos de município como chaves
        betweenness = {indice_para_codigo[k]: v for k, v in betweenness_idx.items()}
        degree = {indice_para_codigo[k]: v for k, v in degree_idx.items()}
        strength = {indice_para_codigo[k]: v for k, v in strength_idx.items()}

        # Verificação
        print("\nExemplo de valores calculados (usando códigos reais):")
        sample_code = next(iter(codigo_para_indice.keys()))
        print(f"Código município: {sample_code}")
        print(f"Betweenness: {betweenness.get(sample_code, 'n/a')}")
        print(f"Degree: {degree.get(sample_code, 'n/a')}")
        print(f"Strength: {strength.get(sample_code, 'n/a')}")

        node_metrics_dict = {
            'betweenness': betweenness,
            'degree': degree,
            'strength': strength,
            'codigo_para_indice': codigo_para_indice
        }

        print("Informações do grafo e mapeamento carregadas com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar informações do grafo: {str(e)}")
        return

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

    # Dicionário para armazenar todos os datasets por configuração
    config_datasets = {}

    # Criar pasta para salvar as tabelas
    analysis_folder = os.path.join(pasta_projeto, 'results_tune_1', 'analysis_results')
    os.makedirs(analysis_folder, exist_ok=True)

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

            # Calcular métricas por nó
            node_metrics_df = calculate_node_metrics(y_real_concat, y_pred_concat, node_metrics_dict)

            if not node_metrics_df.empty:
                # Identificador único da configuração
                config_key = f"K{K}_hc{hc}_lag{lag}_out{out}"

                # Se já existe dataset para esta configuração, adiciona as colunas do modelo atual
                if config_key in config_datasets:
                    existing_df = config_datasets[config_key]
                    # Adiciona colunas para este modelo
                    existing_df[f'{model}_RMSE'] = node_metrics_df['RMSE']
                    existing_df[f'{model}_MAE'] = node_metrics_df['MAE']
                else:
                    # Cria novo dataset para esta configuração
                    config_df = node_metrics_df[['codigo_municipio', 'betweenness', 'degree', 'strength']].copy()
                    config_df[f'{model}_RMSE'] = node_metrics_df['RMSE']
                    config_df[f'{model}_MAE'] = node_metrics_df['MAE']
                    config_datasets[config_key] = config_df

    # Processa cada configuração salva
    for config_key, config_df in config_datasets.items():
        print(f"\n\n=== Análise para configuração {config_key} ===")

        # Salva o dataset completo
        config_file = os.path.join(analysis_folder, f"config_{config_key}.csv")
        config_df.to_csv(config_file, index=False)
        print(f"Dataset salvo em: {config_file}")

        # Lista de modelos presentes nesta configuração
        models = list(set([col.split('_')[0] for col in config_df.columns if '_RMSE' in col or '_MAE' in col]))

        # Análise para betweenness, degree e strength
        for metric in ['betweenness', 'degree', 'strength']:
            print(f"\nTop 10 cidades por {metric}:")

            # Ordena pelo valor da métrica de rede
            top_cities = config_df.sort_values(metric, ascending=False).head(10)

            # Mostra os valores de RMSE e MAE para cada modelo
            for model in models:
                print(f"\nModelo {model}:")
                print(top_cities[['codigo_municipio', metric, f'{model}_RMSE', f'{model}_MAE']])

                # Calcula médias para os top 10
                avg_rmse = top_cities[f'{model}_RMSE'].mean()
                avg_mae = top_cities[f'{model}_MAE'].mean()
                print(f"Média RMSE top 10: {avg_rmse:.4f}, Média MAE top 10: {avg_mae:.4f}")

    print("\nAnálise concluída. Todos os resultados foram salvos em:", analysis_folder)


if __name__ == '__main__':
    main()
