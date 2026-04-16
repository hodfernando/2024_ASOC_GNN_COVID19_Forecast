import os
import igraph as ig
import polars as pl
import numpy as np

# Path do diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path do diretório principal
project_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Caminho para a pasta "raw_data"
raw_data_dir = os.path.join(project_dir, "raw_data")

# Caminho para a pasta "networks"
networks_dir = os.path.join(raw_data_dir, "networks")

# Listar arquivos de redes no diretório "networks"
networks_files = os.listdir(networks_dir)

# Lista de arquivos de redes de entrada "baidu_in" e de saída "baidu_out"
networks_files_in = [file for file in networks_files if 'baidu_in' in file]
networks_files_out = [file for file in networks_files if 'baidu_out' in file]

# Listas para armazenar os valores de vértices e arestas
g_in_v = []
g_in_e = []
g_out_v = []
g_out_e = []

# Listas para todos os vértices únicos e todas as arestas únicas
all_vertices = set()
all_edges = set()


def calculate_net_metrics(g, mode='all'):
    metrics = {}

    degrees = g.degree(mode=mode)
    metrics['degrees'] = degrees

    closeness = g.closeness(vertices=None, mode=mode, cutoff=None, weights=None, normalized=True)
    metrics['closeness'] = closeness

    betweenness = g.betweenness(vertices=None, directed=True, cutoff=None)
    metrics['betweenness'] = betweenness

    strength = g.strength(weights='weight', mode=mode)
    metrics['strength'] = strength

    # Calcular heterogeneidade
    avg_degree = sum(degrees) / len(degrees)
    squared_degrees = [(d - avg_degree) ** 2 for d in degrees]
    heterogeneity = sum(squared_degrees) / (len(degrees) * (avg_degree ** 2))
    metrics['heterogeneity'] = heterogeneity

    density = g.density()
    metrics['density'] = density

    # Inverter o fluxo
    g.es['w_inv'] = 1.0 / np.array(g.es['weight'])
    diameter = g.diameter(directed=True, weights='w_inv')
    metrics['diameter'] = diameter

    betweenness_w = g.betweenness(vertices=None, directed=True, cutoff=None, weights='w_inv')
    metrics['betweenness_w'] = betweenness_w

    closeness_w = g.closeness(vertices=None, mode=mode, cutoff=None, weights='w_inv', normalized=True)
    metrics['closeness_w'] = closeness_w

    strength_w = g.strength(weights='w_inv', mode=mode)
    metrics['strength_w'] = strength_w

    return metrics


# Crie DataFrames separados para entrada e saída
columns = ["GraphFile", "Degree", "Betweenness", "Closeness", "Strength", "Heterogeneity", "Density", "Diameter",
           "Betweenness_w", "Closeness_w", "Strength_w"]
input_df = pl.DataFrame({col: [] for col in columns})
output_df = pl.DataFrame({col: [] for col in columns})

# Loop para verificar os grafos em 'networks_files_in' e 'networks_files_out'
for graph_file_in, graph_file_out in zip(networks_files_in, networks_files_out):
    graph_path_in = os.path.join(networks_dir, graph_file_in)
    graph_path_out = os.path.join(networks_dir, graph_file_out)

    # Lê o grafo de entrada e saída
    g_in = ig.Graph.Read_GraphML(graph_path_in)
    g_out = ig.Graph.Read_GraphML(graph_path_out)

    # Calcule as métricas
    metrics_in = calculate_net_metrics(g_in, mode='in')
    metrics_out = calculate_net_metrics(g_out, mode='out')

    # Crie um DataFrame para esta iteração
    df_in = pl.DataFrame({
        "GraphFile": [graph_file_in],
        "Degree": [metrics_in['degrees']],
        "Betweenness": [metrics_in['betweenness']],
        "Closeness": [metrics_in['closeness']],
        "Strength": [metrics_in['strength']],
        "Heterogeneity": [metrics_in['heterogeneity']],
        "Density": [metrics_in['density']],
        "Diameter": [metrics_in['diameter']],
        "Betweenness_w": [metrics_in['betweenness_w']],
        "Closeness_w": [metrics_in['closeness_w']],
        "Strength_w": [metrics_in['strength_w']]
    })

    df_out = pl.DataFrame({
        "GraphFile": [graph_file_out],
        "Degree": [metrics_out['degrees']],
        "Closeness": [metrics_out['closeness']],
        "Betweenness": [metrics_out['betweenness']],
        "Strength": [metrics_out['strength']],
        "Heterogeneity": [metrics_out['heterogeneity']],
        "Density": [metrics_out['density']],
        "Diameter": [metrics_out['diameter']],
        "Betweenness_w": [metrics_out['betweenness_w']],
        "Closeness_w": [metrics_out['closeness_w']],
        "Strength_w": [metrics_out['strength_w']]
    })

    # Anexe DataFrames de entrada e saída
    input_df = pl.concat([input_df, df_in], how="diagonal_relaxed")
    output_df = pl.concat([output_df, df_out], how="diagonal_relaxed")

# Diretório para salvar os arquivos .csv
results_dir = os.path.join(project_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# Salve os DataFrames de entrada e saída em arquivos .csv na pasta de resultados
input_df.write_json(os.path.join(results_dir, "network_metrics_input.json"))
output_df.write_json(os.path.join(results_dir, "network_metrics_output.json"))
