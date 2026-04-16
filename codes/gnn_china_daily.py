# coding: utf-8
# Importing libraries
import datetime
import glob
import os
import pickle
import random
import math
from functools import partial
import numpy as np
import networkx as nx
import pandas as pd
import polars as pl
from itertools import product
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

# from models.gcn_based_rnn_model import GCRN
# from models.gcn_based_lstm_model import GCLSTM
# from models.TemporalLSTM_model import TemporalLSTM


import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from gcn_based_rnn_model import GCRN
from gcn_based_lstm_model import GCLSTM
from TemporalLSTM_model import TemporalLSTM

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def extract_backbone(graph, alpha, g_strength=None, g_degree=None, ignored_nodes={}):
    backbone = nx.Graph()
    codes = set()
    sorted_edges = sorted(graph.edges.data(), key=lambda x: x[2]['weight'], reverse=True)

    for source, target, data in sorted_edges:
        weight = data['weight']
        s_source, s_target = g_strength[source], g_strength[target]
        k_source, k_target = g_degree[source], g_degree[target]

        weight_ratio_source = weight / s_source
        weight_ratio_target = weight / s_target

        pij = (1 - weight_ratio_source) ** (k_source - 1)
        pji = (1 - weight_ratio_target) ** (k_target - 1)

        p = min(pij, pji)

        if (source in ignored_nodes and ignored_nodes[source] <= 5) or \
                (target in ignored_nodes and ignored_nodes[target] <= 5) or \
                (p < alpha):
            if source in ignored_nodes:
                ignored_nodes[source] += 1
            if target in ignored_nodes:
                ignored_nodes[target] += 1

            backbone.add_edge(source, target, weight=weight)
            backbone.add_nodes_from([source], City_EN=graph.nodes[source]['City_EN'])
            backbone.add_nodes_from([target], City_EN=graph.nodes[target]['City_EN'])

            codes.add(graph.nodes[source]['City_EN'])
            codes.add(graph.nodes[target]['City_EN'])

    return backbone, np.array(sorted(codes))


def format_dates(networks_files):
    """
    Formata as datas extraídas dos nomes de arquivos de redes.

    Args:
        networks_files (list): Lista de nomes de arquivos de redes.

    Returns:
        list: Lista de datas formatadas no formato "YYYY-MM-DD".
    """
    # Lista para armazenar as datas formatadas
    formatted_dates = []

    # Iterar sobre os nomes de arquivos de redes
    for file in networks_files:
        # Extrair a data do nome do arquivo
        date = os.path.splitext(os.path.basename(file))[0].split('_')[-1]

        # Tentar converter a data para um objeto datetime
        try:
            date_obj = datetime.datetime.strptime(date, "%Y%m%d")
            # Formatar a data no formato "YYYY-MM-DD" e adicioná-la à lista
            formatted_dates.append(date_obj.strftime("%Y-%m-%d"))
        except ValueError:
            # Tratar erros de formatação de data
            print(f"Erro na data: {date}")

    return formatted_dates


def construindo_dataframe_networks(networks_dir, dataset_in_out='in'):
    """
    Constrói um DataFrame a partir dos arquivos de redes de entrada ou saída.

    Args:
        networks_dir (str): O diretório contendo os arquivos de redes.
        dataset_in_out (str, optional): O tipo de dataset a ser processado ('in' para entradas, 'out' para saídas).
            O padrão é 'in'.

    Returns:
        tuple: Uma tupla contendo a lista de datas e o DataFrame construído.
    """
    print(f"Construindo DataFrame para os arquivos de redes de {dataset_in_out}put")

    # Lista apenas arquivos .csv no diretório de redes
    networks_files = [file for file in glob.glob(os.path.join(networks_dir, "*.csv")) if os.path.isfile(file)]

    # Filtrar os arquivos que contêm '-'
    networks_files = list(filter(lambda file: '-' not in os.path.basename(file), networks_files))

    # Lista de arquivos de redes de entrada "baidu_in" e de saída "baidu_out"
    networks_files_in = [file for file in networks_files if ('baidu_in' in file)]
    networks_files_out = [file for file in networks_files if ('baidu_out' in file)]

    # Formatar as datas dos arquivos de entrada e saída
    date_networks_in = format_dates(networks_files_in)
    date_networks_out = format_dates(networks_files_out)

    # Criar DataFrames com as datas formatadas
    df_networks_in = pl.DataFrame({'date_networks': date_networks_in, 'networks_files_in': networks_files_in})

    df_networks_out = pl.DataFrame({'date_networks': date_networks_out, 'networks_files_out': networks_files_out})

    return (date_networks_in, df_networks_in) if dataset_in_out == 'in' else (date_networks_out, df_networks_out)


def construindo_dataframe_covid(pasta_raw_data):
    """
    Constrói um DataFrame para os dados de COVID-19 a partir de um diretório e salva o DataFrame em um arquivo CSV.

    Args:
        pasta_raw_data (str): O diretório contendo os dados de COVID-19.
        save_path (str): O diretório onde o arquivo CSV será salvo.

    Returns:
        DataFrame: O DataFrame contendo os dados de COVID-19.

    Raises:
        FileNotFoundError: Se o arquivo de dados de COVID-19 não for encontrado no diretório especificado.
    """
    print("Construindo DataFrame para os dados de COVID-19")

    # Caminho para a pasta "Covid-19 daily cases in China"
    covid_dir = os.path.join(pasta_raw_data, "Covid-19 daily cases in China")

    # Ler os dados de COVID-19 do arquivo Excel
    df_covid_temporal = pl.read_excel(source=os.path.join(covid_dir, "covid-19 daily confirmed cases.xlsx"),
                                      engine="xlsx2csv", read_options={"has_header": True})

    # Remover a última coluna (índice 129)
    df_covid_temporal = df_covid_temporal.drop(df_covid_temporal.columns[129])

    # Renomear as colunas para o formato "YYYY-MM-DD"
    new_columns = {}
    for date in df_covid_temporal.columns[2:]:
        new_date = datetime.datetime.strptime(date, "%d/%m/%Y").strftime("%Y-%m-%d")
        new_columns[date] = new_date

    # Renomear as colunas usando o dicionário de mapeamento
    df_covid_temporal = df_covid_temporal.rename(new_columns)

    return df_covid_temporal


def construindo_dataframe_common_dates(df_covid_temporal, date_networks, df_networks):
    """
    Constrói um DataFrame contendo as datas comuns entre os arquivos de redes e os dados de COVID-19.

    Args:
        df_covid_temporal (polars.DataFrame): O DataFrame contendo os dados de COVID-19.
        date_networks (list): A lista de datas dos arquivos de redes.
        df_networks (polars.DataFrame): O DataFrame contendo os dados das redes.

    Returns:
        polars.DataFrame: O DataFrame contendo as datas comuns entre os arquivos de redes e os dados de COVID-19.
    """
    print("Construindo DataFrame com as datas comuns entre os arquivos de redes e os dados de COVID-19")

    # Selecione todas as colunas a partir da terceira coluna
    date_covid = df_covid_temporal.columns[2:]

    # Datas comuns entre os arquivos de redes e os dados de covid
    common_dates = sorted(set(date_networks).intersection(date_covid))

    # Criar DataFrame com base nas datas comuns
    df_common_dates = df_networks.filter(pl.col('date_networks').is_in(common_dates))

    for i in range(1, len(common_dates)):
        date_prev = datetime.datetime.strptime(common_dates[i - 1], '%Y-%m-%d')
        date_curr = datetime.datetime.strptime(common_dates[i], '%Y-%m-%d')

        # Verificar a diferença em dias entre datas consecutivas
        days_diff = (date_curr - date_prev).days

        if days_diff > 1:
            print(f"Salto temporal detectado no índice {i}: {common_dates[i - 1]} -> {common_dates[i]}")

            # Adicionar as datas ausentes ao DataFrame auxiliar
            missing_dates = [date_prev + datetime.timedelta(days=day) for day in range(1, days_diff)]

            # Filtrar a linha correspondente à data anterior em df_common_dates
            filtered_row = df_common_dates.filter(pl.col('date_networks') == common_dates[i - 1])

            # Criar DataFrame com a coluna de date_networks e missing_dates
            empty_df = pl.DataFrame({'date_networks': [date.strftime('%Y-%m-%d') for date in missing_dates]})

            # Inserir a série no DataFrame vazio
            empty_df.insert_column(1, pl.Series(df_common_dates.columns[1],
                                                [filtered_row[df_common_dates.columns[1]][0]] * len(missing_dates)))

            # Concatenar o DataFrame vazio com df_common_dates
            df_common_dates = pl.concat([df_common_dates, empty_df])

    # Reordenar o DataFrame por data
    df_common_dates = df_common_dates.sort('date_networks')

    return df_common_dates


def construir_grafo_temporal(networks_dir, df_common_dates, df_covid_temporal, dataset_in_out='in', cumulativo=True):
    """
    Constrói o grafo espaço-temporal usando os dados de redes e COVID-19.

    Args:
        networks_dir (str): O diretório contendo os arquivos de redes.
        df_common_dates (polars.DataFrame): O DataFrame contendo as datas comuns entre os arquivos de redes e os dados de COVID-19.
        df_covid_temporal (polars.DataFrame): O DataFrame contendo os dados de COVID-19.
        dataset_in_out (str, optional): O tipo de conjunto de dados a ser usado, 'in' para entrada e 'out' para saída. Defaults to 'in'.
        cumulativo (bool, optional): Se True, retorna os dados cumulativos. Se False, retorna os dados diários. Defaults to True.

    Returns:
        tuple: Um par de DataFrames contendo o índice das cidades e o grafo espaço-temporal.
    """
    print("Construindo o grafo espaço-temporal")

    # Caminho para a pasta "networks"
    info_dir = os.path.join(networks_dir, "info")
    df_index = pl.read_csv(info_dir + "/Index_City_CH_EN.csv")
    df_index = df_index.with_columns(pl.col("GbCity").cast(pl.String))

    # Carregar o DataFrame com a população da China
    df_pop_china = pl.read_excel(source=os.path.join(pasta_raw_data, "worldcities.xlsx"),
                                 engine="xlsx2csv", read_options={"has_header": True})

    # Filtrar por país "China" e por população maior que zero
    df_china = df_pop_china.filter(
        (pl.col("country") == "China") &
        (pl.col("population").is_not_null()) &
        (pl.col("population") > 0)
    ).select(["city", "city_ascii", "population", "country"])

    # Retirando as cidades que não serão usadas
    file = df_common_dates[f'networks_files_{dataset_in_out}'][0]
    df_current = pl.read_csv(file)
    intersection = set(df_current.columns) & set(df_index['City_CH'])

    # Filtrar df_index mantendo apenas as linhas que têm valores em intersection
    df_index = df_index.filter(df_index['City_CH'].is_in(intersection))
    # Retirar as linhas duplicadas
    df_index = df_index.group_by('City_EN').agg(pl.first('*'))

    # Aplicar o filtro no DataFrame df_covid_temporal
    df_graph_temporal = df_covid_temporal.filter(df_covid_temporal['City_name'].is_in(df_index['City_EN']) &
                                                 df_covid_temporal['City_name'].is_in(df_china['city_ascii']))

    # Criar um filtro para selecionar apenas as cidades presentes no df_graph_temporal
    index_filter = df_index['City_EN'].is_in(df_graph_temporal['City_name'])

    # Aplicar o filtro no DataFrame df_index
    df_index = df_index.filter(index_filter)

    # Seleciona apenas as colunas (Datas) que são comuns a ambos os DataFrames
    common_columns = ['City_name', 'City_code'] + df_common_dates['date_networks'].to_list()
    df_graph_temporal = df_graph_temporal[common_columns]

    # Aplicar soma cumulativa apenas se cumulativo=True
    if cumulativo:
        # Selecionar as colunas desejadas e aplicar a soma cumulativa horizontal
        cumulative_sum = df_graph_temporal.select(pl.cum_sum_horizontal(common_columns[2:])).unnest('cum_sum')
        # Substituir as colunas originais pelas selecionadas juntamente com as colunas cumulativas
        df_graph_temporal = df_graph_temporal.select(common_columns[:2]).with_columns(cumulative_sum)

    # Reordenar os DataFrames por 'City_name'/'City_EN'
    df_index = df_index.sort('City_EN')
    df_graph_temporal = df_graph_temporal.sort('City_name')

    return df_index, df_graph_temporal


def edges_and_weights(df_index, df_current, df_current_col, df_index_col, edges_weights):
    """
    Calcula as arestas e os pesos do grafo com base nos DataFrames fornecidos.

    Args:
        df_index (polars.DataFrame): DataFrame contendo o índice das cidades.
        df_current (polars.DataFrame): DataFrame contendo os dados atuais.
        df_current_col (str): Nome da coluna no DataFrame atual.
        df_index_col (str): Nome da coluna no DataFrame de índice.
        edges_weights (dict): Dicionário para armazenar as arestas e pesos calculados.

    Returns:
        dict: Dicionário atualizado com as arestas e pesos calculados.
    """
    # Filtra as linhas em df_current onde df_current[df_current_col] está em df_index[df_index_col]
    df_current_filtered = df_current.filter(df_current[df_current_col].is_in(df_index[df_index_col]))

    # Lista de cidades
    cities = df_index[df_index_col].to_list()

    # Iterar sobre as chaves do dicionário
    for i, col_current in enumerate(cities):
        if col_current in df_current_filtered.columns and df_current_filtered[col_current].dtype is not str:
            for j in range(i + 1, cities.__len__()):
                # Obtém a coluna atual usando o índice de df_current
                col = df_current_filtered[col_current]
                # Verifica se a coluna é diferente de zero
                if col[j] is not None and col[j] != 0.0:
                    key = f"{i}_{j}"
                    # Verifica se a chave existe no dicionário edges_weights
                    if key in edges_weights:
                        # Atualiza os valores da chave no dicionário edges_weights
                        current_value = edges_weights[key]
                        new_value = [current_value[0], current_value[1] + np.float64(col[j]), current_value[2] + 1]
                        edges_weights[key] = new_value
                    else:
                        # Cria uma nova entrada no dicionário edges_weights
                        edges_weights[key] = [[i, j], np.float64(col[j]), 1]


def construindo_rede_mobilidade(pasta_pre_processed, df_index, df_common_dates, dataset_in_out='in',
                                backbone=False, alpha=0.01, tipo_grafo="grafo_original", ):
    """
    Constrói a rede de mobilidade com base nos DataFrames fornecidos.

    Args:
        df_index (polars.DataFrame): DataFrame contendo o índice das cidades.
        df_common_dates (polars.DataFrame): DataFrame contendo as datas comuns entre os arquivos de redes e os dados de COVID-19.
        dataset_in_out (str, optional): Tipo de conjunto de dados a ser considerado (entrada ou saída). Defaults to 'in'.
        save_path (str, optional): Caminho para salvar os arquivos de edges e weights. Defaults to None.

    Returns:
        tuple: Tupla contendo os arrays numpy de edges e weights.
    """
    print("Construindo a rede de mobilidade")

    # --- 1. Definição de Nomes de Arquivos ---
    sufixo_arquivo = tipo_grafo
    if backbone:
        sufixo_arquivo += f"_backbone_{alpha}"

    caminho_edges = os.path.join(pasta_pre_processed, f'edges_{sufixo_arquivo}.npy')
    caminho_weights = os.path.join(pasta_pre_processed, f'weights_{sufixo_arquivo}.npy')
    caminho_grafo = os.path.join(pasta_pre_processed, f'grafo_{sufixo_arquivo}.graphml')

    # --- 2. Carregamento de Dados (se já existirem) ---
    if all(os.path.exists(p) for p in
           [caminho_edges, caminho_weights, caminho_grafo]):
        print("Carregando dados pré-processados...")
        edges = np.load(caminho_edges)
        weights = np.load(caminho_weights)
        G = nx.read_graphml(caminho_grafo)
        G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        return edges, weights, G

    # --- 3. Construção do Grafo Base ---
    print("Construindo grafo base a partir dos dados brutos...")

    # Criando o grafo
    G = nx.Graph()

    edges_weights = {}

    for file in df_common_dates[f'networks_files_{dataset_in_out}']:
        df_current = pl.read_csv(file)
        if "city_name" in df_current.columns:
            edges_and_weights(df_index, df_current, "city_name", "City_CH", edges_weights)
        elif "City_EN" in df_current.columns:
            edges_and_weights(df_index, df_current, "City_EN", "City_EN", edges_weights)
        elif "GbCity_EN" in df_current.columns:
            break
        #     # Criar uma cópia do DataFrame
        #     df_processed = df_current
        #     # Colunas originais
        #     columns = df_processed.columns
        #     # Aplicar a função de extração em todas as colunas, exceto 'GbCity_EN'
        #     new_columns = ['GbCity_EN'] + [col.split('_')[0] for col in columns[1:]]
        #     # Renomear as cidades em 'GbCity_EN'
        #     df_processed = df_processed.with_columns(
        #         pl.col('GbCity_EN').replace(new=pl.Series(new_columns[1:]), old=pl.Series(columns[1:])))
        #     # Remover as cidades duplicadas em 'GbCity_EN'
        #     df_processed = df_processed.unique(subset=['GbCity_EN'], keep='first', maintain_order=True)
        #     # Identificar as colunas duplicadas e as unicas
        #     duplicated_columns = set()
        #     unique_columns = []
        #     for old_col, new_col in zip(columns, new_columns):
        #         if new_col not in duplicated_columns:
        #             unique_columns.append(old_col)
        #         if new_columns.count(new_col) > 1:
        #             duplicated_columns.add(new_col)
        #     # Seleciona apenas as colunas unicas
        #     df_processed = df_processed[unique_columns]
        #     # Filtrar as linhas em df_current onde df_current['GbCity_EN'] está em df_index['City_EN']
        #     edges_and_weights(df_index, df_processed, "GbCity_EN", "City_EN", edges_weights)
        else:
            print(f"Erro no arquivo: {file}")

    # Inicializando o array de edges
    edges = np.zeros((2, edges_weights.__len__()), dtype=np.int64)

    # Inicializando o array de weights
    weights = np.zeros(edges_weights.__len__(), dtype=np.float64)

    municipios = list(df_index['City_EN'])
    # municipio_para_indice = {municipio: i for i, municipio in enumerate(municipios)}

    # Processamento para cada elemento do dicionário
    for i, (key, value) in enumerate(edges_weights.items()):
        # Inserindo os valores de edges
        edges[:, i] = np.array(value[0], dtype=np.int64)
        # Calculando e armazenando os valores dos weights
        weights[i] = value[1] / value[2]

        src, dst = edges[0, i], edges[1, i]
        G.add_edge(src, dst, weight=weights[i])
        G.add_nodes_from([src], City_EN=municipios[src])
        G.add_nodes_from([dst], City_EN=municipios[dst])

    # --- 4. Salvamento ---
    print(f"Salvando resultados em arquivos com sufixo: '{sufixo_arquivo}'")
    os.makedirs(pasta_pre_processed, exist_ok=True)

    np.save(caminho_edges, edges)
    np.save(caminho_weights, weights)
    nx.write_graphml(G, caminho_grafo)

    print(f"Rede '{sufixo_arquivo}' construída e salva com sucesso.")

    if backbone:
        g_strength = G.degree(weight='weight')
        g_degree = G.degree()
        ignored_nodes = {node: 0 for node in list(G.nodes)}

        # Aplicando o algoritmo de backbone
        backbone_g, municipios_g = extract_backbone(G, alpha, g_strength, g_degree, ignored_nodes)

        municipio_para_indice_g = {municipio: i for i, municipio in enumerate(municipios_g)}

        # Obtendo o número de arestas
        num_edges = backbone_g.number_of_edges()

        # Criando os arrays NumPy
        edges_g = np.zeros((2, num_edges), dtype=np.int32)
        weights_g = np.zeros(num_edges, dtype=np.float32)

        # Preenchendo os arrays
        i = 0
        for u, v, data in backbone_g.edges(data=True):
            src = municipio_para_indice_g[backbone_g.nodes[u]['City_EN']]
            trg = municipio_para_indice_g[backbone_g.nodes[v]['City_EN']]
            edges_g[:, i] = [src, trg]
            weights_g[i] = data['weight']
            i += 1

        # Salvando os resultados
        # --- 4. Salvamento ---
        print(f"Salvando resultados em arquivos com sufixo: '{sufixo_arquivo}'")
        os.makedirs(pasta_pre_processed, exist_ok=True)

        np.save(caminho_edges, edges_g)
        np.save(caminho_weights, weights_g)
        nx.write_graphml(backbone_g, caminho_grafo)

        print(f"Rede '{sufixo_arquivo}' construída e salva com sucesso.")

        return edges_g, weights_g, backbone_g

    return edges, weights, G


def populacao_normalizada(pasta_pre_processed, pasta_raw_data, city_to_index, sufixo_arquivo):
    """
    Calcula a população normalizada para cada município.
    """
    # Caminho para o arquivo a ser salvo usando o sufixo
    caminho_pop_norm = os.path.join(pasta_pre_processed, f'pop_normalizada_{sufixo_arquivo}.pkl')

    # Verifica se o arquivo já existe
    if os.path.exists(caminho_pop_norm):
        print("Carregando dicionário de população normalizada...")
        with open(caminho_pop_norm, 'rb') as f:
            pop_normalizada = pickle.load(f)
    else:
        # Carregar o DataFrame com a população da China
        df_pop_china = pl.read_excel(source=os.path.join(pasta_raw_data, "worldcities.xlsx"),
                                     engine="xlsx2csv", read_options={"has_header": True})

        # Filtrar por país "China" e por população maior que zero
        df_china = df_pop_china.filter(
            (pl.col("country") == "China") &
            (pl.col("population").is_not_null()) &
            (pl.col("population") > 0)
        ).select(["city", "city_ascii", "population", "country"])

        # Filtrar para incluir apenas as cidades presentes em city_to_index
        df_cities_filtered = df_china.filter(pl.col("city_ascii").is_in(city_to_index.keys())).sort("city_ascii")

        # Remover duplicatas, mantendo a primeira ocorrência
        df_cities_filtered = df_cities_filtered.unique(subset="city", keep="first")

        # Criar o dicionário com a normalização para 100 mil habitantes
        pop_normalizada = {
            row["city_ascii"]: row["population"] / 100_000 for row in df_cities_filtered.to_dicts()
        }

        # Salvando o dicionário
        with open(caminho_pop_norm, 'wb') as f:
            pickle.dump(pop_normalizada, f)

    return pop_normalizada


def calcular_targets(dataset, pop_norm, thresholds, janela=7):
    """
    Calcula os targets de classificação com base no threshold fornecidos e dados de população normalizada.

    Args:
        dataset (numpy.ndarray): Conjunto de dados espaço temporal.
        pop_norm (list): Array da população normalizada.
        janela (int): Tamanho da janela para a média móvel (padrão 7).
        threshold (int): Limite para classificação binária.

    Returns:
        numpy.ndarray: Targets de classificação binário.
    """
    # --- 1. Preparação dos Inputs ---
    if not isinstance(thresholds, (list, dict)):
        raise TypeError("O parâmetro 'thresholds' deve ser uma lista ou um dicionário.")

    # Se for um dicionário, extrai os valores e os ordena
    if isinstance(thresholds, dict):
        thresholds_list = sorted(list(thresholds.values()))
    else:
        thresholds_list = sorted(thresholds)

    # Garante que pop_norm seja um array numpy para operações vetoriais
    pop_norm = np.array(pop_norm)

    # --- 2. Expansão do Dataset e Inicialização ---
    dataset_expandido = np.vstack([np.zeros((janela - 1, dataset.shape[1])), dataset])
    targets = np.zeros_like(dataset, dtype=np.int32)
    dias_regressao = np.arange(janela).reshape(-1, 1)

    # --- 3. Loop Principal para Cálculo dos Targets ---
    for dia in range(janela - 1, dataset.shape[0] + janela - 1):
        # Seleciona os dados da janela atual
        janela_dados = dataset_expandido[dia - janela + 1:dia + 1]

        # --- Cálculo da Métrica Combinada (Média Móvel + Tendência) ---
        media_movel = np.mean(janela_dados, axis=0)

        # Evita divisão por zero na normalização
        media_normalizada = np.divide(media_movel, pop_norm, out=np.zeros_like(media_movel), where=pop_norm != 0)

        # Regressão Linear para calcular a tendência na janela
        X_mean = np.mean(dias_regressao)
        Y_mean = np.mean(janela_dados, axis=0)
        # Covariância e Variância para o cálculo da inclinação (slope)
        XY_cov = np.sum((dias_regressao - X_mean) * (janela_dados - Y_mean), axis=0)
        XX_cov = np.sum((dias_regressao - X_mean) ** 2)

        # O cálculo original da taxa de tendência parecia multiplicar a média pela inclinação
        # Mantendo a fórmula original: trend_rate = Y_mean * (slope)
        slope = np.divide(XY_cov, XX_cov, out=np.zeros_like(XY_cov), where=XX_cov != 0)
        trend_rate = Y_mean * slope

        # A métrica final combina a média normalizada com a taxa de tendência
        # Esta métrica será usada para a classificação
        metrica_final = media_normalizada * trend_rate

        # Substitui possíveis NaNs por 0 para evitar erros na classificação
        metrica_final_sem_nan = np.nan_to_num(metrica_final)

        # --- Classificação com Base nos Thresholds Dinâmicos ---
        # np.digitize classifica cada valor da métrica nos "bins" definidos pelos thresholds
        targets[dia - (janela - 1), :] = np.digitize(metrica_final_sem_nan, bins=thresholds_list)

    # --- 4. Análise da Distribuição das Classes ---
    classes, contagens = np.unique(targets, return_counts=True)
    dist_df = pd.DataFrame({
        'Classe': classes,
        'Contagem': contagens
    }).sort_values(by='Classe')

    # --- 5. Geração do Gráfico e Salvamento em PDF ---
    plt.figure(figsize=(12, 7))
    bars = plt.bar(dist_df['Classe'], dist_df['Contagem'], color='skyblue', tick_label=dist_df['Classe'])
    plt.xlabel('Classe do Target', fontsize=12)
    plt.ylabel('Contagem Total', fontsize=12)
    plt.title(f'Distribuição das Classes (Janela={janela})', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(dist_df['Classe'])  # Garante que todos os labels de classe sejam mostrados

    # Adiciona a contagem no topo de cada barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), va='bottom', ha='center')

    # Salva a figura em formato PDF
    plt.savefig('distribuicao_classes.pdf', format='pdf', bbox_inches='tight')
    plt.close()  # Fecha a figura para não ser exibida no notebook/script

    print("Figura com a distribuição das classes foi salva em 'distribuicao_classes.pdf'")

    return targets, dist_df


def construindo_StaticGraphTemporalSignal(dataset, pop_norm, lag, out, edges, weights):
    """
    Constrói o StaticGraphTemporalSignal com base no conjunto de dados fornecido utilizando janela deslizante.

    Args:
        dataset (numpy.ndarray): Conjunto de dados espaço temporal.
        pop_norm (list): Array da população normalizada.
        lag (int): Número de passos de tempo anteriores usados como features.
        out (int): Número de passos de tempo futuros usados como targets.
        edges (numpy.ndarray): Array numpy contendo os índices de arestas.
        weights (numpy.ndarray): Array numpy contendo os pesos das arestas.

    Returns:
        tuple: Tupla contendo o dataset_classification, dataset_regression, mean_dataset e std_dataset.
    """
    print("Construindo StaticGraphTemporalSignal")

    # Calculando a média e o desvio padrão
    mean_dataset = np.mean(dataset, axis=0)
    std_dataset = np.std(dataset, axis=0)

    # Normalização (Z)
    numerador = np.subtract(dataset, mean_dataset)
    denominador = std_dataset
    dataset_std = np.zeros(numerador.shape)
    np.divide(numerador, denominador, out=dataset_std, where=denominador != 0)

    # Separando Features e Targets utilizando janela deslizante
    num_amostras = dataset.shape[0] - lag - out + 1
    features = [dataset[i:i + lag].T for i in range(num_amostras)]
    features_std = [dataset_std[i:i + lag].T for i in range(num_amostras)]

    # print("Executando para dados ACUMULADOS:")
    # targets_classification, dist_df_ac = calcular_targets(
    #     dataset=dataset,
    #     pop_norm=pop_norm,
    #     tipo_dado='acumulado',
    #     thresholds=[10],
    #     janela=7
    # )

    print("Executando para dados de VARIAÇÃO:")
    targets_classification, dist_df_ac = calcular_targets(
        dataset=dataset,
        pop_norm=pop_norm,
        thresholds=[1],  # [1, 10, 25],
        janela=7
    )

    targets = [targets_classification[i + lag:i + lag + out].T for i in range(num_amostras)]
    targets_std = [dataset_std[i + lag:i + lag + out].T for i in range(num_amostras)]

    # Salva o dataset de classificação
    dataset_classification = StaticGraphTemporalSignal(edges, weights, features_std, targets)

    # Salva o dataset de regressão
    dataset_regression = StaticGraphTemporalSignal(edges, weights, features_std, targets_std)

    return dataset_classification, dataset_regression, mean_dataset, std_dataset


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def processar_iteracao(args, edges, weights, pasta_results, dataset, pop_norm, device, train_ratio=0.8):
    lag, out, nameModel, rep, task_type, K, hidden_channel = args
    print(
        f"Lag: {lag} Out: {out} Modelo: {nameModel} Task: {task_type} K:{K} HC:{hidden_channel} Repetição: {rep} Device: {device}")

    pasta_results = os.path.join(pasta_results, f'k_{K}_hc_{hidden_channel}')
    if not os.path.exists(pasta_results):
        os.makedirs(pasta_results)

    # Seed base derivada de parâmetros (ex: lag, out e rep)
    base_seed = 42 + rep + lag * 100 + out * 10000  # Ajuste essa fórmula conforme necessário

    # Fixa a seed
    fix_seed(base_seed)

    # Define o caminho para a pasta 'results' dentro do projeto
    pasta_results_model = os.path.join(pasta_results, nameModel, f'lags_{lag}_out_{out}')

    # Verifica se o diretório de pasta_results existe, se não, cria-o
    if not os.path.exists(pasta_results_model):
        os.makedirs(pasta_results_model)

    file_y_real = pasta_results_model + f'/y_real_rep_{rep}.npy'
    file_y_pred = pasta_results_model + f'/y_pred_rep_{rep}.npy'
    file_y_real_no_norm = pasta_results_model + f'/y_real_no_norm_rep_{rep}.npy'
    file_y_pred_no_norm = pasta_results_model + f'/y_pred_no_norm_rep_{rep}.npy'

    if task_type == 'regression':
        if os.path.exists(file_y_real) and os.path.exists(file_y_pred) and os.path.exists(
                file_y_real_no_norm) and os.path.exists(file_y_pred_no_norm):
            print(f"Modelo {nameModel} lag {lag} out {out} rep {rep} já foi processado.")
            return
    elif os.path.exists(file_y_real) and os.path.exists(file_y_pred):
        print(f"Modelo {nameModel} lag {lag} out {out} rep {rep} já foi processado.")
        return

    # Obtendo os datasets para df_totalCases
    dataset_classification, dataset_regression, mean_dataset, std_dataset = (
        construindo_StaticGraphTemporalSignal(dataset, list(pop_norm.values()), lag, out, edges, weights))

    # Dividindo o dataset em treino e teste
    if task_type == 'classification':
        train_dataset, test_dataset = temporal_signal_split(dataset_classification, train_ratio=train_ratio)
    else:  # regression
        train_dataset, test_dataset = temporal_signal_split(dataset_regression, train_ratio=train_ratio)

    # Carregando Model
    if nameModel == 'GCRN':
        model = GCRN(in_channels=lag, out_channels=hidden_channel, K=K, out=out, num_classes=2, task_type=task_type)
    elif nameModel == 'GCLSTM':
        model = GCLSTM(in_channels=lag, out_channels=hidden_channel, K=K, out=out, num_classes=2, task_type=task_type)
    elif nameModel == 'LSTM':
        model = TemporalLSTM(input_size=1, hidden_size=hidden_channel, num_outputs=out, num_classes=2,
                             task_type=task_type, num_layers=3, dropout=0.2)

    # Mova o modelo para a GPU
    model.to(device)

    # warmup_epochs = 3
    # initial_lr = 1e-5
    # base_lr = 1e-2
    # min_lr = 1e-5
    # weight_decay = 5e-3
    # epochs = 100

    # Configurações
    epochs = 100
    warmup_epochs = 5
    initial_lr = 1e-5
    base_lr = 1e-2
    min_lr = 1e-4
    weight_decay = 1e-3

    # Otimizador
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Agendador de taxa de aprendizado combinado (Warmup + Cosine)
    def warmup_cosine_scheduler(epoch):
        if epoch < warmup_epochs:
            # Warmup linear
            return initial_lr + (base_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_scheduler)

    # Early Stopping
    # early_stopping = EarlyStopping(patience=10, min_delta=0.005)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    train_size = int(0.8 * train_dataset.snapshot_count)

    # Loop de treinamento modificado
    best_val_loss = float('inf')

    # Progresso do treinamento
    for epoch in tqdm(range(1, epochs + 1), desc="Épocas", unit="época"):
        model.train()
        train_loss = 0
        train_samples = 0

        model.eval()
        val_loss = 0
        val_samples = 0

        for time, snapshot in enumerate(train_dataset):
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            snapshot.edge_attr = snapshot.edge_attr.to(device)
            snapshot.y = snapshot.y.to(device)

            if time < train_size:  # Treino (80% iniciais)
                model.train()
                optimizer.zero_grad()

                # Forward pass
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                # Função de custo
                if task_type == 'classification':
                    y_hat = y_hat.view(-1, model.num_classes)
                    snapshot_y = snapshot.y.reshape(-1).long()
                    loss = F.cross_entropy(y_hat, snapshot_y)
                else:
                    loss = torch.mean((y_hat - snapshot.y) ** 2)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_samples += 1

            else:  # Validação (20% finais)
                model.eval()
                with torch.no_grad():
                    # Forward pass
                    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

                    # Função de custo
                    if task_type == 'classification':
                        y_hat = y_hat.view(-1, model.num_classes)
                        snapshot_y = snapshot.y.reshape(-1).long()
                        loss = F.cross_entropy(y_hat, snapshot_y)
                    else:
                        loss = torch.mean((y_hat - snapshot.y) ** 2)

                    val_loss += loss.item()
                    val_samples += 1

        # Calcula médias e aplica early stopping
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples

        early_stopping(avg_val_loss)
        scheduler.step()

        print(f"Epoch {epoch}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(pasta_results_model, 'best_model.pth'))

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Treinamento concluído.")

    print("Iniciando o teste...")
    model.eval()

    y_real = np.zeros(
        (test_dataset.snapshot_count, test_dataset[0].y.shape[0], test_dataset[0].y.shape[1]))
    y_pred = np.zeros(
        (test_dataset.snapshot_count, test_dataset[0].y.shape[0], test_dataset[0].y.shape[1]))

    if task_type == 'classification':
        for time, snapshot in enumerate(test_dataset):
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            snapshot.edge_attr = snapshot.edge_attr.to(device)
            snapshot.y = snapshot.y.to(device)

            with torch.no_grad():
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            y_hat_argmax = torch.argmax(y_hat, dim=-1).cpu().numpy()

            y_real[time] = snapshot.y.cpu().numpy()
            y_pred[time] = y_hat_argmax

    else:  # regression

        y_real_no_norm = np.zeros(
            (test_dataset.snapshot_count, test_dataset[0].y.shape[0], test_dataset[0].y.shape[1]))
        y_pred_no_norm = np.zeros(
            (test_dataset.snapshot_count, test_dataset[0].y.shape[0], test_dataset[0].y.shape[1]))

        for time, snapshot in enumerate(test_dataset):
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            snapshot.edge_attr = snapshot.edge_attr.to(device)
            snapshot.y = snapshot.y.to(device)

            with torch.no_grad():
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            y_real[time] = snapshot.y.cpu().numpy()
            y_pred[time] = y_hat.cpu().numpy()

            # Revertendo a normalização
            y_real_no_norm[time] = (
                    (y_real[time] * np.repeat(std_dataset, test_dataset[0].y.shape[1]).
                     reshape(-1, test_dataset[0].y.shape[1]))
                    + np.repeat(mean_dataset, test_dataset[0].y.shape[1]).
                    reshape(-1, test_dataset[0].y.shape[1]))

            y_pred_no_norm[time] = (
                    (y_pred[time] * np.repeat(std_dataset, test_dataset[0].y.shape[1]).
                     reshape(-1, test_dataset[0].y.shape[1]))
                    + np.repeat(mean_dataset, test_dataset[0].y.shape[1]).
                    reshape(-1, test_dataset[0].y.shape[1]))

    print("Fim do teste")

    np.save(file_y_real, y_real)
    np.save(file_y_pred, y_pred)

    if task_type == 'regression':
        np.save(file_y_real_no_norm, y_real_no_norm)
        np.save(file_y_pred_no_norm, y_pred_no_norm)

    print(f"Modelo {nameModel} lag {lag} out {out} rep {rep} concluída.")


if __name__ == "__main__":

    # Define o pais
    country = 'China'

    # Define o tipo de tarefa
    task_type = 'classification'  # 'classification' ou 'regression'

    # Define se a rede fará extração de backbone e o alpha
    backbone = True
    alpha = 0.01

    # Obtém o diretório atual do script (pasta 'codes')
    pasta_atual = os.path.dirname(os.path.realpath(__file__))

    # Retorna o diretório pai (pasta do projeto)
    pasta_projeto = os.path.dirname(pasta_atual)

    # Define o caminho para a pasta 'raw_data' dentro do projeto
    pasta_raw_data = os.path.join(pasta_projeto, 'raw_data', country)

    # Define o caminho para a pasta 'pre_processed' dentro do projeto
    pasta_pre_processed = os.path.join(pasta_projeto, 'pre_processed', country)

    # Verifica se o diretório de pasta_pre_processed existe, se não, cria-o
    if not os.path.exists(pasta_pre_processed):
        os.makedirs(pasta_pre_processed)

    # Caminho para a pasta "dataverse_files"
    networks_dir = os.path.join(pasta_raw_data, "dataverse_files")

    dataset_in_out = 'in'

    date_networks, df_networks = construindo_dataframe_networks(networks_dir, dataset_in_out)

    # Lista com todos os tipos de grafo que a função pode gerar
    tipos_de_grafo = [
        "grafo_original"
    ]

    # Itera sobre cada tipo de grafo na lista
    for tipo in tipos_de_grafo:
        print("-" * 50)
        print(f"PROCESSANDO: Tipo='{tipo}', Backbone={backbone}")

        # --- Define o caminho para a pasta de resultados ---
        sufixo_arquivo = tipo
        subpasta_resultado = tipo
        if backbone:
            sufixo_arquivo += f"_backbone_{alpha}"
            subpasta_resultado = os.path.join(tipo, 'backbone_alpha_{:.0f}'.format(alpha * 100))

        # 1. Construa o caminho original normalmente
        pasta_results = os.path.join(pasta_projeto, 'results_daily', country, task_type, subpasta_resultado)

        # 2. Verifique e modifique a string 'pasta_results' se a condição for verdadeira
        caminho_a_verificar = '/media/work/fernandoduarte/ASOP/'
        if pasta_results.startswith(caminho_a_verificar):
            # Substitui apenas a primeira ocorrência de 'work' por 'data' para segurança
            pasta_results = pasta_results.replace('work', 'data', 1)
            print(f"AVISO: O caminho de resultados foi modificado para usar '/media/data/'.")

        # 3. O restante do código agora usa a variável 'pasta_results' (seja a original ou a modificada)
        os.makedirs(pasta_results, exist_ok=True)
        print(f"Pasta de pré-processamento: {pasta_pre_processed}")
        print(f"Pasta final de resultados: {pasta_results}")

        # Construindo o DataFrame para os dados de COVID-19
        df_covid_temporal = construindo_dataframe_covid(pasta_raw_data)

        # Construindo o DataFrame com as datas comuns entre os arquivos de redes e os dados de COVID-19
        df_common_dates = construindo_dataframe_common_dates(df_covid_temporal, date_networks, df_networks)

        # Construindo o grafo espaço-temporal
        df_index, df_graph_temporal = construir_grafo_temporal(networks_dir, df_common_dates, df_covid_temporal,
                                                               dataset_in_out, cumulativo=False)

        # Criar um dicionário que mapeia cidades para índices
        city_to_index = {city: index for index, city in enumerate(df_graph_temporal['City_name'])}

        # População normalizada
        pop_norm = populacao_normalizada(pasta_pre_processed, pasta_raw_data, city_to_index, sufixo_arquivo)

        # Construir a rede de mobilidade
        edges, weights, G = construindo_rede_mobilidade(pasta_pre_processed, df_index, df_common_dates,
                                                        dataset_in_out, backbone=backbone, alpha=alpha)

        # Remova as duas primeiras colunas do DataFrame df_graph_temporal (City_name e City_code)
        dataset = df_graph_temporal[:, 2:].to_numpy().T

        # Definindo parametros para os modelos
        train_ratio = 0.8  # 80% dos dados para treino
        reps = 5  # 5 repetições
        K = [1]  # 1 pulos
        hidden_channels = [256]
        lags = outs = 14  # 14 lags e 14 outs

        namemodels = ['GCRN', 'GCLSTM', 'LSTM']
        # 'GCRN', 'GCLSTM', 'DCRN', 'STGCN', 'LSTM', 'Timesfm', 'Dumb', 'STSGT'

        # Verifique se a MPS está disponível
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print('MPS disponível. Usando a MPS...')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print('GPU disponível. Usando a GPU...')
        else:
            device = torch.device('cpu')
            print('CPU disponível. Usando a CPU...')

        # Iterável dos valores de lags e outs
        iteravel = product(range(1, lags + 1), range(1, outs + 1), namemodels, range(reps), [task_type], K,
                           hidden_channels)

        # Função parcial com os parâmetros que serão iterados
        partial_processar_iteracao = partial(processar_iteracao, edges=edges, weights=weights,
                                             pasta_results=pasta_results,
                                             dataset=dataset, pop_norm=pop_norm, device=device,
                                             train_ratio=train_ratio)

        for item in iteravel:
            partial_processar_iteracao(item)

        print(f"Processamento de '{tipo}' (Backbone={backbone}) concluído.")

    print("-" * 50)
    print("Todas as redes foram geradas com sucesso!")
