# coding: utf-8
# Importing libraries
import os
import datetime
import glob
import pickle
import random
import numpy as np
import networkx as nx
import polars as pl
from functools import partial
from itertools import product
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from models.gcn_based_rnn_model import GCRN
from models.gcn_based_lstm_model import GCLSTM
from models.dcrnn_model import DCRN


# import sys
#
# sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
# from gcn_based_rnn_model import GCRN
# from gcn_based_lstm_model import GCLSTM
# from dcrnn_model import DCRN

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


def construir_grafo_temporal(networks_dir, df_common_dates, df_covid_temporal, dataset_in_out='in'):
    """
    Constrói o grafo espaço-temporal usando os dados de redes e COVID-19.

    Args:
        networks_dir (str): O diretório contendo os arquivos de redes.
        df_common_dates (polars.DataFrame): O DataFrame contendo as datas comuns entre os arquivos de redes e os dados de COVID-19.
        df_covid_temporal (polars.DataFrame): O DataFrame contendo os dados de COVID-19.
        save_path (str): O diretório onde os DataFrames serão salvos.
        dataset_in_out (str, optional): O tipo de conjunto de dados a ser usado, 'in' para entrada e 'out' para saída. Defaults to 'in'.

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

    # Aplique o filtro no DataFrame df_covid_temporal
    df_graph_temporal = df_covid_temporal.filter(df_covid_temporal['City_name'].is_in(df_index['City_EN']) &
                                                 df_covid_temporal['City_name'].is_in(df_china['city_ascii']))

    # Crie um filtro para selecionar apenas as cidades presentes no df_graph_temporal
    index_filter = df_index['City_EN'].is_in(df_graph_temporal['City_name'])

    # Aplique o filtro no DataFrame df_index
    df_index = df_index.filter(index_filter)

    # Seleciona apenas as colunas (Datas) que são comuns a ambos os DataFrames
    common_columns = ['City_name', 'City_code'] + df_common_dates['date_networks'].to_list()
    df_graph_temporal = df_graph_temporal[common_columns]

    # Selecionar as colunas desejadas e aplicar a soma cumulativa horizontal
    cumulative_sum = df_graph_temporal.select(pl.cum_sum_horizontal(common_columns[2:])).unnest('cum_sum')

    # Substituir as colunas originais pelas selecionadas juntamente com as colunas cumulativas
    df_graph_temporal = df_graph_temporal.select(common_columns[:2]).with_columns(cumulative_sum)

    # Reordena o DataFrame por 'City_name'
    df_index = df_index.sort('City_EN')
    # Reordena o DataFrame por 'City_name'
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


def construindo_rede_mobilidade(df_index, df_common_dates, pasta_pre_processed, dataset_in_out='in', backbone=False,
                                threshold=0.01):
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

    if backbone:
        # Caminhos para os arquivos a serem salvos
        caminho_edges = os.path.join(pasta_pre_processed, f'edges_back_{threshold}.npy')
        caminho_weights = os.path.join(pasta_pre_processed, f'weights_back_{threshold}.npy')
        caminho_grafo = os.path.join(pasta_pre_processed, f'grafo_back_{threshold}.pkl')
    else:
        # Caminhos para os arquivos a serem salvos
        caminho_edges = os.path.join(pasta_pre_processed, 'edges.npy')
        caminho_weights = os.path.join(pasta_pre_processed, 'weights.npy')
        caminho_grafo = os.path.join(pasta_pre_processed, f'grafo.pkl')

    # Verifica se os arquivos já existem
    if os.path.exists(caminho_edges) and os.path.exists(caminho_weights) and os.path.exists(caminho_grafo):
        print("Carregando dados pré-processados...")
        edges = np.load(caminho_edges)
        weights = np.load(caminho_weights)
        with open(caminho_grafo, 'rb') as f:
            G = pickle.load(f)
        return edges, weights, G
    else:

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

        # Salvando os resultados
        np.save(caminho_edges, edges)
        np.save(caminho_weights, weights)
        with open(caminho_grafo, 'wb') as f:
            pickle.dump(G, f)

        print("Rede de mobilidade construída.")

        if backbone:
            g_strength = G.degree(weight='weight')
            g_degree = G.degree()
            ignored_nodes = {node: 0 for node in list(G.nodes)}

            # Aplicando o algoritmo de backbone
            backbone_g, municipios_g = extract_backbone(G, threshold, g_strength, g_degree, ignored_nodes)

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
            np.save(caminho_edges, edges_g)
            np.save(caminho_weights, weights_g)
            with open(caminho_grafo, 'wb') as f:
                pickle.dump(backbone_g, f)

            return edges_g, weights_g, backbone_g

        return edges, weights, G


def populacao_normalizada(pasta_raw_data, city_to_index):
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

    # # Verificar se há cidades com nomes duplicados
    # cidades_duplicadas = df_cities_filtered.group_by("city").len().filter(pl.col("len") > 1)
    # if not cidades_duplicadas.is_empty():
    #     print(f"Cidades duplicadas encontradas: {cidades_duplicadas['city'].to_list()}")
    #     print("Numero de cidades duplicadas:", cidades_duplicadas.shape[0])
    #     print("Removendo cidades duplicadas...")
    #     # Remover cidades duplicadas
    df_cities_filtered = df_cities_filtered.unique(subset="city", keep="first")

    # Criar o dicionário com a normalização para 100 mil habitantes
    dict_pop_normalizada = {
        row["city_ascii"]: row["population"] / 100_000 for row in df_cities_filtered.to_dicts()
    }

    return dict_pop_normalizada


def calcular_targets(dataset, pop_norm, janela=7, threshold=10):
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

    # Cria uma cópia do dataset e adiciona 'janela - 1' linhas no topo com zeros
    dataset_expandido = np.vstack([np.zeros((janela - 1, dataset.shape[1])), dataset])

    # Inicializa o array targets com zeros (mesmo tamanho do dataset original)
    targets = np.zeros_like(dataset, dtype=np.int32)

    # Pré-calcula as features (dias) da regressão
    dias = np.arange(janela).reshape(-1, 1)

    # Calcula a média móvel de 7 dias e aplica as classificações
    for dia in range(janela - 1, dataset.shape[0] + janela - 1):
        media_movel = np.mean(dataset_expandido[dia - janela + 1:dia + 1], axis=0)
        media_normalizada = media_movel / pop_norm

        # Calculando a inclinação para todas as cidades ao mesmo tempo (regressão linear)
        janela_dados = dataset_expandido[dia - janela + 1:dia + 1]
        X_mean = np.mean(dias)
        Y_mean = np.mean(janela_dados, axis=0)
        XY_cov = np.sum((dias - X_mean) * (janela_dados - Y_mean), axis=0)
        XX_cov = np.sum((dias - X_mean) ** 2)
        trend_rate = Y_mean * XY_cov / XX_cov

        # Combinando métricas para cada cidade
        for cidade in range(media_normalizada.shape[0]):

            # Combina a métrica com a taxa de tendência
            media_combinada = media_normalizada[cidade] * trend_rate[cidade]

            if media_combinada > threshold:
                targets[dia - (janela - 1), cidade] = 1  # yellow
            else:
                targets[dia - (janela - 1), cidade] = 0  # green

    return targets


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

    # Calculando os targets do classificador
    targets_classification = calcular_targets(dataset, pop_norm, janela=7, threshold=10)

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


def processar_iteracao(args, edges, weights, pasta_results, dataset, pop_norm, device, filters=32, K=2, lr=0.01,
                       epochs=100, train_ratio=0.8, task_type='regression'):
    lag, out, nameModel, rep = args
    print(f"Lag: {lag} Out: {out} Modelo: {nameModel} Task: {task_type} Repetição: {rep} Device: {device}")

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
        model = GCRN(in_channels=lag, out_channels=filters, K=K, out=out, num_classes=2, task_type=task_type)
    elif nameModel == 'GCLSTM':
        model = GCLSTM(in_channels=lag, out_channels=filters, K=K, out=out, num_classes=2, task_type=task_type)
    elif nameModel == 'DCRN':
        model = DCRN(in_channels=lag, out_channels=filters, K=K, out=out, num_classes=2, task_type=task_type)

    # Mova o modelo para a GPU
    model.to(device)

    # Inicializa o otimizador
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Início do treinamento
    print("Iniciando o treinamento...")
    model.train()

    # Progresso do treinamento
    for epoch in tqdm(range(1, epochs + 1), desc="Épocas", unit="época"):
        for time, snapshot in enumerate(train_dataset):
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            snapshot.edge_attr = snapshot.edge_attr.to(device)
            snapshot.y = snapshot.y.to(device)

            # Forward pass
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            # Função de custo
            if task_type == 'classification':
                y_hat = y_hat.view(-1, model.num_classes)
                snapshot_y = snapshot.y.reshape(-1).long()
                cost = F.cross_entropy(y_hat, snapshot_y)
            else:
                cost = torch.mean((y_hat - snapshot.y) ** 2)

            # Backward pass e otimização
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

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
    task_type = 'regression'  # 'classification' ou 'regression'

    # Define se a rede fará extração de backbone e o threshold
    backbone = True
    threshold = 0.01

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

    # Define o caminho para a pasta 'results' dentro do projeto
    if backbone:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type,
                                     'backbone_threshold_{:.0f}'.format(threshold * 100))
    else:
        pasta_results = os.path.join(pasta_projeto, 'results', country, task_type)

    # Verifica se o diretório de pasta_results existe, se não, cria-o
    if not os.path.exists(pasta_results):
        os.makedirs(pasta_results)

    # Caminho para a pasta "dataverse_files"
    networks_dir = os.path.join(pasta_raw_data, "dataverse_files")

    dataset_in_out = 'in'

    date_networks, df_networks = construindo_dataframe_networks(networks_dir, dataset_in_out)

    # Construindo o DataFrame para os dados de COVID-19
    df_covid_temporal = construindo_dataframe_covid(pasta_raw_data)

    # Construindo o DataFrame com as datas comuns entre os arquivos de redes e os dados de COVID-19
    df_common_dates = construindo_dataframe_common_dates(df_covid_temporal, date_networks, df_networks)

    # Construindo o grafo espaço-temporal
    df_index, df_graph_temporal = construir_grafo_temporal(networks_dir, df_common_dates, df_covid_temporal,
                                                           dataset_in_out)

    # Criar um dicionário que mapeia cidades para índices
    city_to_index = {city: index for index, city in enumerate(df_graph_temporal['City_name'])}

    # População normalizada
    pop_norm = populacao_normalizada(pasta_raw_data, city_to_index)

    # Construir a rede de mobilidade
    edges, weights, G = construindo_rede_mobilidade(df_index, df_common_dates, pasta_pre_processed,
                                                    dataset_in_out, backbone=backbone, threshold=threshold)

    # Remova as duas primeiras colunas do DataFrame df_graph_temporal (City_name e City_code)
    dataset = df_graph_temporal[:, 2:].to_numpy().T

    # Definindo numero de lags e saidas
    lags = 14
    outs = 14

    # Definindo parametros para os modelos
    train_ratio = 0.8  # 80% dos dados para treino
    reps = 2  # 2 repetições
    K = 2  # 2 pulos
    filters = 64  # 64 filtros
    lr = 0.01  # taxa de aprendizado
    epochs = 100  # 100 épocas

    namemodels = ['GCRN', 'GCLSTM']  # 'GCRN', 'GCLSTM', 'DCRN'

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
    iteravel = product(range(1, lags + 1), range(1, outs + 1), namemodels, range(reps))

    # Função parcial com os parâmetros que serão iterados pelo executor.map()
    partial_processar_iteracao = partial(processar_iteracao, edges=edges, weights=weights, pasta_results=pasta_results,
                                         dataset=dataset, pop_norm=pop_norm, device=device, filters=filters, K=K, lr=lr,
                                         epochs=epochs, train_ratio=train_ratio, task_type=task_type)

    # # Crie um pool de threads
    # with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    #     # Mapeie a função para todas as combinações de lag e out usando o pool de threads
    #     executor.map(partial_processar_iteracao, iteravel)

    for item in iteravel:
        partial_processar_iteracao(item)
