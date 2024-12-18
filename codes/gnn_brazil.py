# coding: utf-8
# Importing libraries
import os
import pickle
import random
import requests
import numpy as np
import networkx as nx
import polars as pl
from functools import partial
import concurrent.futures
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
            backbone.add_nodes_from([source], ibge=graph.nodes[source]['ibge'])
            backbone.add_nodes_from([target], ibge=graph.nodes[target]['ibge'])

            codes.add(graph.nodes[source]['ibge'])
            codes.add(graph.nodes[target]['ibge'])

    return backbone, np.array(sorted(codes))


def construindo_rede_mobilidade(pasta_raw_data, pasta_pre_processed, backbone=False, threshold=0.4):
    print("Construindo a rede de mobilidade...")

    if backbone:
        # Caminhos para os arquivos a serem salvos
        caminho_edges = os.path.join(pasta_pre_processed, f'edges_back_{threshold}.npy')
        caminho_weights = os.path.join(pasta_pre_processed, f'weights_back_{threshold}.npy')
        caminho_codigos = os.path.join(pasta_pre_processed, f'codigos_municipios_back_{threshold}.pkl')
        caminho_mapeamento = os.path.join(pasta_pre_processed, f'codigo_para_indice_back_{threshold}.pkl')
        caminho_grafo = os.path.join(pasta_pre_processed, f'grafo_back_{threshold}.pkl')
    else:
        # Caminhos para os arquivos a serem salvos
        caminho_edges = os.path.join(pasta_pre_processed, 'edges.npy')
        caminho_weights = os.path.join(pasta_pre_processed, 'weights.npy')
        caminho_codigos = os.path.join(pasta_pre_processed, 'codigos_municipios.pkl')
        caminho_mapeamento = os.path.join(pasta_pre_processed, 'codigo_para_indice.pkl')
        caminho_grafo = os.path.join(pasta_pre_processed, f'grafo.pkl')

    # Verifica se os arquivos já existem
    if os.path.exists(caminho_edges) and os.path.exists(caminho_weights) and os.path.exists(caminho_grafo) and\
            os.path.exists(caminho_codigos) and os.path.exists(caminho_mapeamento):
        print("Carregando dados pré-processados...")
        edges = np.load(caminho_edges)
        weights = np.load(caminho_weights)
        with open(caminho_codigos, 'rb') as f:
            codigos_municipios = pickle.load(f)
        with open(caminho_mapeamento, 'rb') as f:
            codigo_para_indice = pickle.load(f)
        with open(caminho_grafo, 'rb') as f:
            G = pickle.load(f)
        return edges, weights, codigos_municipios, codigo_para_indice, G
    else:
        # DataFrame do IBGE de 2016 na pasta 'raw_data'
        caminho_df_ibge = os.path.join(pasta_raw_data, 'dataset_transform_IBGE.xlsx')

        # Carregando o dataset IBGE de ligações entre cidades com pesos
        df_ibge = pl.read_excel(source=caminho_df_ibge, engine='xlsx2csv', read_options={"has_header": True})

        # Obtendo códigos únicos de ambas as colunas
        codigos_municipios = np.unique(df_ibge[['CODMUNDV_A', 'CODMUNDV_B']].to_numpy().flatten())

        # Criando o dicionário de mapeamento
        codigo_para_indice = {codigo: i for i, codigo in enumerate(codigos_municipios)}

        # Criando o grafo
        G = nx.Graph()

        # Criando arrays para armazenar edges e pesos
        edges = np.zeros((2, len(df_ibge)), dtype=np.int32)
        weights = np.zeros(len(df_ibge), dtype=np.float32)

        # Preenchendo os arrays e adicionando as arestas ao grafo
        for idx, row in enumerate(df_ibge.iter_rows(named=True)):
            src, dst = codigo_para_indice[row['CODMUNDV_A']], codigo_para_indice[row['CODMUNDV_B']]
            weight = row['VAR05'] + row['VAR06'] + row['VAR12']
            edges[:, idx] = [src, dst]
            weights[idx] = weight
            G.add_edge(src, dst, weight=weight)
            G.add_nodes_from([src], ibge=row['CODMUNDV_A'])
            G.add_nodes_from([dst], ibge=row['CODMUNDV_B'])

        # Salvando os resultados
        np.save(caminho_edges, edges)
        np.save(caminho_weights, weights)
        with open(caminho_codigos, 'wb') as f:
            pickle.dump(codigos_municipios, f)
        with open(caminho_mapeamento, 'wb') as f:
            pickle.dump(codigo_para_indice, f)
        with open(caminho_grafo, 'wb') as f:
            pickle.dump(G, f)

        print("Rede de mobilidade construída.")

        if backbone:
            g_strength = G.degree(weight='weight')
            g_degree = G.degree()
            ignored_nodes = {node: 0 for node in list(G.nodes)}

            # Aplicando o algoritmo de backbone
            backbone_g, codigos_municipios_g = extract_backbone(G, threshold, g_strength, g_degree, ignored_nodes)

            codigo_para_indice_g = {codigo: i for i, codigo in enumerate(codigos_municipios_g)}

            # Obtendo o número de arestas
            num_edges = backbone_g.number_of_edges()

            # Criando os arrays NumPy
            edges_g = np.zeros((2, num_edges), dtype=np.int32)
            weights_g = np.zeros(num_edges, dtype=np.float32)

            # Preenchendo os arrays
            i = 0
            for u, v, data in backbone_g.edges(data=True):
                src = codigo_para_indice_g[backbone_g.nodes[u]['ibge']]
                trg = codigo_para_indice_g[backbone_g.nodes[v]['ibge']]
                edges_g[:, i] = [src, trg]
                weights_g[i] = data['weight']
                i += 1

            # Salvando os resultados
            np.save(caminho_edges, edges_g)
            np.save(caminho_weights, weights_g)
            with open(caminho_codigos, 'wb') as f:
                pickle.dump(codigos_municipios_g, f)
            with open(caminho_mapeamento, 'wb') as f:
                pickle.dump(codigo_para_indice_g, f)
            with open(caminho_grafo, 'wb') as f:
                pickle.dump(backbone_g, f)

            return edges_g, weights_g, codigos_municipios_g, codigo_para_indice_g, backbone_g

        return edges, weights, codigos_municipios, codigo_para_indice, G


def obter_dados_covid_brasil(pasta_raw_data):
    print("Baixando o arquivo 'cases-brazil-cities-time_changesOnly.csv.gz'...")
    # Nome do arquivo local
    nome_arquivo_local = "cases-brazil-cities-time_changesOnly.csv.gz"
    caminho_arquivo = os.path.join(pasta_raw_data, nome_arquivo_local)

    # Verifica se o arquivo 'cases-brazil-cities-time.csv.gz' já existe na pasta 'raw_data'
    if os.path.exists(caminho_arquivo):
        print(f"O arquivo '{nome_arquivo_local}' já existe na pasta 'raw_data'.")
        return caminho_arquivo

    # URL do arquivo no GitHub
    url = "https://github.com/wcota/covid19br/raw/master/cases-brazil-cities-time_changesOnly.csv.gz"

    # Baixa o arquivo em partes
    with requests.get(url, stream=True) as response:
        # Verifica se a requisição foi bem-sucedida (código 200)
        if response.status_code == 200:
            # Salva o conteúdo do arquivo localmente em partes
            with open(caminho_arquivo, 'wb') as arquivo:
                for parte in response.iter_content(chunk_size=128):
                    arquivo.write(parte)
            print(f"Arquivo '{nome_arquivo_local}' baixado com sucesso.")
        else:
            print(f"Falha ao baixar o arquivo. Código de status: {response.status_code}")
            # Encerra a execução do script
            exit()

    # Retorna o caminho do arquivo se tudo estiver correto
    return caminho_arquivo


def dataframes_covid(pasta_pre_processed, df_cidades, datas, codigos_municipios, backbone=False, threshold=0.4):
    print("Construindo DataFrames para as variáveis de COVID-19...")

    if backbone:
        # Caminhos para os arquivos a serem salvos
        caminho_newDeaths = os.path.join(pasta_pre_processed, f'df_newDeaths_{threshold}.pkl')
        caminho_deaths = os.path.join(pasta_pre_processed, f'df_deaths_{threshold}.pkl')
        caminho_newCases = os.path.join(pasta_pre_processed, f'df_newCases_{threshold}.pkl')
        caminho_totalCases = os.path.join(pasta_pre_processed, f'df_totalCases_{threshold}.pkl')
    else:
        # Caminhos para os arquivos a serem salvos
        caminho_newDeaths = os.path.join(pasta_pre_processed, 'df_newDeaths.pkl')
        caminho_deaths = os.path.join(pasta_pre_processed, 'df_deaths.pkl')
        caminho_newCases = os.path.join(pasta_pre_processed, 'df_newCases.pkl')
        caminho_totalCases = os.path.join(pasta_pre_processed, 'df_totalCases.pkl')

    # Verifica se os arquivos já existem
    if os.path.exists(caminho_newDeaths) and os.path.exists(caminho_deaths) and \
            os.path.exists(caminho_newCases) and os.path.exists(caminho_totalCases):
        print("Carregando DataFrames pré-processados...")
        with open(caminho_newDeaths, 'rb') as f:
            df_newDeaths = pickle.load(f)
        with open(caminho_deaths, 'rb') as f:
            df_deaths = pickle.load(f)
        with open(caminho_newCases, 'rb') as f:
            df_newCases = pickle.load(f)
        with open(caminho_totalCases, 'rb') as f:
            df_totalCases = pickle.load(f)
    else:
        # Inicializando DataFrames vazios para cada variável
        df_newDeaths = pl.DataFrame({'date': datas})
        df_deaths = pl.DataFrame({'date': datas})
        df_newCases = pl.DataFrame({'date': datas})
        df_totalCases = pl.DataFrame({'date': datas})

        # Percorrendo cada cidade única
        for codigo_ibge in codigos_municipios:
            # Filtra o DataFrame original para a cidade específica
            df_cidade = df_cidades.filter(df_cidades['ibgeID'] == codigo_ibge)
            nome_cidade = df_cidade['city'][0]

            # Inicializa arrays NumPy para a cidade
            newDeaths_values = np.zeros(len(datas))
            deaths_values = np.zeros(len(datas))
            newCases_values = np.zeros(len(datas))
            totalCases_values = np.zeros(len(datas))

            # Preenche os arrays com os valores da cidade específica
            for row in df_cidade.iter_rows(named=True):
                data = row['date']
                index = np.where(datas == data)[0][0]  # Encontra o índice correspondente à data

                newDeaths_values[index] = row['newDeaths']
                newCases_values[index] = row['newCases']
                deaths_values[index] = row['deaths']
                totalCases_values[index] = row['totalCases']

            # Lidar com a condição de valores cumulativos
            for i in range(1, len(deaths_values)):
                if deaths_values[i - 1] > deaths_values[i]:
                    deaths_values[i] = deaths_values[i - 1]

                if totalCases_values[i - 1] > totalCases_values[i]:
                    totalCases_values[i] = totalCases_values[i - 1]

            # Atribui os arrays como colunas nos DataFrames
            df_newDeaths = df_newDeaths.with_columns(pl.Series(name=nome_cidade, values=newDeaths_values))
            df_newCases = df_newCases.with_columns(pl.Series(name=nome_cidade, values=newCases_values))
            df_deaths = df_deaths.with_columns(pl.Series(name=nome_cidade, values=deaths_values))
            df_totalCases = df_totalCases.with_columns(pl.Series(name=nome_cidade, values=totalCases_values))

        # Salvando os resultados
        with open(caminho_newDeaths, 'wb') as f:
            pickle.dump(df_newDeaths, f)
        with open(caminho_deaths, 'wb') as f:
            pickle.dump(df_deaths, f)
        with open(caminho_newCases, 'wb') as f:
            pickle.dump(df_newCases, f)
        with open(caminho_totalCases, 'wb') as f:
            pickle.dump(df_totalCases, f)

        print("DataFrames criados com sucesso.")
    return df_newDeaths, df_deaths, df_newCases, df_totalCases


def populacao_normalizada(pasta_pre_processed, pasta_raw_data, codigos_municipios, backbone=False, threshold=0.4):
    """
    Calcula a população normalizada para cada município.

    Args:
        pasta_pre_processed: Pasta para salvar os dados processados.
        pasta_raw_data: Pasta com os dados brutos.
        codigos_municipios: Lista de códigos IBGE dos municípios.

    Returns:
        dict: Dicionário com a população normalizada para cada município.
    """
    if backbone:
        # Caminhos para os arquivos a serem salvos
        caminho_pop_norm = os.path.join(pasta_pre_processed, f'pop_normalizada_{threshold}.pkl')
    else:
        # Caminho para o arquivo a ser salvo
        caminho_pop_norm = os.path.join(pasta_pre_processed, 'pop_normalizada.pkl')

    # Verifica se o arquivo já existe
    if os.path.exists(caminho_pop_norm):
        print("Carregando dicionário de população normalizada...")
        with open(caminho_pop_norm, 'rb') as f:
            pop_normalizada = pickle.load(f)
    else:
        # Carregando o dataset do Brasil
        df_pop_brasil = pl.read_excel(
            source=os.path.join(pasta_raw_data, "CD2022_Populacao_Coletada_Imputada_e_Total_Municipio_e_UF.xlsx"),
            engine="xlsx2csv",
            read_options={"has_header": True, "schema_overrides": {'COD. UF': str, 'COD. MUNIC': str}},
        )

        # Concatenar as colunas 'COD. UF' e 'COD. MUNIC' para formar o código IBGE
        df_pop_brasil = df_pop_brasil.with_columns(
            (pl.col("COD. UF").cast(str) + pl.col("COD. MUNIC").cast(str)).alias("codigo_ibge").cast(pl.Int64),
            pl.col("POP. TOTAL").alias("populacao")
        )

        # Filtrar para incluir apenas os municípios presentes em codigos_municipios
        df_cities_filtered = df_pop_brasil.filter(
            pl.col("codigo_ibge").is_in(codigos_municipios)
        ).select(["codigo_ibge", "populacao"])

        # Criar o dicionário com a normalização para 100 mil habitantes
        pop_normalizada = {
            row["codigo_ibge"]: row["populacao"] / 100_000 for row in df_cities_filtered.to_dicts()
        }

        # Salvando o dicionário
        with open(caminho_pop_norm, 'wb') as f:
            pickle.dump(pop_normalizada, f)

    return pop_normalizada


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
    country = 'Brazil'

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

    edges, weights, codigos_municipios, codigo_para_indice, G = (
        construindo_rede_mobilidade(pasta_raw_data, pasta_pre_processed, backbone=backbone, threshold=threshold))

    # Obtém o caminho completo para o arquivo 'casos_covid_brasil.csv.gz'
    caminho_arquivo_covid_brasil = obter_dados_covid_brasil(pasta_raw_data)

    # DataFrame para os dados completos de COVID-19 no Brasil
    dataframe_covid_brasil = pl.read_csv(caminho_arquivo_covid_brasil)

    # Extraindo as colunas necessárias do DataFrame original
    colunas_necessarias = ['date', 'city', 'ibgeID', 'newDeaths', 'deaths', 'newCases', 'totalCases']
    df_cidades = dataframe_covid_brasil[colunas_necessarias].clone()

    # Filtrando as linhas onde 'ibgeID' está presente em codigos_municipios do IBGE
    df_cidades = df_cidades.filter(df_cidades['ibgeID'].is_in(codigos_municipios))

    # Dias reportados
    datas = dataframe_covid_brasil['date'].unique().sort()

    # Cria dataframes para cada variável
    df_newDeaths, df_deaths, df_newCases, df_totalCases = dataframes_covid(pasta_pre_processed, df_cidades, datas,
                                                                           codigos_municipios, backbone=backbone,
                                                                           threshold=threshold)

    # População normalizada
    pop_norm = populacao_normalizada(pasta_pre_processed, pasta_raw_data, codigos_municipios, backbone=backbone,
                                     threshold=threshold)

    # Remova a primeira coluna do DataFrame df_totalCases (date)
    dataset = df_totalCases[:, 1:].to_numpy()

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

    # Crie um pool de threads
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # Mapeie a função para todas as combinações de lag e out usando o pool de threads
        executor.map(partial_processar_iteracao, iteravel)

    # for item in iteravel:
    #     partial_processar_iteracao(item)

    # import torch.multiprocessing as mp
    #
    # # mp.set_start_method('spawn', force=True)  # Set start method before creating processes
    # # torch.cuda.set_per_process_memory_fraction(1/8, device=device)
    # processes = []
    # for i, iter in enumerate(iteravel):
    #     p = mp.Process(target=processar_iteracao, args=(iter, edges, weights,
    #                                                     pasta_results, dataset,
    #                                                     pop_norm, device, filters,
    #                                                     K, lr, epochs, train_ratio,
    #                                                     task_type), name=f"Iteração-{i}")
    #     p.start()
    #     processes.append(p)
    #     print(f"Iniciado {p.name}")
    # # Aguarda a conclusão de todos os processos
    # for p in processes:
    #     p.join()
    #     print(f"Terminado {p.name}")
    # print("Todos os processos terminaram.")

    # processes = []
    # queue = mp.Queue()  # Create a queue to manage tasks
    #
    # # Function to manage worker processes
    # def worker(queue):
    #     while True:
    #         iter = queue.get()  # Get an iteration from the queue
    #         if iter is None:  # Check for termination signal
    #             break
    #         processar_iteracao(iter, edges, weights, pasta_results, dataset, pop_norm, device, filters, K, lr, epochs,
    #                            train_ratio, task_type)
    #         queue.task_done()  # Signal completion of the task
    #
    #
    # # Spawn worker processes (up to 8)
    # for _ in range(min(8, mp.cpu_count() // 2)):  # Limit to half the CPU cores to avoid overload
    #     p = mp.Process(target=worker, args=(queue,))
    #     p.start()
    #
    # # Add iterations to the queue for processing
    # for i, iter in enumerate(iteravel):
    #     queue.put(iter)  # Add iteration to the task queue
    #
    # # Wait for all tasks to finish
    # queue.join()
    #
    # # Send termination signal to workers
    # for _ in range(min(8, mp.cpu_count() // 2)):
    #     queue.put(None)  # Add None to signal termination
    #
    # # Wait for worker processes to finish
    # for p in processes:
    #     p.join()
    #
    # print("Todos os processos terminaram.")
