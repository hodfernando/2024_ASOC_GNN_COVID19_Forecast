# coding: utf-8
# Importing libraries
import os
import pickle
import random
import math
import pandas as pd
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
import requests
import networkx as nx
from itertools import product
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
            backbone.add_nodes_from([source], ibge=graph.nodes[source]['ibge'])
            backbone.add_nodes_from([target], ibge=graph.nodes[target]['ibge'])

            codes.add(graph.nodes[source]['ibge'])
            codes.add(graph.nodes[target]['ibge'])

    return backbone, np.array(sorted(codes))


def _aplicar_extracao_backbone(G, alpha):
    """
    Aplica o algoritmo de extração de backbone em um grafo.
    """
    print(f"Aplicando extração de backbone com alpha = {alpha}...")
    if G.number_of_nodes() == 0:
        print("Grafo vazio, não é possível aplicar backbone.")
        return G, np.array([]), np.array([]), [], {}

    g_strength = G.degree(weight='weight')
    g_degree = G.degree()
    ignored_nodes = {node: 0 for node in list(G.nodes)}

    # Chama a função externa para extrair o backbone
    backbone_g, codigos_municipios_g = extract_backbone(G, alpha, g_strength, g_degree, ignored_nodes)

    # Recalcula todos os artefatos a partir do novo grafo de backbone
    codigo_para_indice_g = {codigo: i for i, codigo in enumerate(codigos_municipios_g)}

    mapa_nos_antigo_para_novo = {
        n: codigo_para_indice_g[data['ibge']]
        for n, data in backbone_g.nodes(data=True) if data['ibge'] in codigo_para_indice_g
    }
    backbone_g_remapeado = nx.relabel_nodes(backbone_g, mapa_nos_antigo_para_novo, copy=True)

    num_edges = backbone_g_remapeado.number_of_edges()
    edges_g = np.zeros((2, num_edges), dtype=np.int32)
    weights_g = np.zeros(num_edges, dtype=np.float32)

    for i, (u, v, data) in enumerate(backbone_g_remapeado.edges(data=True)):
        edges_g[:, i] = [u, v]
        weights_g[i] = data['weight']

    return backbone_g_remapeado, edges_g, weights_g, codigos_municipios_g, codigo_para_indice_g


def construindo_rede_mobilidade(
        pasta_raw_data,
        pasta_pre_processed,
        tipo_grafo="grafo_original",
        backbone=False,
        alpha=0.4
):
    """
    Constrói ou carrega uma rede de mobilidade, com a opção de aplicar extração de backbone.
    """
    print(f"Iniciando construção. Tipo: '{tipo_grafo}', Aplicar Backbone: {backbone}")

    # --- 1. Definição de Nomes de Arquivos ---
    sufixo_arquivo = tipo_grafo
    if backbone:
        sufixo_arquivo += f"_backbone_{alpha}"

    caminho_edges = os.path.join(pasta_pre_processed, f'edges_{sufixo_arquivo}.npy')
    caminho_weights = os.path.join(pasta_pre_processed, f'weights_{sufixo_arquivo}.npy')
    caminho_codigos = os.path.join(pasta_pre_processed, f'codigos_municipios_{sufixo_arquivo}.pkl')
    caminho_mapeamento = os.path.join(pasta_pre_processed, f'codigo_para_indice_{sufixo_arquivo}.pkl')
    caminho_grafo = os.path.join(pasta_pre_processed, f'grafo_{sufixo_arquivo}.graphml')

    # --- 2. Carregamento de Dados (se já existirem) ---
    if all(os.path.exists(p) for p in
           [caminho_edges, caminho_weights, caminho_grafo, caminho_codigos, caminho_mapeamento]):
        print("Carregando dados pré-processados...")
        edges = np.load(caminho_edges)
        weights = np.load(caminho_weights)
        with open(caminho_codigos, 'rb') as f:
            codigos_municipios = pickle.load(f)
        with open(caminho_mapeamento, 'rb') as f:
            codigo_para_indice = pickle.load(f)
        G = nx.read_graphml(caminho_grafo)
        G = nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})
        return edges, weights, codigos_municipios, codigo_para_indice, G

    # --- 3. Construção do Grafo Base ---
    print("Construindo grafo base a partir dos dados brutos...")
    caminho_df_ibge = os.path.join(pasta_raw_data, 'dataset_transform_IBGE.xlsx')
    df_ibge = pl.read_excel(source=caminho_df_ibge, engine='xlsx2csv', read_options={"has_header": True})

    codigos_municipios_orig = np.unique(df_ibge[['CODMUNDV_A', 'CODMUNDV_B']].to_numpy().flatten())
    codigo_para_indice_orig = {codigo: i for i, codigo in enumerate(codigos_municipios_orig)}

    G_base = nx.Graph()
    for row in df_ibge.iter_rows(named=True):
        src_code, dst_code = row['CODMUNDV_A'], row['CODMUNDV_B']
        src_idx, dst_idx = codigo_para_indice_orig[src_code], codigo_para_indice_orig[dst_code]
        weight = row['VAR05'] + row['VAR06'] + row['VAR12']

        G_base.add_edge(src_idx, dst_idx, weight=weight)
        G_base.add_nodes_from([src_idx], ibge=src_code)
        G_base.add_nodes_from([dst_idx], ibge=dst_code)

    # --- 4. Geração do Grafo de Trabalho (conforme tipo_grafo) ---
    G_trabalho = G_base if tipo_grafo == "grafo_original" else None
    # _gerar_grafo_pesos_iguais(G_base) if tipo_grafo == "grafo_pesos_iguais" else \
    #     _gerar_grafo_pesos_redistribuidos(G_base) if tipo_grafo == "grafo_pesos_redistribuidos" else \
    #         _gerar_grafo_aleatorio(G_base) if tipo_grafo == "grafo_aleatorio" else \
    #             _gerar_grafo_erdos_renyi(G_base) if tipo_grafo == "grafo_erdos_renyi" else \
    #                 None
    if G_trabalho is None:
        raise ValueError(f"Tipo de grafo desconhecido: '{tipo_grafo}'")

    # --- 5. Aplicação Opcional do Backbone e Preparação para Salvamento ---
    if backbone:
        G_final, edges_final, weights_final, codigos_final, mapeamento_final = \
            _aplicar_extracao_backbone(G_trabalho, alpha)
    else:
        G_final = G_trabalho
        codigos_final = codigos_municipios_orig
        mapeamento_final = codigo_para_indice_orig

        num_edges = G_final.number_of_edges()
        edges_final = np.zeros((2, num_edges), dtype=np.int32)
        weights_final = np.zeros(num_edges, dtype=np.float32)

        for i, (u, v, data) in enumerate(G_final.edges(data=True)):
            edges_final[:, i] = [u, v]
            weights_final[i] = data.get('weight', 0.0)

    # --- 6. Salvamento dos Artefatos Finais ---
    print(f"Salvando resultados em arquivos com sufixo: '{sufixo_arquivo}'")
    os.makedirs(pasta_pre_processed, exist_ok=True)

    np.save(caminho_edges, edges_final)
    np.save(caminho_weights, weights_final)
    with open(caminho_codigos, 'wb') as f:
        pickle.dump(codigos_final, f)
    with open(caminho_mapeamento, 'wb') as f:
        pickle.dump(mapeamento_final, f)
    nx.write_graphml(G_final, caminho_grafo)

    print(f"Rede '{sufixo_arquivo}' construída e salva com sucesso.")

    return edges_final, weights_final, codigos_final, mapeamento_final, G_final


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


def dataframes_covid(pasta_pre_processed, df_cidades, datas, codigos_municipios, sufixo_arquivo):
    print("Construindo DataFrames para as variáveis de COVID-19...")

    # Caminhos para os arquivos a serem salvos usando o sufixo
    caminho_newDeaths = os.path.join(pasta_pre_processed, f'df_newDeaths_{sufixo_arquivo}.pkl')
    caminho_deaths = os.path.join(pasta_pre_processed, f'df_deaths_{sufixo_arquivo}.pkl')
    caminho_newCases = os.path.join(pasta_pre_processed, f'df_newCases_{sufixo_arquivo}.pkl')
    caminho_totalCases = os.path.join(pasta_pre_processed, f'df_totalCases_{sufixo_arquivo}.pkl')

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
            if df_cidade.height == 0:
                continue
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


def populacao_normalizada(pasta_pre_processed, pasta_raw_data, codigos_municipios, sufixo_arquivo):
    """
    Calcula a população normalizada para cada município.
    """
    # Caminho para o arquivo a ser salvo usando o sufixo
    caminho_pop_norm = os.path.join(pasta_pre_processed, f'pop_normalizada_{sufixo_arquivo}.pkl')

    # O resto da função permanece exatamente o mesmo...
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


def calcular_targets(dataset, pop_norm, tipo_dado, thresholds, janela=7):
    """
    Calcula os targets de classificação com base no tipo de dado, thresholds dinâmicos e dados de população.

    Args:
        dataset (numpy.ndarray): Conjunto de dados espaço-temporal (acumulado ou variação).
        pop_norm (list ou numpy.ndarray): Array da população para normalização.
        tipo_dado (str): Tipo de dado a ser processado. Deve ser 'acumulado' ou 'variacao'.
        thresholds (list ou dict): Uma lista ou dicionário com os limites para classificação.
        janela (int): Tamanho da janela para a média móvel (padrão 7).

    Returns:
        tuple: Uma tupla contendo:
               - numpy.ndarray: Array com os targets de classificação.
               - pandas.DataFrame: DataFrame com a distribuição (contagem) de cada classe.
    """
    # --- 1. Preparação dos Inputs ---
    if not isinstance(thresholds, (list, dict)):
        raise TypeError("O parâmetro 'thresholds' deve ser uma lista ou um dicionário.")

    # Se for um dicionário, extrai os valores e os ordena
    if isinstance(thresholds, dict):
        thresholds_list = sorted(list(thresholds.values()))
    else:
        thresholds_list = sorted(thresholds)

    if tipo_dado not in ['acumulado', 'variacao']:
        raise ValueError("O parâmetro 'tipo_dado' deve ser 'acumulado' ou 'variacao'.")

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
    plt.title(f'Distribuição das Classes (Janela={janela}, Tipo={tipo_dado.capitalize()})', fontsize=14)
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
        tipo_dado='variacao',
        thresholds=[1], # [1, 10, 25],
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
    country = 'Brazil'

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

    # Obtém o caminho completo para o arquivo 'casos_covid_brasil.csv.gz'
    caminho_arquivo_covid_brasil = obter_dados_covid_brasil(pasta_raw_data)

    # DataFrame para os dados completos de COVID-19 no Brasil
    dataframe_covid_brasil = pl.read_csv(caminho_arquivo_covid_brasil)

    # Extraindo as colunas necessárias do DataFrame original
    colunas_necessarias = ['date', 'city', 'ibgeID', 'newDeaths', 'deaths', 'newCases', 'totalCases']
    df_covid_base = dataframe_covid_brasil[colunas_necessarias].clone()

    # Dias reportados
    datas = dataframe_covid_brasil['date'].unique().sort()

    # Lista com todos os tipos de grafo que a função pode gerar
    tipos_de_grafo = [
        "grafo_original",
        # "grafo_aleatorio",
        # "grafo_pesos_iguais",
        # "grafo_pesos_redistribuidos",
        # "grafo_erdos_renyi"
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

        # --- Chama a função para construir a rede específica ---
        # A função já salva os arquivos .npy, .pkl e .graphml na pasta 'pasta_pre_processed'
        edges, weights, codigos_municipios, codigo_para_indice, G = construindo_rede_mobilidade(
            pasta_raw_data=pasta_raw_data,
            pasta_pre_processed=pasta_pre_processed,
            tipo_grafo=tipo,
            backbone=backbone,
            alpha=alpha
        )
        # Filtrando as linhas onde 'ibgeID' está presente em codigos_municipios do IBGE
        df_cidades_filtrado = df_covid_base.filter(df_covid_base['ibgeID'].is_in(codigos_municipios))

        # Cria dataframes para cada variável
        df_newDeaths, df_deaths, df_newCases, df_totalCases = dataframes_covid(
            pasta_pre_processed,
            df_cidades_filtrado,
            datas,
            codigos_municipios,
            sufixo_arquivo=sufixo_arquivo
        )

        # População normalizada
        pop_norm = populacao_normalizada(
            pasta_pre_processed,
            pasta_raw_data,
            codigos_municipios,
            sufixo_arquivo=sufixo_arquivo
        )

        # Remova a primeira coluna do DataFrame
        dataset = df_newCases[:, 1:].to_numpy()

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
