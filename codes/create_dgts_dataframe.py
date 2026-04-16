import datetime
import os
import glob
import pandas as pd
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


class DGTSDatasetLoader(object):

    def __init__(self):
        # Caracteristicas importantes
        self.df_index = None
        self.common_dates = None
        self.std_stacked_dataset = None
        self.mean_stacked_dataset = None
        self.city_to_index = None
        self.rearranged_weights = None
        self.rearranged_edges = None
        self.dataset = None
        self.df_graph_temporal = None
        self.date_covid = None
        self.df_covid_temporal = None
        self.date_networks = None
        self.targets = None  # Classes
        self.features = None  # Atributos
        self.targets_standardized = None  # Classes normalizados
        self.features_standardized = None  # Atributos normalizados
        self.edges = None  # Conjunto dos caminhos
        self.weights = None  # Conjunto dos pesos dos caminhos
        # Saida
        self.dataset_out = None  # Dataset que sera retornado
        self.dataset_out_standardized = None  # Dataset normalizado que sera retornado
        # Entradas
        self.lags = 14  # Define o intervalo temporal
        self.dataset_in_out = 'in'  # Define se usa o dataset de entrada ou de saida
        self.out = 1  # Define o número de saidas

    def generate_dataset(self):
        """Returning the Chinese Graph + Cases Covid Data.
        Args types:
            lags:
                - Número de dias do intervalo usado para previsão. Padrão 14 dias.
        Return types:
            * **dataset** *(Normalized DynamicGraphTemporalSignal)*
            * **dataset** *(DynamicGraphTemporalSignal)*
                - Graph Chinese + Cases Covid Dataset both change in time.
        """

        print("Inicio generate_dataset")
        # Path do diretório atual
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Path do diretório principal
        project_dir = os.path.abspath(os.path.join(current_dir, ".."))

        # Define o caminho para a pasta 'raw_data' dentro do projeto
        pasta_raw_data = os.path.join(project_dir, 'raw_data', 'China')

        # Caminho para a pasta "dataverse_files"
        networks_dir = os.path.join(pasta_raw_data, "dataverse_files")

        # Lista apenas arquivos .csv no diretório
        networks_files = [file for file in glob.glob(os.path.join(networks_dir, "*.csv")) if os.path.isfile(file)]

        # Filtrar os arquivos que não contêm '-'
        networks_files = list(filter(lambda file: '-' not in os.path.basename(file), networks_files))

        # Lista de arquivos de redes de entrada "baidu_in" e de saída "baidu_out"
        networks_files_in = [file for file in networks_files if ('baidu_in' in file)]
        networks_files_out = [file for file in networks_files if ('baidu_out' in file)]

        date_networks_in = self.format_dates(networks_files_in)
        date_networks_out = self.format_dates(networks_files_out)

        if self.dataset_in_out == 'in':
            self.date_networks = date_networks_in
            df_networks = pd.DataFrame(
                {'date_networks': date_networks_in, 'networks_files_in': networks_files_in})
        else:
            self.date_networks = date_networks_out
            df_networks = pd.DataFrame(
                {'date_networks': date_networks_out, 'networks_files_out': networks_files_out})

        # Caminho para a pasta "Covid-19 daily cases in China"
        covid_dir = os.path.join(pasta_raw_data, "Covid-19 daily cases in China")

        # Listar arquivos de redes no diretório "Covid-19 daily cases in China"
        # covid_files = os.listdir(covid_dir)

        self.df_covid_temporal = pd.read_excel(covid_dir + "/covid-19 daily confirmed cases.xlsx", header=0)

        # Remover coluna pelo índice
        self.df_covid_temporal = self.df_covid_temporal.drop(self.df_covid_temporal.columns[129], axis=1)

        # Converta as colunas de datas para o formato 'YYYY-MM-DD'
        self.date_covid = [date.strftime("%Y-%m-%d") for date in self.df_covid_temporal.columns[2:]]
        self.df_covid_temporal.columns = self.df_covid_temporal.columns[:2].tolist() + self.date_covid

        # Datas comuns entre os arquivos de redes e os dados de covid
        self.common_dates = sorted(set(self.date_networks).intersection(self.date_covid))

        # Criar cópia do DataFrame com base nas datas comuns
        df_common_dates = df_networks.loc[df_networks['date_networks'].isin(self.common_dates)].copy().reset_index(
            drop=True)

        for i in range(1, len(self.common_dates)):
            date_prev = pd.to_datetime(self.common_dates[i - 1])
            date_curr = pd.to_datetime(self.common_dates[i])

            # Verifica a diferença em dias entre datas consecutivas
            days_diff = (date_curr - date_prev).days

            # Define um limite para o salto temporal (por exemplo, 1 dia)
            if days_diff > 1:
                print(f"Salto temporal detectado no índice {i}: {self.common_dates[i - 1]} -> {self.common_dates[i]}")

                # Cria um DataFrame auxiliar copiando a linha anterior
                aux_df = df_common_dates[df_common_dates['date_networks'] == self.common_dates[i - 1]].copy()

                # Adiciona as datas ausentes ao DataFrame auxiliar
                missing_dates = pd.date_range(date_prev + pd.DateOffset(days=1), date_curr - pd.DateOffset(days=1),
                                              freq='D')
                for new_date in missing_dates:
                    new_date_str = new_date.strftime('%Y-%m-%d')
                    aux_df.loc[:, 'date_networks'] = new_date_str
                    df_common_dates = pd.concat([df_common_dates, aux_df], ignore_index=True)

        # Reordena o DataFrame por data
        df_common_dates.sort_values(by=['date_networks'], inplace=True, ignore_index=True)

        # Caminho para a pasta "networks"
        info_dir = os.path.join(networks_dir, "info")
        self.df_index = pd.read_csv(info_dir + "/Index_City_CH_EN.csv")
        self.df_index["GbCity"] = self.df_index["GbCity"].astype(str)

        # Retirando as cidades que não serão usadas
        file = df_common_dates.networks_files_in.iloc[0] if self.dataset_in_out == 'in' \
            else df_common_dates.networks_files_out.iloc[0]
        df_current = pd.read_csv(file)
        intersection = set(df_current.columns) & set(self.df_index['City_CH'])

        # Filtrar df_index mantendo apenas as linhas que têm valores em intersection
        self.df_index = self.df_index[self.df_index['City_CH'].isin(intersection)].reset_index(drop=True)
        # Retirar as linhas duplicadas
        self.df_index = self.df_index[~self.df_index.duplicated(subset='City_EN', keep=False)].reset_index(drop=True)

        # Crie um filtro para selecionar apenas as cidades presentes no gráfico
        city_filter = self.df_covid_temporal['City_name'].isin(self.df_index['City_EN'])

        # Aplique o filtro no DataFrame df_covid_temporal
        self.df_graph_temporal = self.df_covid_temporal[city_filter]

        # Reindexe o DataFrame df_graph_temporal
        self.df_graph_temporal = self.df_graph_temporal.reset_index(drop=True)

        # Crie um filtro para selecionar apenas as cidades presentes no df_graph_temporal
        index_filter = self.df_index['City_EN'].isin(self.df_graph_temporal['City_name'])

        # Aplique o filtro no DataFrame df_index
        self.df_index = self.df_index[index_filter].reset_index(drop=True)

        # Seleciona apenas as colunas (Datas) que são comuns a ambos os DataFrames
        common_columns = ['City_name', 'City_code'] + df_common_dates['date_networks'].tolist()
        self.df_graph_temporal = self.df_graph_temporal[common_columns]

        # Transforme as colunas de casos diários em casos acumulativos
        self.df_graph_temporal[self.df_graph_temporal.columns[2:]] = self.df_graph_temporal[
            self.df_graph_temporal.columns[2:]].cumsum(axis=1)

        # Remova as duas primeiras colunas do DataFrame df_graph_temporal (City_name e City_code)
        self.dataset = self.df_graph_temporal.iloc[:, 2:].T.to_numpy()

        # Inicialize listas vazias para as arestas e pesos
        self.edges = []
        self.weights = []

        # Criar um dicionário que mapeia cidades para índices
        self.city_to_index = {city: index for index, city in enumerate(self.df_graph_temporal['City_name'])}

        for file in df_common_dates[f'networks_files_{self.dataset_in_out}']:
            edges = []
            weights = []
            df_current = pd.read_csv(file)
            # Transforma os valores NaN em 0.0
            df_current.fillna(0.0, inplace=True)
            if "city_name" in df_current.columns:
                edges, weights = self.edges_and_weight(df_current, "city_name", "City_CH")
            elif "City_EN" in df_current.columns:
                edges, weights = self.edges_and_weight(df_current, "City_EN", "City_EN")
            elif "GbCity_EN" in df_current.columns:
                # Criar uma cópia do DataFrame
                df_processed = df_current.copy()
                # Extrai os últimos 4 dígitos de cada coluna, exceto 'GbCity_EN'
                df_processed.columns = ['GbCity_EN'] + df_processed.columns[1:].str.extract(r'(\d{4})',
                                                                                            expand=False).tolist()
                # Extrai os últimos 4 dígitos da coluna 'GbCity_EN'
                df_processed['GbCity_EN'] = df_processed['GbCity_EN'].str.extract(r'(\d{4})', expand=False)
                edges, weights = self.edges_and_weight(df_processed, "GbCity_EN", "GbCity")
            else:
                print(f"Erro no arquivo: {file}")

            if edges:
                self.edges.append(np.array(edges, dtype=np.int64).T)
                self.weights.append(np.array(weights, dtype=np.float64))
            else:
                print(f"Lista vazia para o arquivo: {file}")

        # Número de vezes que o padrão deve ser repetido
        num_repeticoes = self.dataset.__len__() // (self.lags + self.out)

        # Calculando os índices de início e fim para cada repetição
        inicio_indices = np.arange(num_repeticoes) * (self.lags + self.out)
        fim_indices = inicio_indices + self.lags

        # # Concatenando as fatias correspondentes de edges usando indexação direta
        # self.rearranged_edges = [self.edges[inicio:fim] for inicio, fim in zip(inicio_indices, fim_indices)]
        #
        # # Concatenando as fatias correspondentes de weights usando indexação direta
        # self.rearranged_weights = [self.weights[inicio:fim] for inicio, fim in zip(inicio_indices, fim_indices)]

        # Concatenando as fatias correspondentes de edges usando indexação direta
        self.rearranged_edges = [self.edges[inicio] for inicio in inicio_indices]

        # Concatenando as fatias correspondentes de weights usando indexação direta
        self.rearranged_weights = [self.weights[inicio] for inicio in inicio_indices]

        # Transformando os dados
        # stacked_dataset = np.stack(self.dataset)

        # # Número de vezes que o padrão deve ser repetido
        # num_repeticoes = self.dataset.__len__() // (self.lags + self.out)
        #
        # # Calculando os índices de início e fim para cada repetição
        # inicio_indices = np.arange(num_repeticoes) * (self.lags + self.out)
        # fim_indices = inicio_indices + self.lags

        # Separando Features e Targets
        self.features = [self.dataset[inicio:fim].T for inicio, fim in zip(inicio_indices, fim_indices)]

        # Separando targets
        self.targets = [self.dataset[fim:fim + self.out].T for fim in fim_indices]

        # Salva o dataset e retorna
        self.dataset_out = DynamicGraphTemporalSignal(self.rearranged_edges, self.rearranged_weights, self.features,
                                                      self.targets)

        # Normalizando (Z) os dados
        self.mean_stacked_dataset = np.mean(self.dataset, axis=0)
        self.std_stacked_dataset = np.std(self.dataset, axis=0)
        numerador = (self.dataset - self.mean_stacked_dataset)
        denominador = self.std_stacked_dataset
        standardized_dataset = np.zeros(numerador.shape)
        np.divide(numerador, denominador, out=standardized_dataset, where=denominador != 0, )

        # Separando Features e Targets standardizados
        self.features_standardized = [standardized_dataset[inicio:fim].T for inicio, fim in
                                      zip(inicio_indices, fim_indices)]

        # Separando targets
        self.targets_standardized = [standardized_dataset[fim:fim + self.out].T for fim in fim_indices]

        # Salva o dataset e retorna
        self.dataset_out_standardized = DynamicGraphTemporalSignal(self.rearranged_edges, self.rearranged_weights,
                                                                   self.features_standardized,
                                                                   self.targets_standardized)
        print("Fim generate_dataset")

    def format_dates(self, networks_files):
        formatted_dates = []
        for file in networks_files:
            date = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
            try:
                date_obj = datetime.datetime.strptime(date, "%Y%m%d")
                formatted_dates.append(date_obj.strftime("%Y-%m-%d"))
            except ValueError:
                print(f"Erro na data: {date}")
        return formatted_dates

    def edges_and_weight(self, df_current, df_current_col, df_index_col):
        edges = []
        weights = []

        # Filtra as linhas em df_current onde df_current['city_name'] está em df_index['City_CH']
        df_current_filtered = df_current[df_current[df_current_col].isin(self.df_index[df_index_col])]

        # Filtra as linhas em df_index
        df_index_filtered = self.df_index[self.df_index[df_index_col].isin(df_current_filtered[df_current_col])]

        # Reordena df_current_filtered para corresponder à sequência de df_index['City_CH']
        df_current_reordered = df_current_filtered.set_index(df_current_col).reindex(
            df_index_filtered[df_index_col]).reset_index()

        # Percorre as colunas
        for col_name, col_index in zip(df_index_filtered[df_index_col].tolist(), df_index_filtered.index):
            # Obtém a coluna atual usando o índice de df_current
            col = df_current_reordered[col_name]

            # Para cada valor não nulo, adiciona as informações à arrays
            for dest_index in col.index:
                weight = col[dest_index]
                if weight != 0.0:
                    edges.append([col_index, dest_index])
                    weights.append(weight)
        return edges, weights

    def get_dataset(self, lags=14, dataset_in_out='in', out=1) -> [DynamicGraphTemporalSignal,
                                                                   DynamicGraphTemporalSignal]:
        self.lags = lags
        self.dataset_in_out = dataset_in_out
        self.out = out
        print("Inicio get_dataset")
        if self.dataset_out is None or self.dataset_out_standardized is None:
            self.generate_dataset()
        print("Fim get_dataset")
        return self.dataset_out, self.dataset_out_standardized
