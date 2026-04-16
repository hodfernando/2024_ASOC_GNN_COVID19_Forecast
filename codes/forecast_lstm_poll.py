import bz2
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch_geometric_temporal import temporal_signal_split
from codes.create_dgts_dataframe import DGTSDatasetLoader
from multiprocessing import Pool

warnings.filterwarnings("ignore")


def sequence_lag(input_data, tw):
    return [input_data[tw * (i + 1)] for i in range(((len(input_data) - 1) // tw))]


def create_inout_sequences(input_data, tw):
    return [(input_data[tw * i: tw * (i + 1)], input_data[tw * (i + 1)]) for i in range(((len(input_data) - 1) // tw))]


class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def train_lstm(city_data):
    city, train_data, test_data, lags = city_data

    train_inout_seq = create_inout_sequences(torch.FloatTensor(train_data).view(-1), lags)
    test_inout_seq = create_inout_sequences(torch.FloatTensor(test_data).view(-1), lags)

    print('Cidade', city)
    print('LSTM')

    model = LSTM()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 100

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'Cidade - {city}\nepoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'Cidade - {city}\nepoch: {i:3} loss: {single_loss.item():10.8f}')

    model.eval()

    test_outs = []
    test_targets = []
    for seq, labels in test_inout_seq:
        with torch.no_grad():
            model.hidden = (
                torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, model.hidden_layer_size))
            test_outs.append(model(seq).item())
            test_targets.append(np.float32(labels))

    pred_lstm = np.array(test_outs)

    lstm_rmse_error_city = mean_squared_error(test_targets, pred_lstm, squared=True)
    lstm_r2_city = r2_score(test_targets, pred_lstm)

    return city, pred_lstm, lstm_rmse_error_city, lstm_r2_city


if __name__ == "__main__":
    lags = 14
    nameModel = 'LSTM'
    train_ratio = 0.8

    # Path do diretório atual
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path do diretório principal
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # Caminho do diretório 'results'
    results_dir = os.path.join(os.path.join(project_dir, "results"), nameModel)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Caminho para a pasta "pre_processed_data"
    pre_processed_data_dir = os.path.join(os.path.join(project_dir, "raw_data"), "pre_processed_data")
    Path(project_dir + '/raw_data/pre_processed_data/').mkdir(parents=True, exist_ok=True)

    # Carregando Dataset
    file_path = os.path.join(pre_processed_data_dir, f'DGTS_Dataset_lags_{lags}')
    if os.path.exists(file_path):
        file = bz2.BZ2File(file_path, 'rb')
        datasetLoader = pickle.load(file)
        dataset, dataset_standardized = datasetLoader.get_dataset(lags=lags, dataset_in_out='in')
        file.close()
        print('Dataset carregado pelo pickle data')
    else:
        datasetLoader = DGTSDatasetLoader()
        dataset, dataset_standardized = datasetLoader.get_dataset(lags=lags, dataset_in_out='in')
        file = bz2.BZ2File(file_path, 'wb')
        pickle.dump(datasetLoader, file)
        file.close()
        print('Dataset carregado e gerado arquivo pickle')

    print("Separando dataset em teste e treinamento")

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)

    # Calcular o índice que divide as datas em 80% para treinamento e 20% para teste
    split_index = train_dataset.snapshot_count * lags

    # Dividir as datas em conjuntos de treinamento e teste
    days_train = datasetLoader.date_covid[:split_index + 1]
    days_test = datasetLoader.date_covid[split_index: split_index + 1 + test_dataset.snapshot_count * 14]

    # Armazenando dados temporais de covid em treinamento e teste
    train_data_list = []
    for time, snapshot in enumerate(train_dataset):
        # Adiciona os dados da feature (x) à lista
        train_data_list.extend(snapshot.x.T.cpu().detach().numpy())
    # Adiciona os dados do rótulo (y) à lista como uma única lista
    train_data_list.append(snapshot.y.T.cpu().detach().numpy().tolist())
    train_data = np.array(train_data_list)

    test_data_list = []
    for time, snapshot in enumerate(test_dataset):
        # Adiciona os dados da feature (x) à lista
        test_data_list.extend(snapshot.x.T.cpu().detach().numpy())
    # Adiciona os dados do rótulo (y) à lista como uma única lista
    test_data_list.append(snapshot.y.T.cpu().detach().numpy().tolist())
    test_data = np.array(test_data_list)

    # Criar uma lista de argumentos para cada cidade
    city_data_list = [(city, train_data[:, city], test_data[:, city], lags) for city in range(train_data.shape[1])]

    # Usar Pool para paralelizar o treinamento em várias cidades
    with Pool() as pool:
        results = pool.map(train_lstm, city_data_list)

    # LSTM
    pred_lstm = np.zeros((test_data.shape[0] // 14, test_data.shape[1]))
    lstm_rmse_error_city = np.zeros((test_data.shape[1]))
    lstm_r2_city = np.zeros((test_data.shape[1]))

    # Descompactar os resultados
    for city, pred, rmse_error_city, r2_city in results:
        pred_lstm[:, city] = pred
        lstm_rmse_error_city[city] = rmse_error_city
        lstm_r2_city[city] = r2_city

    cities_names = datasetLoader.df_graph_temporal.City_name.to_list()
    metrics = pd.DataFrame(
        {"City_name": cities_names, "LSTM RMSE": lstm_rmse_error_city, "LSTM R2": lstm_r2_city})

    city = 0

    plt.figure(figsize=(16, 9))

    # Convertendo as datas para o formato matplotlib
    dates_test = sequence_lag(days_test, lags)
    dates_test = [pd.to_datetime(date) for date in dates_test]

    # Configurando o formato da data no eixo x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.plot_date(x=dates_test, y=sequence_lag(test_data[:, city], lags), linestyle="-", label="test_data")
    plt.plot_date(x=dates_test, y=pred_lstm[:, city], linestyle=":", label="pred_lstm")
    plt.legend(title=cities_names[city])
    plt.show()

    metrics.to_csv(path_or_buf=results_dir + f'/R2_RMSE.csv', sep=';', index=False)
    np.save(results_dir + f'/pred_lstm.npy', pred_lstm)
