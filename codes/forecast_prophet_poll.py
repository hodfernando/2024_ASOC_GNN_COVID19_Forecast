import bz2
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error
from prophet import Prophet
import warnings
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch_geometric_temporal import temporal_signal_split
from codes.create_dgts_dataframe import DGTSDatasetLoader
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")


def run_prophet(city, train_data, test_data):
    print('Cidade', city)
    print('Prophet')

    m = Prophet()
    m.fit(train_data)

    future = m.make_future_dataframe(periods=len(test_data))
    prophet_pred = m.predict(future)

    return prophet_pred


def sequence_lag(input_data, tw):
    return [input_data[tw * (i + 1)] for i in range(((len(input_data) - 1) // tw))]


if __name__ == '__main__':
    lags = 14
    nameModel = 'Prophet'
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
    split_index = train_dataset.snapshot_count * lags + 1

    # Dividir as datas em conjuntos de treinamento e teste
    days_train = datasetLoader.date_covid[:split_index]
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

    # PROPHET
    pred_prophet = np.zeros((test_data.shape[0], test_data.shape[1]))
    prophet_rmse_error_city = np.zeros((test_data.shape[1]))
    prophet_r2_city = np.zeros((test_data.shape[1]))

    # Use ProcessPoolExecutor para paralelizar o loop
    with ProcessPoolExecutor() as executor:
        futures = []
        for city in range(train_data.shape[1]):
            train_data_pr = pd.DataFrame({'ds': pd.to_datetime(days_train), 'y': train_data[:, city]})
            test_data_pr = pd.DataFrame({'ds': pd.to_datetime(days_test), 'y': test_data[:, city]})

            # Submeter a tarefa para execução em paralelo
            future = executor.submit(run_prophet, city, train_data_pr, test_data_pr)
            futures.append(future)

            pred_prophet[:, city] = future.result()['yhat'][-test_data[:, city].shape[0]:].values

            # plt.figure(figsize=(10, 8))
            # ax = sns.lineplot(x=[x for x in range(test_data_no_norm[:, city].shape[0])], y=test_data_no_norm[:, city])
            # sns.lineplot(x=[x for x in range(test_data_no_norm[:, city].shape[0])],
            #              y=prophet_pred['yhat'][-test_data_no_norm[:, city].shape[0]:].values)
            # plt.show()

            prophet_rmse_error_city[city] = mean_squared_error(test_data[:, city], pred_prophet[:, city], squared=True)

            prophet_r2_city[city] = r2_score(test_data[:, city],
                                             future.result()['yhat'][-test_data[:, city].shape[0]:].values)

    #################

    cities_names = datasetLoader.df_graph_temporal.City_name.to_list()
    metrics = pd.DataFrame(
        {"City_name": cities_names, "Prophet RMSE": prophet_rmse_error_city, "Prophet R2": prophet_r2_city})

    city = 0
    plt.figure(figsize=(16, 9))

    # Convertendo as datas para o formato matplotlib
    dates_test = sequence_lag(days_test, lags)
    dates_test = [pd.to_datetime(date) for date in dates_test]

    # Configurando o formato da data no eixo x
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.plot_date(x=dates_test, y=sequence_lag(test_data[:, city], lags), linestyle="-", label="test_data")
    plt.plot_date(x=dates_test, y=sequence_lag(pred_prophet[:, city], lags), linestyle=":", label="pred_prophet")
    plt.legend(title=cities_names[city])
    plt.show()

    metrics.to_csv(path_or_buf=results_dir + f'/R2_RMSE.csv', sep=';', index=False)
    np.save(results_dir + f'/pred_prophet.npy', pred_prophet)
