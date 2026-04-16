# =====================
# 1. IMPORTS & CONFIG
# =====================
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import os
from itertools import product

import pickle

import polars as pl

#import gnn_brazil as gnnb

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# =====================
# 2. DATA PREPARATION
# =====================
def prepare_data(dataset):

    train_list, val_list, test_list = [], [], []

    # Fazer os splits primeiro, para cada cidade, para preservar a ordem temporal
    for city in dataset.columns[1:]: # not using the first column (date)
        valores = dataset[city].values
        
        # Temporal split (60-20-20) # UTILIZAREMOS APENAS OS 80% INICIAIS, POIS OS 20% FINAIS NÃO DEVEM SER TOCADOS
        split_1, split_2 = int(len(valores)*0.6), int(len(valores)*0.80)
        train, val, test = valores[:split_1], valores[split_1:split_2], valores[split_2:]

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    return train_list, val_list, test_list


def create_sequences(data_list, window_size, horizon):
    X, y = [], []

    # for each city, create the sequences and append to the whole dataset
    for data in data_list:
        for i in range(len(data)-window_size-horizon+1):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size:i+window_size+horizon])
    
    y = np.array(y).reshape(-1, horizon)
    
    return np.array(X), np.array(y)

# =====================
# 3. LSTM MODEL DEFINITION
# =====================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, output_size=7):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =====================
# 4. TRAINING FUNCTION
# =====================
def train_model(X_train, y_train, X_val, y_val, params, epochs=100, patience=10):
    print(f'{params=}')

    #print(f"Training shapes - X: {X_train.shape}, y: {y_train.shape}")
    #print(f"Validation shapes - X: {X_val.shape}, y: {y_val.shape}")

    model = LSTMModel(
        hidden_size=int(params['hidden_size']),
        num_layers=int(params['num_layers']),
        dropout=params['dropout'],
        output_size=y_train.shape[1] # this is important to ensure output has the correct dimension
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_data, batch_size=int(params['batch_size']), shuffle=True)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        #if epoch % 10 == 0:
        #    print(f'{epoch=}')

        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device))
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device))
            val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(device)).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model

# =====================
# 5. OPTUNA OPTIMIZATION
# =====================
def objective(trial):
    params = {
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        #'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        #'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    }
    
    model = train_model(X_train, y_train, X_val, y_val, params)
    
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
        return np.sqrt(mean_squared_error(y_val, y_pred))
    

'''
def retrain_with_varying_windows(best_params, scaler, train_data, val_data, test_data, 
                               window_sizes, horizons,
                               results_file="results/window_retraining_results.csv"):
    results = []
    
    for window_size in window_sizes:
        for horizon in horizons:
            print(f'{window_size=} {horizon=}')

            # Create sequences with current window/horizon
            X_train, y_train = create_sequences(train_data, window_size, horizon)
            X_val, y_val = create_sequences(val_data, window_size, horizon)
            X_test, y_test = create_sequences(test_data, window_size, horizon)
            
            # Skip if not enough data
            if len(X_train) == 0 or len(X_test) == 0:
                continue
                
            #print(f"\nTraining with window={window_size}, horizon={horizon}")
            #print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Train model with fixed hyperparameters but new window/horizon
            model = train_model(
                X_train, y_train,
                X_val, y_val,
                best_params,
                epochs=200,  # Increased epochs for better convergence
                patience=20
            )
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

                #print(f'{len(y_pred)=} {len(y_test)=}')
                
                # Inverse transform if needed
                y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, horizon)
                y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, horizon)
                
                # Calculate metrics
                rmse = root_mean_squared_error(y_test_orig, y_pred_orig)
                mae = mean_absolute_error(y_test_orig, y_pred_orig)
                mse = mean_squared_error(y_test_orig, y_pred_orig)
                mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)
                
            results.append({
                'window_size': window_size,
                'horizon': horizon,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
            
            # Save intermediate results
            pd.DataFrame(results).to_csv(results_file, index=False)
    
    return pd.DataFrame(results)'''


if __name__ == "__main__":
    # pip install optuna
    # pip install polars==1.12.0 --upgrade
    # Sem a versao correta, nem o pickle funciona pra subir o polars dataframe

    # pip install pyarrow

    with (open("pre_processed/Brazil/df_totalCases.pkl", "rb")) as openfile:
        dataset = pickle.load(openfile)

    dataset = dataset.to_pandas()
    dataset['date'] = pd.to_datetime(dataset['date'])

    # From now on, I am using the dataset to tune the LSTM through Optuna

    print(f'{dataset.head=}')
    
    # returns a list of numpy arrays. Each list component is a city.
    train_list, val_list, test_list = prepare_data(dataset)

    # Now we create the sequences considering window size and horizon
    window_size, horizon = 7, 7

    X_train, y_train = create_sequences(train_list, window_size, horizon)
    X_val, y_val = create_sequences(val_list, window_size, horizon)
    X_test, y_test = create_sequences(test_list, window_size, horizon)

    # Scaling everything based on the training data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).reshape(-1, 7, 1)
    y_train = scaler.transform(y_train)
    X_val = scaler.transform(X_val).reshape(-1, 7, 1)
    y_val = scaler.transform(y_val)
    X_test = scaler.transform(X_test).reshape(-1, 7, 1)
    y_test = scaler.transform(y_test)

    sample_10_percent = True
    if(sample_10_percent):
        max_size = int( len(X_train) * 0.05 ) # 5% of samples
        train_indexes = np.random.choice(len(X_train), max_size, replace=False)

        max_size = int( len(X_val) * 0.05 ) # 5% of samples
        val_indexes = np.random.choice(len(X_val), max_size, replace=False)

        X_train = X_train[train_indexes]
        y_train = y_train[train_indexes]
        X_val = X_val[val_indexes]
        y_val = y_val[val_indexes]

    print(f'{X_train[:5]}')
    
    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    # save results
    best_params = study.best_params
    print(f'{best_params=}')
    pd.DataFrame([best_params]).to_csv("models/best_params_lstm.csv", index=False)