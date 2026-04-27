import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_outputs=1, num_classes=2,
                 task_type='regression', num_layers=1, dropout=0.2):
        """
        Modelo LSTM para dados temporais por cidade.

        Args:
            input_size (int): Número de features por passo de tempo (1 para univariado).
            hidden_size (int): Tamanho da camada oculta da LSTM.
            num_outputs (int): Número de saídas por cidade (geralmente 1).
            num_classes (int): Número de classes (usado apenas em classificação).
            task_type (str): 'regression' ou 'classification'.
            num_layers (int): Número de camadas LSTM empilhadas.
            dropout (float): Probabilidade de dropout entre as camadas LSTM.
        """
        super().__init__()
        self.task_type = task_type
        self.num_outputs = num_outputs
        self.num_classes = num_classes

        # Camada LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Camada de saída totalmente conectada
        if self.task_type == 'classification':
            self.linear = nn.Linear(hidden_size, num_classes * num_outputs)
        else:
            self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, x, edge_index=None, edge_weight=None, explainer=False):
        """
        Args:
            x (Tensor): Tensor de entrada com forma [num_cidades, seq_len].
        """
        # Adiciona a dimensão de feature: [num_cidades, seq_len, 1]
        x = x.unsqueeze(-1)

        # A LSTM inicializa os estados ocultos com zeros por padrão se não forem fornecidos.
        # out: [num_cidades, seq_len, hidden_size]
        out, _ = self.lstm(x)

        # Pega a saída do último passo de tempo: [num_cidades, hidden_size]
        last_out = out[:, -1, :]

        # Passa pela camada linear para obter a predição final
        predictions = self.linear(last_out)

        # Pós-processamento para tarefas de classificação
        if self.task_type == 'classification':
            predictions = predictions.view(-1, self.num_outputs, self.num_classes)
            predictions = F.log_softmax(predictions, dim=-1)
            if explainer:
                predictions = predictions.squeeze(1)

        return predictions
