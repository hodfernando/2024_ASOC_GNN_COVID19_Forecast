import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM


class GCLSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K=2, normalization="sym", out=1, num_classes=2,
                 task_type='regression'):
        super(GCLSTM, self).__init__()

        # Camada GConvLSTM
        self.recurrent = GConvLSTM(in_channels=in_channels, out_channels=out_channels, K=K, normalization=normalization,
                                   bias=True)

        # Atributos de configuração
        self.task_type = task_type
        self.out = out
        self.num_classes = num_classes

        if task_type == 'classification':
            # Camada linear para classificação multiclasse
            self.linear = torch.nn.Linear(in_features=out_channels, out_features=num_classes * out)
        else:
            # Camada linear para regressão
            self.linear = torch.nn.Linear(in_features=out_channels, out_features=out)

    def forward(self, x, edge_index, edge_weight, explainer=False):

        # Extração de características com GConvLSTM
        h, c = self.recurrent(x, edge_index, edge_weight)

        # Aplicação da ReLU e camada linear
        h = F.relu(h)
        h = self.linear(h)

        # Tratamento para classificação
        if self.task_type == 'classification':
            # Log softmax para obter probabilidades
            h = h.view(-1, self.out, self.num_classes)  # [5385, 7, num_classes]
            h = F.log_softmax(h, dim=-1)  # Aplicar log_softmax ao longo da última dimensão (classes)

            if explainer:
                h = h.squeeze(1)  # [5385, num_classes]

        return h
