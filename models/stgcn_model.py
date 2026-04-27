import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.attention import STConv


class STGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, K=2, normalization="sym", out=1, num_classes=2,
                 task_type='regression'):
        super(STGCN, self).__init__()

        # Configurações
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.task_type = task_type
        self.out = out
        self.num_classes = num_classes

        # Camada STConv
        self.recurrent = STConv(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            K=K,
            normalization=normalization,
            bias=True
        )

        # Camada linear
        if task_type == 'classification':
            self.linear = torch.nn.Linear(out_channels, num_classes * out)
        else:
            self.linear = torch.nn.Linear(out_channels, out)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Tensor de forma [num_nodes, in_channels]
            edge_index: Índices das arestas [2, num_edges]
            edge_weight: Pesos das arestas [num_edges] (opcional)
        """
        # Adicionar dimensões de batch e timesteps
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, in_channels]

        # Passar pela STConv
        h = self.recurrent(x, edge_index, edge_weight)  # [1, 1, num_nodes, out_channels]

        # Remover dimensões extras
        h = h.squeeze(0).squeeze(0)  # [num_nodes, out_channels]

        # Camada linear
        h = F.relu(h)
        h = self.linear(h)

        # Tratamento para classificação
        if self.task_type == 'classification':
            h = h.view(-1, self.out, self.num_classes)
            h = F.log_softmax(h, dim=-1)

        return h