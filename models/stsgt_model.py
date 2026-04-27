import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class STSGT(nn.Module):
    def __init__(self, in_channels, out_channels, out=1, num_classes=2, task_type='regression', num_layers=1, nhead=4):
        """
        Modelo STSGT para dados espaço-temporais - Versão adaptada da LSTM original

        Args:
            in_channels: Número de passos temporais (seq_len)
            out_channels: Dimensão dos embeddings espaciais/temporais
            out: Número de saídas por nó
            num_classes: Número de classes (para classificação)
            task_type: 'regression' ou 'classification'
            num_layers: Número de camadas STSGT
            nhead: Número de heads de atenção
        """
        super(STSGT, self).__init__()

        # Configurações (mantendo a interface original)
        self.hidden_size = out_channels
        self.task_type = task_type
        self.out = out
        self.num_classes = num_classes

        # Camadas STSGT
        self.stsgt_layers = nn.ModuleList([
            STSGT_Layer(
                node_features=out_channels,
                time_steps=in_channels,
                nhead=nhead
            ) for _ in range(num_layers)
        ])

        # Projeção inicial (temporal para espaço de features)
        self.init_proj = nn.Linear(1, out_channels)

        # Camada de saída (igual à LSTM original)
        if task_type == 'classification':
            self.linear = nn.Linear(out_channels, num_classes * out)
        else:
            self.linear = nn.Linear(out_channels, out)

    def forward(self, x, edge_index, edge_weight, explainer=False):
        """
        Args:
            x: Tensor [num_nodes, seq_len] (seq_len = in_channels)
            edge_index: Tensor [2, num_edges] com índices das arestas
            edge_weight: pesos das arestas
        """
        num_nodes, seq_len = x.shape

        # Preparar entrada [num_nodes, seq_len, 1] -> projetar para [num_nodes, seq_len, out_channels]
        x = x.unsqueeze(-1)  # [num_nodes, seq_len, 1]
        x = self.init_proj(x)  # [num_nodes, seq_len, out_channels]

        # Processar através das camadas STSGT
        for layer in self.stsgt_layers:
            x = layer(x, edge_index)  # [num_nodes, seq_len, out_channels]

        # Pegar a última saída temporal (como na LSTM)
        last_out = x[:, -1, :]  # [num_nodes, out_channels]

        # Saída final (igual à LSTM original)
        predictions = self.linear(last_out)

        if self.task_type == 'classification':
            predictions = predictions.view(num_nodes, self.out, self.num_classes)
            predictions = F.log_softmax(predictions, dim=-1)
            if explainer:
                predictions = predictions.squeeze(1)

        return predictions


class STSGT_Layer(nn.Module):
    """Camada individual do STSGT com atenção espaço-temporal síncrona"""

    def __init__(self, node_features, time_steps, nhead=4):
        super().__init__()
        # Atenção espacial (usando GNN Transformer)
        self.spatial_attn = TransformerConv(
            in_channels=node_features,
            out_channels=node_features // nhead,
            heads=nhead,
            concat=True
        )

        # Atenção temporal
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=node_features,
            num_heads=nhead,
            batch_first=True
        )

        # Normalização e FFN
        self.norm1 = nn.LayerNorm(node_features)
        self.norm2 = nn.LayerNorm(node_features)
        self.ffn = nn.Sequential(
            nn.Linear(node_features, node_features * 2),
            nn.ReLU(),
            nn.Linear(node_features * 2, node_features)
        )

    def forward(self, x, edge_index):
        # x: [num_nodes, seq_len, features]
        # edge_index: [2, num_edges]

        # Atenção Temporal
        x_temp, _ = self.temporal_attn(x, x, x)  # [num_nodes, seq_len, features]
        x = self.norm1(x + x_temp)

        # Atenção Espacial por timestep
        spatial_feats = []
        for t in range(x.size(1)):
            x_spatial = self.spatial_attn(x[:, t, :], edge_index)  # [num_nodes, features]
            spatial_feats.append(x_spatial)

        x_spatial = torch.stack(spatial_feats, dim=1)  # [num_nodes, seq_len, features]
        x = self.norm2(x + x_spatial)

        # FFN
        x = x + self.ffn(x)

        return x