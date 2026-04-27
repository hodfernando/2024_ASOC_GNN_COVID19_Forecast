import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimesFMModel(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=64, out=1, num_classes=2,
                 task_type='regression', num_layers=4, nhead=4):
        """
        Modelo TimesFM corrigido para entrada [num_cidades, in_channels]

        Args:
            in_channels: Número de passos temporais (seq_len)
            out_channels: Não utilizado (mantido para compatibilidade)
            d_model: Dimensão do modelo
            out: Número de saídas por cidade
            num_classes: Número de classes (para classificação)
            task_type: 'regression' ou 'classification'
            num_layers: Número de camadas do transformer
            nhead: Número de heads de atenção
        """
        super(TimesFMModel, self).__init__()

        # Configurações
        self.d_model = d_model
        self.task_type = task_type
        self.out = out
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Embedding para valores temporais (agora mapeia de 1 valor para d_model)
        self.value_embedding = nn.Linear(1, d_model)

        # Embedding para posições temporais
        self.pos_embedding = nn.Embedding(in_channels, d_model)

        # Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Camada de saída
        if task_type == 'classification':
            self.linear = nn.Linear(d_model, num_classes * out)
        else:
            self.linear = nn.Linear(d_model, out)

    def forward(self, x, edge_index=None, edge_weight=None, explainer=False):
        """
        Args:
            x: Tensor de forma [num_cidades, in_channels]
            edge_index: Ignorado
            edge_weight: Ignorado
        """
        num_cities, seq_len = x.shape

        # Criar embeddings de posição [seq_len, d_model]
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)  # [seq_len, d_model]

        # Embedding dos valores [num_cidades, seq_len, 1] -> [num_cidades, seq_len, d_model]
        x = x.unsqueeze(-1)  # Adiciona dimensão de feature
        val_emb = self.value_embedding(x)

        # Combinar embeddings [num_cidades, seq_len, d_model]
        x = val_emb + pos_emb.unsqueeze(0)  # Broadcast das posições

        # Passar pelo transformer
        x = self.transformer_encoder(x)  # [num_cidades, seq_len, d_model]

        # Pegar última saída temporal
        last_out = x[:, -1, :]  # [num_cidades, d_model]

        # Camada linear final
        predictions = self.linear(last_out)

        # Processamento para classificação
        if self.task_type == 'classification':
            predictions = predictions.view(num_cities, self.out, self.num_classes)
            predictions = F.log_softmax(predictions, dim=-1)
            if explainer:
                predictions = predictions.squeeze(1)

        return predictions