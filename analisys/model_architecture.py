import torch
from torchviz import make_dot

# Cria um tensor de exemplo para passar pelo modelo, com o número de nós como dimensão
x = torch.randn(data.x.shape)
# Para gráficos, você não precisa usar `data.edge_index` aqui, mas sim o tensor `x`
y = model(x, data.edge_index)

# Gera o gráfico do modelo
make_dot(y, params=dict(model.named_parameters())).render("model_architecture", format="png")