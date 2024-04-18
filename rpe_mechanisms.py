import torch
import torch.nn as nn

#general learnable function where weights are shared across all attention heads
class GeneralLearnableFunctionParallel(nn.Module):
    def __init__(self, embed_dim):
        super(GeneralLearnableFunctionParallel, self).__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Linear(1, embed_dim)

    def forward(self, distance_matrix):
        num_patches = distance_matrix.shape[0]
        distance_matrix = distance_matrix.view(-1, 1)
        embeddings = self.embeddings(distance_matrix)
        embeddings = embeddings.view(num_patches, num_patches, -1)

        return embeddings

#monotonically decreasing function where weights are shared across all attention heads
class MonotonicallyDecreasingFunctionParallel(nn.Module):
    def __init__(self, embed_dim):
        super(MonotonicallyDecreasingFunctionParallel, self).__init__()
        self.embed_dim = embed_dim
        self.a = nn.Parameter(torch.randn(embed_dim))

    def forward(self, distance_matrix):
        num_patches = distance_matrix.shape[0]
        distance_matrix = distance_matrix.view(-1, 1)
        embeddings = torch.exp(-torch.matmul(distance_matrix.unsqueeze(-1), self.a.unsqueeze(0)))
        embeddings = embeddings.view(num_patches, num_patches, self.embed_dim)
        return embeddings