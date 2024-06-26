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
    
#general learnable function where weights are individual for each attention head
class GeneralLearnableFunctionIndividual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GeneralLearnableFunctionIndividual, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embeddings = nn.Linear(1, embed_dim*num_heads)

    def forward(self, distance_matrix):
        num_patches = distance_matrix.shape[0]
        distance_matrix = distance_matrix.view(-1, 1)
        embeddings = self.embeddings(distance_matrix)
        embeddings = embeddings.view(self.num_heads, num_patches, num_patches, -1)
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
    
#monotonically decreasing function where weights are individual for each attention head
class MonotonicallyDecreasingFunctionIndividual(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MonotonicallyDecreasingFunctionIndividual, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.a = nn.Parameter(torch.randn(embed_dim*num_heads))

    def forward(self, distance_matrix):
        num_patches = distance_matrix.shape[0]
        distance_matrix = distance_matrix.view(-1, 1)
        embeddings = torch.exp(-torch.matmul(distance_matrix.unsqueeze(-1), self.a.unsqueeze(0)))
        embeddings = embeddings.view(self.num_heads, num_patches, num_patches, self.embed_dim)
        return embeddings

class RatioPolynomialsParallel(nn.Module):
    def __init__(self, embed_dim, max_degree = 5):
        super(RatioPolynomialsParallel, self).__init__()
        self.embed_dim = embed_dim
        self.max_degree = max_degree

        self.degree_f = nn.Parameter(torch.tensor(1.0))
        self.degree_g = nn.Parameter(torch.tensor(1.0))
        
        self.coeffs_f = nn.Parameter(torch.randn(self.max_degree + 1, embed_dim))
        
        self.coeffs_g = nn.Parameter(torch.randn(self.max_degree + 1, embed_dim))
    
    def forward(self, distance_matrix):
        num_patches = distance_matrix.shape[0]
        distance_matrix = distance_matrix.view(-1, 1)
        distance_matrix = distance_matrix.unsqueeze(-1)

        f_x = torch.zeros(num_patches * num_patches, self.embed_dim, dtype=distance_matrix.dtype, device=distance_matrix.device)
        for i in range(int(self.degree_f) + 1):
            term = self.coeffs_f[i].unsqueeze(0) * distance_matrix ** i
            f_x += term.squeeze(1)
        
        g_x = torch.zeros(num_patches * num_patches, self.embed_dim, dtype=distance_matrix.dtype, device=distance_matrix.device)
        for i in range(int(self.degree_g) + 1):
            term = self.coeffs_g[i].unsqueeze(0) * distance_matrix ** i
            g_x += term.squeeze(1)
        
        ratio = f_x / (g_x + 1e-8)

        embeddings = ratio.view(num_patches, num_patches, self.embed_dim)
        return embeddings

class RatioPolynomialsIndividual(nn.Module):
    def __init__(self, embed_dim, num_heads, max_degree = 5):
        super(RatioPolynomialsIndividual, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_degree = max_degree

        self.degree_f = nn.Parameter(torch.tensor(1.0))
        self.degree_g = nn.Parameter(torch.tensor(1.0))
        
        self.coeffs_f = nn.Parameter(torch.randn(self.max_degree + 1, embed_dim * num_heads))
        
        self.coeffs_g = nn.Parameter(torch.randn(self.max_degree + 1, embed_dim * num_heads))

    def forward(self, distance_matrix):
        num_patches = distance_matrix.shape[0]
        distance_matrix = distance_matrix.view(-1, 1)
        distance_matrix = distance_matrix.unsqueeze(-1)

        f_x = torch.zeros(num_patches * num_patches, self.embed_dim * self.num_heads, dtype=distance_matrix.dtype, device=distance_matrix.device)
        for i in range(int(self.degree_f) + 1):
            term = self.coeffs_f[i].unsqueeze(0) * distance_matrix ** i
            f_x += term.squeeze(1)
        
        g_x = torch.zeros(num_patches * num_patches, self.embed_dim * self.num_heads, dtype=distance_matrix.dtype, device=distance_matrix.device)
        for i in range(int(self.degree_g) + 1):
            term = self.coeffs_g[i].unsqueeze(0) * distance_matrix ** i
            g_x += term.squeeze(1)
        
        ratio = f_x / (g_x + 1e-8)

        embeddings = ratio.view(self.num_heads, num_patches, num_patches, self.embed_dim)
        return embeddings
