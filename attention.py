import torch
import torch.nn as nn
import rpe_mechanisms

class MultiHeadAttentionParallel(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, distance_matrix, rpe_type):
        super().__init__()
        self.dim_head = hidden_dim//num_heads
        self.num_heads = num_heads
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc_q = nn.Linear(embedding_dim, hidden_dim)
        self.fc_k = nn.Linear(embedding_dim, hidden_dim)
        self.fc_v = nn.Linear(embedding_dim, hidden_dim)

        self.to_out = nn.Linear(hidden_dim, embedding_dim)

        # self.relative_k = GeneralLearnableFunctionParallel(self.dim_head)
        # self.relative_v = GeneralLearnableFunctionParallel(self.dim_head)
        self.relative_k = rpe_mechanisms.MonotonicallyDecreasingFunctionParallel(self.dim_head)
        self.relative_v = rpe_mechanisms.MonotonicallyDecreasingFunctionParallel(self.dim_head)
        #self.relative_k, self.relative_v = [num_patches, num_patches, dim_head]

        self.distance_matrix = distance_matrix

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        #x = [batch_size, num_patches, embedding_dim]
        
        x = self.norm(x)

        batch_size = x.shape[0]
        num_patches = x.shape[1]

        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        #q,k,v = [batch_size, num_patches, inner_dim]

        q = q.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        k = k.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        v = v.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        #q,k = [batch_size, num_heads, num_patches, dim_head]

        QKT = torch.matmul(q, k.permute(0, 1, 3, 2))
        #QKT = [batch_size, num_heads, num_patches, num_patches]

        #obtain relative positional embeddings
        relative_k = self.relative_k(self.distance_matrix)
        relative_v = self.relative_v(self.distance_matrix)
        #relative_k, relative_v = [num_patches, num_patches, dim_head]

        q_reshaped = q.view(num_patches, batch_size*self.num_heads, self.dim_head)
        #modified_q = [num_patches, batch_size*num_heads, dim_head]

        QAT = torch.matmul(q_reshaped, relative_k.permute(0, 2, 1))
        #QAT = [num_patches, batch_size*num_heads, num_patches]

        QAT = QAT.view(batch_size, self.num_heads, num_patches, num_patches)

        attn = (QKT + QAT) * self.scale
        attn = self.softmax(attn)
        #attn = [batch_size, num_heads, num_patches, num_patches]

        attn_V = torch.matmul(attn, v)
        #attn_V = [batch_size, num_heads, num_patches, dim_head]

        attn_reshaped = attn.view(num_patches, batch_size*self.num_heads, num_patches)
        attn_relative_v = torch.matmul(attn_reshaped, relative_v)
        #attn_relative_v = [num_patches, batch_size*num_heads, dim_head]

        attn_relative_v = attn_relative_v.view(batch_size, self.num_heads, num_patches, self.dim_head)

        out = attn_V + attn_relative_v
        out = out.view(batch_size, num_patches, self.hidden_dim)
        out = self.to_out(out)
        out = self.dropout(out)
        return out


class MultiHeadAttentionIndividual(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, distance_matrix, rpe_type):
        super().__init__()
        self.dim_head = hidden_dim//num_heads
        self.num_heads = num_heads
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.scale = self.dim_head ** -0.5
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc_q = nn.Linear(embedding_dim, hidden_dim)
        self.fc_k = nn.Linear(embedding_dim, hidden_dim)
        self.fc_v = nn.Linear(embedding_dim, hidden_dim)

        self.to_out = nn.Linear(hidden_dim, embedding_dim)

        self.relative_k = rpe_mechanisms.GeneralLearnableFunctionIndividual(self.dim_head, self.num_heads)
        self.relative_v = rpe_mechanisms.GeneralLearnableFunctionIndividual(self.dim_head, self.num_heads)
        #self.relative_k, self.relative_v = [num_patches, num_patches, dim_head]

        self.distance_matrix = distance_matrix

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x):
        #x = [batch_size, num_patches, embedding_dim]
        
        x = self.norm(x)

        batch_size = x.shape[0]
        num_patches = x.shape[1]

        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        #q,k,v = [batch_size, num_patches, inner_dim]

        q = q.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        k = k.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        v = v.view(x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        #q,k = [batch_size, num_heads, num_patches, dim_head]

        QKT = torch.matmul(q, k.permute(0, 1, 3, 2))
        #QKT = [batch_size, num_heads, num_patches, num_patches]

        #obtain relative positional embeddings that have an extra dimension for the number of heads
        relative_k = self.relative_k(self.distance_matrix)
        relative_v = self.relative_v(self.distance_matrix)
        #relative_k, relative_v = [num_heads, num_patches, num_patches, dim_head]

        q_reshaped = q.view(self.num_heads, num_patches, batch_size, self.dim_head)
        #modified_q = [num_heads, num_patches, batch_size, dim_head]

        QAT = torch.matmul(q_reshaped, relative_k.permute(0,1,3,2))
        #QAT = [num_heads, num_patches, batch_size, num_patches]

        QAT = QAT.view(batch_size, self.num_heads, num_patches, num_patches)

        attn = (QKT + QAT) * self.scale
        attn = self.softmax(attn)
        #attn = [batch_size, num_heads, num_patches, num_patches]

        attn_V = torch.matmul(attn, v)
        #attn_V = [batch_size, num_heads, num_patches, dim_head]

        attn_reshaped = attn.view(self.num_heads, num_patches, batch_size, num_patches)
        attn_relative_v = torch.matmul(attn_reshaped, relative_v)
        #attn_relative_v = [num_heads, num_patches, batch_size, dim_head]

        attn_relative_v = attn_relative_v.view(batch_size, self.num_heads, num_patches, self.dim_head)

        out = attn_V + attn_relative_v
        out = out.view(batch_size, num_patches, self.hidden_dim)
        out = self.to_out(out)
        out = self.dropout(out)
        return out