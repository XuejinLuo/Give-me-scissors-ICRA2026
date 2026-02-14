import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attention_output)
        
        return output, attn_weights

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = Attention(d_model, n_heads)

    def forward(self, x, mask=None):
        return self.attention(x, x, x, mask)

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.attn = Attention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x

class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        self.layers = nn.ModuleList([SelfAttention(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GatedTimeSeriesAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(GatedTimeSeriesAttentionLayer, self).__init__()
        self.cross_attn = Attention(d_model, n_heads)
        self.self_attn = SelfAttentionLayer(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 门控机制：决定如何融合交叉注意力和自注意力
        self.cross_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        self.self_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, current_frame, previous_frames, mask=None):
        cross_attn_output, _ = self.cross_attn(current_frame, previous_frames, previous_frames, mask)
        self_attn_output, _ = self.self_attn(current_frame, mask)
        
        gate_input = torch.cat([cross_attn_output, self_attn_output], dim=-1)
        cross_weight = self.cross_gate(gate_input)
        self_weight = self.self_gate(gate_input)
        
        combined_output = cross_weight * cross_attn_output + self_weight * self_attn_output
        
        current_frame = self.layer_norm1(current_frame + self.dropout(combined_output))
        
        ffn_output = self.ffn(current_frame)
        current_frame = self.layer_norm2(current_frame + self.dropout(ffn_output))
        
        return current_frame


class S6(nn.Module):
    def __init__(self, batch_size, seq_len, d_model, state_size, device):
        super(S6, self).__init__()

        self.fc1 = nn.Linear(d_model, d_model, device=device)
        self.fc2 = nn.Linear(d_model, state_size, device=device)
        self.fc3 = nn.Linear(d_model, state_size, device=device)

        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size


        self.A = nn.Parameter(F.normalize(torch.ones(d_model, state_size, device=device), p=2, dim=-1))
        nn.init.xavier_uniform_(self.A)

        self.B = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)
        self.C = torch.zeros(batch_size, self.seq_len, self.state_size, device=device)

        self.delta = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)
        self.dA = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.dB = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)

        # h [batch_size, seq_len, d_model, state_size]
        self.h = torch.zeros(batch_size, self.seq_len, self.d_model, self.state_size, device=device)
        self.y = torch.zeros(batch_size, self.seq_len, self.d_model, device=device)


    def discretization(self):

        self.dB = torch.einsum("bld,bln->bldn", self.delta, self.B)

        self.dA = torch.exp(torch.einsum("bld,dn->bldn", self.delta, self.A))


        return self.dA, self.dB

    def forward(self, x):
        # Algorithm 2 MAMBA paper
        self.B = self.fc2(x)
        self.C = self.fc3(x)
        self.delta = F.softplus(self.fc1(x))

        self.discretization()
 
        # h [batch_size, seq_len, d_model, state_size]
        h = torch.zeros(x.size(0), self.seq_len, self.d_model, self.state_size, device=x.device)
        y = torch.zeros_like(x)

        h =  torch.einsum('bldn,bldn->bldn', self.dA, h) + rearrange(x, "b l d -> b l d 1") * self.dB

        # y [batch_size, seq_len, d_model]
        y = torch.einsum('bln,bldn->bld', self.C, h)

        return y

class Cross_attn(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, n_features, dropout=0.1):
        super(Cross_attn, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.s6_history = S6(batch_size=1, seq_len=4, d_model=d_model, state_size=256, device='cuda')
        self.cross_attention_layer = GatedTimeSeriesAttentionLayer(d_model, n_heads, d_ff, dropout)
        self.self_attention_layer = SelfAttentionEncoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x_embedded = self.embedding(x)
        current_frame = x_embedded[:, -1:, :]  # [batch_size, 1, d_model]
        previous_frames = x_embedded[:, :-1, :]  # [batch_size, 4, d_model]
        previous_frames = self.s6_history(previous_frames)

        current_frame = self.cross_attention_layer(current_frame, previous_frames)
        self_attn_output = self.self_attention_layer(current_frame)
        out = self.fc_out(self_attn_output.squeeze(1))
        
        return out

def create_model(args):
    model = Cross_attn(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        n_features=args.input_num
    )
    return model