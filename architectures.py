import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, NNConv, GINConv

class MLPClassifier_2(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3, norm="batch"):
        super().__init__()

        NormLayer = nn.BatchNorm1d if norm == "batch" else nn.LayerNorm

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            NormLayer(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            NormLayer(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(min(dropout * 1.5, 0.9)),

            nn.Linear(hidden_size // 2, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
    
class DualMLPClassifier(nn.Module):
    def __init__(self, fp_size, desc_size, hidden_size=128, desc_proj_size=32, dropout=0.3, norm="batch"):
        super().__init__()

        NormLayer = nn.BatchNorm1d if norm == "batch" else nn.LayerNorm
        
        # Branch 1: Fingerprint Head (keeps the bulk of the parameters)
        self.fp_head = nn.Sequential(
            nn.Linear(fp_size, hidden_size),
            NormLayer(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Branch 2: Descriptor Head (projects descriptors to a smaller space)
        self.desc_head = nn.Sequential(
            nn.Linear(desc_size, desc_proj_size),
            NormLayer(desc_proj_size),
            nn.ReLU()
        )

        # Joint Trunk: Takes concatenated (hidden_size + desc_proj_size)
        combined_size = hidden_size + desc_proj_size
        self.trunk = nn.Sequential(
            nn.Linear(combined_size, combined_size // 2),
            NormLayer(combined_size // 2),
            nn.ReLU(),
            nn.Dropout(min(dropout * 1.5, 0.9)),
            nn.Linear(combined_size // 2, 1)
        )

        # Weight Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fp, desc):
        x1 = self.fp_head(fp)
        x2 = self.desc_head(desc)
        
        # Concatenate side-by-side
        x = torch.cat((x1, x2), dim=1)
        return self.trunk(x)
    
class CNN1DClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # makes FC size independent of input length
        )

        self.fc = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)              # -> (batch, 1, features)
        x = self.conv_block(x)          # -> (batch, 128, 1)
        x = x.squeeze(-1)               # -> (batch, 128)
        return self.fc(x)
    
class CNN1DAttentionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, se_reduction=16):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Lightweight channel attention (SE block)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),              # (B, C, 1)
            nn.Flatten(),                          # (B, C)
            nn.Linear(128, 128 // se_reduction),
            nn.ReLU(),
            nn.Linear(128 // se_reduction, 128),
            nn.Sigmoid()
        )

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)        # (B, 1, L)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)         # (B, 128, L')

        # 🔹 Channel attention
        attn = self.se(x)         # (B, 128)
        attn = attn.unsqueeze(-1) # (B, 128, 1)
        x = x * attn              # reweight channels

        x = self.global_pool(x)   # (B, 128, 1)
        x = x.squeeze(-1)         # (B, 128)

        return self.fc(x)
    
class HybridGCN(nn.Module):
    def __init__(self, gcn_in_dim, gcn_hidden_dim=128, mlp_hidden=128, desc_dim=0, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(gcn_in_dim, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)
        self.dropout = dropout
        self.desc_dim = desc_dim
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden_dim + desc_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.ReLU(),
            nn.Dropout(min(dropout*1.5,0.9)),
            nn.Linear(mlp_hidden//2, 1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # descriptors: shape [num_graphs, desc_dim]
        if hasattr(data, "descriptors"):
            desc = data.descriptors
            # if desc is [desc_dim] per graph, make sure it's [num_graphs, desc_dim]
            if len(desc.shape) == 1:
                desc = desc.unsqueeze(0)
        else:
            desc = None

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Pooling
        x = global_mean_pool(x, batch)  # shape [batch_size, gcn_hidden_dim]

        # Concatenate descriptors if they exist
        if desc is not None:
            # Make sure desc is batched
            if desc.shape[0] != x.shape[0]:
                desc = desc.view(x.shape[0], -1)
            x = torch.cat([x, desc], dim=1)

        out = self.mlp(x)
        return out
    
class GCNClassifier(nn.Module):
    def __init__(self, node_feat_dim, hidden_size, dropout, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(node_feat_dim, hidden_size))
        self.bns.append(nn.BatchNorm1d(hidden_size))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_size, hidden_size))
            self.bns.append(nn.BatchNorm1d(hidden_size))

        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        return self.fc(x)
    
class GATClassifier(nn.Module):
    def __init__(self, num_node_features, hidden_dim, heads, dropout):
        super().__init__()

        self.gat1 = GATConv(
            num_node_features,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)

        return self.lin(x)
    
class MPNNClassifier(nn.Module):
    def __init__(self, node_feat_dim, hidden_size, dropout):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv2 = GINConv(nn2)

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        return self.head(x)
    
