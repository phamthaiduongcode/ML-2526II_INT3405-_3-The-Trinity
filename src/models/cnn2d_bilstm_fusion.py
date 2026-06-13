
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, text, adj):
        support = torch.matmul(text, self.weight)
        output = torch.matmul(adj, support)
        return output

class DGCNNBranch(nn.Module):
    def __init__(self, num_nodes=32, in_features=5, hidden_dim=32, out_dim=64):
        super().__init__()
        self.adj = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        nn.init.uniform_(self.adj, 0.0, 1.0) 
        
        self.gc1 = GraphConvolution(in_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 128),
            nn.ELU(),
            nn.Linear(128, out_dim),
            nn.ELU()
        )

    def forward(self, x):
        adj_sym = F.relu(self.adj + self.adj.T) / 2.0
        rowsum = torch.sum(adj_sym, dim=1, keepdim=True) + 1e-6
        adj_norm = adj_sym / rowsum

        h = F.elu(self.gc1(x, adj_norm))
        h = F.elu(self.gc2(h, adj_norm))
        
        h = h.view(h.size(0), -1) 
        return self.fc(h)

class CNN2DBiLSTMFusion(nn.Module):
    def __init__(self, num_channels=32, freq_seq_dim=160, node_feat_dim=5, dropout_rate=0.5):
        super().__init__()
        
        self.b_temporal = nn.Sequential(nn.Conv2d(1, 16, (1, 65), padding=(0, 32), bias=False), nn.BatchNorm2d(16), nn.ELU())
        self.b_spatial = nn.Sequential(nn.Conv2d(16, 32, (num_channels, 1), bias=False), nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 4)))
        self.b_pointwise = nn.Sequential(nn.Conv2d(32, 64, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(64), nn.ELU(), nn.AvgPool2d((1, 2)))
        self.b_se = SEBlock(64, 4)
        self.branch_a_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.b_fc = nn.Linear(64, 128)
        
        self.lstm_hidden_dim = 64
        self.branch_b_bilstm = nn.LSTM(freq_seq_dim, self.lstm_hidden_dim, 1, batch_first=True, bidirectional=True)
        self.branch_b_fc = nn.Sequential(nn.Linear(self.lstm_hidden_dim * 2, 128), nn.ELU())
        
        self.branch_c_dgcnn = DGCNNBranch(num_nodes=num_channels, in_features=node_feat_dim, out_dim=64)
        
        self.fusion = nn.Sequential(nn.Linear(128 + 128 + 64, 128), nn.LayerNorm(128), nn.ELU(), nn.Dropout(dropout_rate))
        
        self.projection_head_v = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Linear(64, 64))
        self.projection_head_a = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Linear(64, 64))
        
        self.fc_valence = nn.Linear(128, 2)
        self.fc_arousal = nn.Linear(128, 2)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_raw, x_freq_seq, x_graph):
        if len(x_raw.shape) == 3: x_raw = x_raw.unsqueeze(1)
        x_a = self.b_fc(self.branch_a_pool(self.b_se(self.b_pointwise(self.b_spatial(self.b_temporal(x_raw))))).view(x_raw.size(0), -1))
        
        self.branch_b_bilstm.flatten_parameters()
        lstm_out, _ = self.branch_b_bilstm(x_freq_seq)
        
        fwd_last = lstm_out[:, -1, :self.lstm_hidden_dim]
        bwd_first = lstm_out[:, 0, self.lstm_hidden_dim:]
        feat_b = self.branch_b_fc(torch.cat([fwd_last, bwd_first], dim=-1))
        
        feat_c = self.branch_c_dgcnn(x_graph)
        
        fused_feat = self.fusion(torch.cat((x_a, feat_b, feat_c), dim=1))
        
        val_pred = self.fc_valence(fused_feat)
        aro_pred = self.fc_arousal(fused_feat)
        
        embed_v = F.normalize(self.projection_head_v(fused_feat), dim=1)
        embed_a = F.normalize(self.projection_head_a(fused_feat), dim=1)
        
        return val_pred, aro_pred, embed_v, embed_a

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = features.float()
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        return - mean_log_prob_pos.mean()